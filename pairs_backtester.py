import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup path to import UpstoxFetcher from nifty_mcp_upstox
UPSTOX_DIR = Path("E:/Trading_Agents/nifty_mcp_upstox")
if str(UPSTOX_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTOX_DIR))

try:
    from fetcher import UpstoxFetcher
    HAS_UPSTOX = True
except ImportError:
    HAS_UPSTOX = False

def run_pairs_backtest(df_a, df_b, symbol_a, symbol_b, window=50, entry_z=2.0, exit_z=0.0, cost_pct=0.0003):
    """
    Runs a market-neutral Pairs Trading backtest on two aligned stock price series.
    Spread = log(Price_A) - log(Price_B)
    """
    # Merge price data on timestamps to ensure alignment
    df_a = df_a[['timestamps', 'close']].rename(columns={'close': 'close_a'})
    df_b = df_b[['timestamps', 'close']].rename(columns={'close': 'close_b'})
    
    df = pd.merge(df_a, df_b, on='timestamps', how='inner')
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    if len(df) < window * 2:
        return None, []
        
    # Calculate Spread and Z-Score
    df['spread'] = np.log(df['close_a']) - np.log(df['close_b'])
    df['mean'] = df['spread'].rolling(window=window).mean()
    df['std'] = df['spread'].rolling(window=window).std(ddof=0)
    df['z'] = (df['spread'] - df['mean']) / df['std']
    
    df = df.dropna(subset=['z']).copy()
    
    position = 0 # 1 = Long Spread (Long A / Short B), -1 = Short Spread (Short A / Long B)
    positions = []
    
    # State Machine
    for idx, row in df.iterrows():
        z = row['z']
        
        if position == 0:
            if z < -entry_z:
                position = 1  # Buy Spread
            elif z > entry_z:
                position = -1 # Sell Spread
        else:
            # Exit at mean reversion
            if exit_z == 0.0:
                if (position == 1 and z >= 0) or (position == -1 and z <= 0):
                    position = 0
            else:
                if abs(z) < exit_z:
                    position = 0
                    
        positions.append(position)
        
    df['position'] = positions
    # Shift position by 1 bar to avoid look-ahead bias
    df['active_pos'] = df['position'].shift(1).fillna(0)
    
    # Calculate returns for both legs
    df['ret_a'] = df['close_a'].pct_change().fillna(0)
    df['ret_b'] = df['close_b'].pct_change().fillna(0)
    
    # Spread return = active_pos * (ret_a - ret_b)
    # Note: Long Spread = Long A + Short B. Net return = ret_a - ret_b.
    df['strategy_returns'] = df['active_pos'] * (df['ret_a'] - df['ret_b'])
    
    # Transaction costs apply to both legs whenever position changes
    # Position change triggers trades on BOTH stocks A and B.
    # Total cost = diff in active position * 2 * cost_pct (entry or exit for both legs)
    df['trade_trigger'] = df['active_pos'].diff().fillna(0).abs()
    df['costs'] = df['trade_trigger'] * 2 * cost_pct
    df['net_returns'] = df['strategy_returns'] - df['costs']
    
    df['equity_curve'] = (1 + df['net_returns']).cumprod()
    
    # Extract trades
    trades = []
    df['trade_id'] = (df['active_pos'] != df['active_pos'].shift()).cumsum()
    active_mask = df[df['active_pos'] != 0]
    
    if not active_mask.empty:
        for trade_id_val, group in active_mask.groupby('trade_id'):
            pos_type = "LONG_SPREAD" if group['active_pos'].iloc[0] > 0 else "SHORT_SPREAD"
            entry_idx = group.index[0]
            prev_idx = entry_idx - 1 if (entry_idx - 1) in df.index else entry_idx
            
            entry_price_a = df.loc[prev_idx, 'close_a']
            entry_price_b = df.loc[prev_idx, 'close_b']
            exit_price_a = group['close_a'].iloc[-1]
            exit_price_b = group['close_b'].iloc[-1]
            
            ret_a = (exit_price_a - entry_price_a) / entry_price_a
            ret_b = (exit_price_b - entry_price_b) / entry_price_b
            
            if pos_type == "LONG_SPREAD":
                raw_ret = ret_a - ret_b
            else:
                raw_ret = ret_b - ret_a
                
            net_ret = raw_ret - (4 * cost_pct) # 2 legs entry + 2 legs exit
            
            trades.append({
                "pair": f"{symbol_a}-{symbol_b}",
                "type": pos_type,
                "entry_time": df.loc[prev_idx, 'timestamps'],
                "exit_time": group['timestamps'].iloc[-1],
                "return_net": net_ret
            })
            
    return df, trades

def main():
    if not HAS_UPSTOX:
        print("ERROR: Could not import UpstoxFetcher.")
        sys.exit(1)
        
    fetcher = UpstoxFetcher()
    backtest_days = 10
    cost_pct = 0.0003
    
    # We will test nine standard cointegrated stock pairs
    pairs_to_test = [
        ("PFC", "REC", "NSE_EQ|INE134E01011", "NSE_EQ|INE020B01018"),
        ("TCS", "INFY", "NSE_EQ|INE467B01029", "NSE_EQ|INE009A01021"),
        ("ICICIBANK", "HDFCBANK", "NSE_EQ|INE090A01021", "NSE_EQ|INE040A01034"),
        ("TATASTEEL", "JSWSTEEL", "NSE_EQ|INE081A01020", "NSE_EQ|INE019A01038"),
        ("BANKBARODA", "CANBK", "NSE_EQ|INE028A01039", "NSE_EQ|INE476A01022"),
        ("NTPC", "POWERGRID", "NSE_EQ|INE733E01010", "NSE_EQ|INE752E01010"),
        ("HEROMOTOCO", "BAJAJ-AUTO", "NSE_EQ|INE158A01026", "NSE_EQ|INE917I01010"),
        ("AXISBANK", "KOTAKBANK", "NSE_EQ|INE238A01034", "NSE_EQ|INE237A01036"),
        ("ACC", "AMBUJACEM", "NSE_EQ|INE012A01025", "NSE_EQ|INE079A01024")
    ]
    
    print("=== Fetching Intraday Data for Pairs Trading ===")
    
    all_pairs_returns = []
    all_trades = []
    
    for symbol_a, symbol_b, key_a, key_b in pairs_to_test:
        print(f"\nFetching & Backtesting Pair: {symbol_a} - {symbol_b}...")
        try:
            df_a = fetcher.get_historical(key_a, "5minute", days=backtest_days)
            df_b = fetcher.get_historical(key_b, "5minute", days=backtest_days)
            
            if df_a.empty or df_b.empty:
                print(f"  WARNING: Failed to fetch data for one of the legs.")
                continue
                
            df_sim, trades = run_pairs_backtest(df_a, df_b, symbol_a, symbol_b, window=50, entry_z=2.0, exit_z=0.0, cost_pct=cost_pct)
            
            if df_sim is not None:
                returns_series = df_sim[['timestamps', 'net_returns']].rename(columns={'net_returns': f"{symbol_a}-{symbol_b}"})
                returns_series.set_index('timestamps', inplace=True)
                all_pairs_returns.append(returns_series)
                all_trades.extend(trades)
                
                # Metrics for this pair
                cum_ret = df_sim['equity_curve'].iloc[-1] - 1
                print(f"  Completed. Trades: {len(trades)} | Net Return: {cum_ret:+.2%}")
            
            time.sleep(0.2)
        except Exception as e:
            print(f"  ERROR: {e}")
            
    if not all_pairs_returns:
        print("ERROR: No pairs backtest data generated. Exiting.")
        sys.exit(1)
        
    # Aggregate Portfolio Returns
    print("\n=== Aggregating Pairs Portfolio ===")
    portfolio_df = pd.concat(all_pairs_returns, axis=1).fillna(0.0)
    portfolio_df['portfolio_return'] = portfolio_df.mean(axis=1)
    portfolio_df['equity_curve'] = (1 + portfolio_df['portfolio_return']).cumprod()
    
    total_net_return = portfolio_df['equity_curve'].iloc[-1] - 1
    max_dd = (portfolio_df['equity_curve'] / portfolio_df['equity_curve'].cummax() - 1).min()
    
    num_trades = len(all_trades)
    if num_trades > 0:
        winning_trades = sum(1 for t in all_trades if t['return_net'] > 0)
        win_rate = winning_trades / num_trades
        avg_trade_ret = np.mean([t['return_net'] for t in all_trades])
    else:
        win_rate = 0.0
        avg_trade_ret = 0.0
        
    ret_std = portfolio_df['portfolio_return'].std()
    sharpe = (portfolio_df['portfolio_return'].mean() / ret_std * np.sqrt(75 * 252)) if ret_std > 0 else 0.0
    
    print("\n==================================================")
    print("        PAIRS PORTFOLIO PERFORMANCE RESULTS       ")
    print("==================================================")
    print(f" Total Net Return (After Costs): {total_net_return:+.2%}")
    print(f" Maximum Drawdown              : {max_dd:.2%}")
    print(f" Annualized Sharpe Ratio       : {sharpe:.2f}")
    print(f" Total Executed Trades         : {num_trades} (across {len(pairs_to_test)} pairs)")
    print(f" Win Rate                      : {win_rate:.2%}")
    print(f" Average Profit per Trade      : {avg_trade_ret:+.3%}")
    print("==================================================")
    
    # Save plot
    plt.style.use('dark_background' if plt.style.available else 'default')
    plt.figure(figsize=(12, 6))
    
    portfolio_df.reset_index(inplace=True)
    plt.plot(portfolio_df['timestamps'], portfolio_df['equity_curve'], label='Pairs Portfolio Net Equity (Equal Weight)', color='#00FFFF', linewidth=2)
    plt.title(f"Pairs Trading Portfolio Performance (PFC-REC, TCS-INFY, ICICI-HDFC)", fontsize=12, fontweight='bold')
    plt.ylabel('Growth of 1 Unit')
    plt.grid(True, alpha=0.15)
    plt.legend()
    plt.tight_layout()
    
    plot_path = Path("pairs_performance.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved performance plot to {plot_path.resolve()}")

if __name__ == "__main__":
    main()
