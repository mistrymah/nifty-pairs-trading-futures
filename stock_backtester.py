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

def run_stock_backtest(df, window=20, entry_z=2.0, exit_z=0.2, stop_loss_z=3.5, cost_pct=0.0003):
    """
    Runs a Z-Score Mean Reversion backtest on a single stock's intraday data.
    Returns:
    - df_sim: DataFrame with calculated returns
    - trades: List of individual trades
    """
    df_sim = df.copy()
    
    # Calculate Z-Score indicators
    df_sim['mean'] = df_sim['close'].rolling(window=window, min_periods=window).mean()
    df_sim['std'] = df_sim['close'].rolling(window=window, min_periods=window).std(ddof=0)
    df_sim['z'] = (df_sim['close'] - df_sim['mean']) / df_sim['std']
    df_sim['ema_trend'] = df_sim['close'].ewm(span=200, adjust=False).mean()
    
    df_sim = df_sim.dropna(subset=['z']).copy()
    if df_sim.empty:
        return None, []
        
    position = 0
    positions = []
    
    # State Machine Signal Generation
    for idx, row in df_sim.iterrows():
        z = row['z']
        price = row['close']
        
        # Stop-Loss Check
        if position == 1 and stop_loss_z is not None and z < -stop_loss_z:
            position = 0  # Stop loss hit on Long
        elif position == -1 and stop_loss_z is not None and z > stop_loss_z:
            position = 0  # Stop loss hit on Short
            
        if position == 0:
            if z < -entry_z:
                # Buy Long if price is above trend EMA
                if price > row['ema_trend']:
                    position = 1
            elif z > entry_z:
                # Sell Short if price is below trend EMA
                if price < row['ema_trend']:
                    position = -1
        else:
            # Normal Exit
            if exit_z == 0.0:
                if (position == 1 and z >= 0) or (position == -1 and z <= 0):
                    position = 0
            else:
                if abs(z) < exit_z:
                    position = 0
                    
        positions.append(position)
        
    df_sim['signal'] = positions
    
    # Shift position to prevent look-ahead bias (trade at next bar's open/returns)
    df_sim['position'] = df_sim['signal'].shift(1).fillna(0)
    df_sim['market_returns'] = df_sim['close'].pct_change().fillna(0)
    df_sim['strategy_returns'] = df_sim['position'] * df_sim['market_returns']
    
    # Apply transaction costs
    df_sim['trade_trigger'] = df_sim['position'].diff().fillna(0).abs()
    df_sim['costs'] = df_sim['trade_trigger'] * cost_pct
    df_sim['net_returns'] = df_sim['strategy_returns'] - df_sim['costs']
    
    # Identify individual trades
    df_sim['trade_id'] = (df_sim['position'] != df_sim['position'].shift()).cumsum()
    trades = []
    active_positions = df_sim[df_sim['position'] != 0]
    
    if not active_positions.empty:
        for trade_id, group in active_positions.groupby('trade_id'):
            pos_type = "LONG" if group['position'].iloc[0] > 0 else "SHORT"
            entry_idx = group.index[0]
            prev_idx = entry_idx - 1 if (entry_idx - 1) in df_sim.index else entry_idx
            entry_price = df_sim.loc[prev_idx, 'close']
            exit_price = group['close'].iloc[-1]
            
            if pos_type == "LONG":
                trade_ret = (exit_price - entry_price) / entry_price
            else:
                trade_ret = (entry_price - exit_price) / entry_price
                
            trade_ret_net = trade_ret - (2 * cost_pct)
            
            trades.append({
                "type": pos_type,
                "entry_time": df_sim.loc[prev_idx, 'timestamps'],
                "exit_time": group['timestamps'].iloc[-1],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_raw": trade_ret,
                "return_net": trade_ret_net
            })
            
    return df_sim, trades

def main():
    if not HAS_UPSTOX:
        print("ERROR: Could not import UpstoxFetcher.")
        sys.exit(1)
        
    # 1. Load selected stocks
    selected_path = Path("selected_stocks.json")
    if not selected_path.exists():
        print("ERROR: selected_stocks.json not found. Please run stock_scanner.py first.")
        sys.exit(1)
        
    with open(selected_path) as f:
        selected_stocks = json.load(f)
        
    print(f"Loaded {len(selected_stocks)} selected stocks for backtesting.")
    
    fetcher = UpstoxFetcher()
    backtest_days = 10  # Backtest over 10 days of intraday data
    cost_pct = 0.0003  # 0.03% execution cost
    
    all_stock_returns = []
    all_trades = []
    
    print("\n=== Fetching Intraday 5m Candles & Simulating Trades ===")
    for idx, item in enumerate(selected_stocks, 1):
        symbol = item['symbol']
        key = item['key']
        
        print(f"[{idx}/{len(selected_stocks)}] Backtesting {symbol}...")
        try:
            df = fetcher.get_historical(key, "5minute", days=backtest_days)
            if df.empty or len(df) < 100:
                print(f"  WARNING: Insufficient intraday data for {symbol}.")
                continue
                
            df = df.sort_values("timestamps").reset_index(drop=True)
            df_sim, trades = run_stock_backtest(df, window=20, entry_z=2.0, exit_z=0.2, stop_loss_z=3.5, cost_pct=cost_pct)
            
            if df_sim is not None:
                # Store net returns series with timestamp
                returns_series = df_sim[['timestamps', 'net_returns']].rename(columns={'net_returns': symbol})
                returns_series.set_index('timestamps', inplace=True)
                all_stock_returns.append(returns_series)
                
                # Tag trades with symbol
                for t in trades:
                    t['symbol'] = symbol
                    all_trades.append(t)
                    
                print(f"  Completed. Trades: {len(trades)} | Net Return: {df_sim['net_returns'].sum():+.2%}")
            
            # Anti-rate limit delay
            time.sleep(0.15)
        except Exception as e:
            print(f"  ERROR backtesting {symbol}: {e}")
            
    if not all_stock_returns:
        print("ERROR: No backtest data generated. Exiting.")
        sys.exit(1)
        
    # 2. Aggregate Portfolio Returns
    print("\n=== Aggregating Portfolio Performance ===")
    # Merge all asset returns on timestamps
    portfolio_df = pd.concat(all_stock_returns, axis=1).fillna(0.0)
    
    # Portfolio return is the average of all assets' returns at each bar
    # This represents equal allocation across all 50 stocks (rebalanced bar-by-bar)
    portfolio_df['portfolio_return'] = portfolio_df.mean(axis=1)
    portfolio_df['equity_curve'] = (1 + portfolio_df['portfolio_return']).cumprod()
    
    # Compute market baseline for comparison (we use Nifty spot as market representation)
    print("Fetching Nifty 50 Spot daily baseline...")
    try:
        nifty_df = fetcher.get_nifty_historical("5minute", days=backtest_days)
        nifty_df = nifty_df.sort_values("timestamps").reset_index(drop=True)
        nifty_df['market_return'] = nifty_df['close'].pct_change().fillna(0.0)
        nifty_df['market_equity'] = (1 + nifty_df['market_return']).cumprod()
        nifty_df.set_index('timestamps', inplace=True)
        
        portfolio_df = portfolio_df.join(nifty_df['market_equity'], how='left').fillna(method='ffill').fillna(1.0)
    except Exception as e:
        print(f"Could not load Nifty baseline: {e}")
        portfolio_df['market_equity'] = 1.0
        
    # 3. Calculate Performance Metrics
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
        
    # Approximate Sharpe (scale 5-minute bars to annualized)
    bars_per_day = 75
    ret_std = portfolio_df['portfolio_return'].std()
    sharpe = (portfolio_df['portfolio_return'].mean() / ret_std * np.sqrt(bars_per_day * 252)) if ret_std > 0 else 0.0
    
    print("\n==================================================")
    print("      PORTFOLIO BACKTEST PERFORMANCE RESULTS      ")
    print("==================================================")
    print(f" Total Net Return (After Costs): {total_net_return:+.2%}")
    print(f" Maximum Portfolio Drawdown    : {max_dd:.2%}")
    print(f" Annualized Portfolio Sharpe   : {sharpe:.2f}")
    print(f" Total Executed Trades         : {num_trades} (across {len(selected_stocks)} assets)")
    print(f" Portfolio Win Rate            : {win_rate:.2%}")
    print(f" Average Profit per Trade      : {avg_trade_ret:+.3%}")
    print("==================================================")
    
    # Save portfolio performance stats to JSON
    stats = {
        "net_return": total_net_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "total_trades": num_trades,
        "win_rate": win_rate,
        "avg_profit_per_trade": avg_trade_ret
    }
    with open("portfolio_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    # 4. Generate Plot & Save
    plt.style.use('dark_background' if plt.style.available else 'default')
    plt.figure(figsize=(12, 6))
    
    portfolio_df.reset_index(inplace=True)
    
    plt.plot(portfolio_df['timestamps'], portfolio_df['equity_curve'], label='Z-Score Mean Reversion Portfolio (Net)', color='#00FFFF', linewidth=2)
    if 'market_equity' in portfolio_df.columns:
        plt.plot(portfolio_df['timestamps'], portfolio_df['market_equity'], label='Nifty 50 (Buy & Hold)', color='white', linestyle='--', alpha=0.6)
        
    plt.title(f"Intraday Multi-Stock Portfolio Performance (Top 50 Volume & ATR)", fontsize=12, fontweight='bold')
    plt.ylabel('Growth of 1 Unit')
    plt.grid(True, alpha=0.15)
    plt.legend()
    plt.tight_layout()
    
    plot_path = Path("portfolio_performance.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Saved performance curve to {plot_path.resolve()}")

if __name__ == "__main__":
    main()
