import os
import sys
import json
import itertools
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

def run_pairs_simulation(df_a, df_b, window, entry_z, exit_z, cost_pct=0.0003):
    df_a_clean = df_a[['timestamps', 'close']].rename(columns={'close': 'close_a'})
    df_b_clean = df_b[['timestamps', 'close']].rename(columns={'close': 'close_b'})
    
    df = pd.merge(df_a_clean, df_b_clean, on='timestamps', how='inner')
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    if len(df) < window * 2:
        return None, 0
        
    df['spread'] = np.log(df['close_a']) - np.log(df['close_b'])
    df['mean'] = df['spread'].rolling(window=window).mean()
    df['std'] = df['spread'].rolling(window=window).std(ddof=0)
    df['z'] = (df['spread'] - df['mean']) / df['std']
    
    df = df.dropna(subset=['z']).copy()
    if df.empty:
        return None, 0
        
    position = 0
    positions = []
    
    for idx, row in df.iterrows():
        z = row['z']
        
        if position == 0:
            if z < -entry_z:
                position = 1
            elif z > entry_z:
                position = -1
        else:
            if exit_z == 0.0:
                if (position == 1 and z >= 0) or (position == -1 and z <= 0):
                    position = 0
            else:
                if abs(z) < exit_z:
                    position = 0
                    
        positions.append(position)
        
    df['position'] = positions
    df['active_pos'] = df['position'].shift(1).fillna(0)
    
    df['ret_a'] = df['close_a'].pct_change().fillna(0)
    df['ret_b'] = df['close_b'].pct_change().fillna(0)
    df['strategy_returns'] = df['active_pos'] * (df['ret_a'] - df['ret_b'])
    
    df['trade_trigger'] = df['active_pos'].diff().fillna(0).abs()
    df['costs'] = df['trade_trigger'] * 2 * cost_pct
    df['net_returns'] = df['strategy_returns'] - df['costs']
    
    trade_count = df['trade_trigger'].sum() / 2
    
    return df['net_returns'], trade_count

def main():
    if not HAS_UPSTOX:
        print("ERROR: Could not import UpstoxFetcher.")
        sys.exit(1)
        
    fetcher = UpstoxFetcher()
    backtest_days = 10
    cost_pct = 0.0003
    
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
    
    # Prefetch all dataframes
    data_cache = {}
    print("=== Prefetching Intraday Data for 9 Pairs ===")
    for symbol_a, symbol_b, key_a, key_b in pairs_to_test:
        try:
            df_a = fetcher.get_historical(key_a, "5minute", days=backtest_days)
            df_b = fetcher.get_historical(key_b, "5minute", days=backtest_days)
            
            if not df_a.empty and not df_b.empty:
                data_cache[f"{symbol_a}-{symbol_b}"] = (
                    df_a.sort_values("timestamps").reset_index(drop=True),
                    df_b.sort_values("timestamps").reset_index(drop=True)
                )
            time.sleep(0.15)
        except Exception as e:
            print(f"Error fetching {symbol_a}-{symbol_b}: {e}")
            
    if not data_cache:
        print("ERROR: No data pre-fetched.")
        sys.exit(1)
        
    # Parameter Sweep Space
    windows = [20, 50, 100]
    entry_zs = [1.8, 2.0, 2.5]
    exit_zs = [0.0, 0.2, 0.5]
    
    combinations = list(itertools.product(windows, entry_zs, exit_zs))
    results = []
    
    print(f"\nRunning Optimization Sweep over {len(combinations)} combinations...")
    for idx, (window, entry_z, exit_z) in enumerate(combinations, 1):
        portfolio_returns = []
        total_trades = 0
        
        for pair_name, (df_a, df_b) in data_cache.items():
            net_ret_series, trade_count = run_pairs_simulation(df_a, df_b, window, entry_z, exit_z, cost_pct=cost_pct)
            if net_ret_series is not None:
                portfolio_returns.append(net_ret_series.to_frame(name=pair_name))
                total_trades += trade_count
                
        if not portfolio_returns:
            continue
            
        portfolio_df = pd.concat(portfolio_returns, axis=1).fillna(0.0)
        portfolio_df['portfolio_return'] = portfolio_df.mean(axis=1)
        portfolio_df['equity'] = (1 + portfolio_df['portfolio_return']).cumprod()
        
        net_return = portfolio_df['equity'].iloc[-1] - 1
        max_dd = (portfolio_df['equity'] / portfolio_df['equity'].cummax() - 1).min()
        
        ret_std = portfolio_df['portfolio_return'].std()
        sharpe = (portfolio_df['portfolio_return'].mean() / ret_std * np.sqrt(75 * 252)) if ret_std > 0 else 0.0
        
        results.append({
            'window': window,
            'entry_z': entry_z,
            'exit_z': exit_z,
            'net_return': net_return,
            'max_drawdown': max_dd,
            'total_trades': total_trades,
            'sharpe': sharpe
        })
        
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No combinations tested.")
        return
        
    res_df = res_df.sort_values(by='net_return', ascending=False)
    
    print("\n=== TOP 5 OPTIMIZED PAIRS PORTFOLIO CONFIGURATIONS ===")
    for rank, (i, row) in enumerate(res_df.head(5).iterrows(), 1):
        print(f"Rank {rank}:")
        print(f"  Params: Window={row['window']}, Entry Z={row['entry_z']}, Exit Z={row['exit_z']}")
        print(f"  Metrics: Net Return={row['net_return']:.2%}, Max DD={row['max_drawdown']:.2%}, Total Trades={int(row['total_trades'])}, Sharpe={row['sharpe']:.2f}\n")
        
    # Re-run best configuration to plot it
    best = res_df.iloc[0]
    print(f"Generating performance curve plot for best setup (Net Return: {best['net_return']:.2%})...")
    
    portfolio_returns = []
    for pair_name, (df_a, df_b) in data_cache.items():
        net_ret_series, _ = run_pairs_simulation(df_a, df_b, int(best['window']), best['entry_z'], best['exit_z'], cost_pct=cost_pct)
        if net_ret_series is not None:
            portfolio_returns.append(net_ret_series.to_frame(name=pair_name))
            
    portfolio_df = pd.concat(portfolio_returns, axis=1).fillna(0.0)
    portfolio_df['portfolio_return'] = portfolio_df.mean(axis=1)
    portfolio_df['equity_curve'] = (1 + portfolio_df['portfolio_return']).cumprod()
    
    plt.style.use('dark_background' if plt.style.available else 'default')
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_df['equity_curve'], label='Optimized 9-Pair Portfolio (Net)', color='#00FFFF', linewidth=2)
    plt.title(f"Optimized Multi-Sector Pairs Trading Portfolio (Return={best['net_return']:.2%}, Max DD={best['max_drawdown']:.2%})", fontsize=12, fontweight='bold')
    plt.ylabel('Growth of 1 Unit')
    plt.grid(True, alpha=0.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pairs_performance_optimized.png', dpi=150)
    plt.close()
    print("Saved optimized pairs plot to pairs_performance_optimized.png")

if __name__ == "__main__":
    main()
