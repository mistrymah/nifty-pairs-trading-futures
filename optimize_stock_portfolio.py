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

def run_simulation(df, window, entry_z, exit_z, stop_loss_z=3.5, use_trend_filter=True, cost_pct=0.0003):
    df_sim = df.copy()
    
    # Calculate Z-Score indicators
    df_sim['mean'] = df_sim['close'].rolling(window=window, min_periods=window).mean()
    df_sim['std'] = df_sim['close'].rolling(window=window, min_periods=window).std(ddof=0)
    df_sim['z'] = (df_sim['close'] - df_sim['mean']) / df_sim['std']
    
    if use_trend_filter:
        df_sim['ema_trend'] = df_sim['close'].ewm(span=200, adjust=False).mean()
        
    df_sim = df_sim.dropna(subset=['z']).copy()
    if df_sim.empty:
        return None, 0
        
    position = 0
    positions = []
    
    for idx, row in df_sim.iterrows():
        z = row['z']
        price = row['close']
        
        # Stop-Loss Check
        if position == 1 and stop_loss_z is not None and z < -stop_loss_z:
            position = 0
        elif position == -1 and stop_loss_z is not None and z > stop_loss_z:
            position = 0
            
        if position == 0:
            if z < -entry_z:
                if use_trend_filter:
                    if price > row['ema_trend']:
                        position = 1
                else:
                    position = 1
            elif z > entry_z:
                if use_trend_filter:
                    if price < row['ema_trend']:
                        position = -1
                else:
                    position = -1
        else:
            # Exit
            if exit_z == 0.0:
                if (position == 1 and z >= 0) or (position == -1 and z <= 0):
                    position = 0
            else:
                if abs(z) < exit_z:
                    position = 0
                
        positions.append(position)
        
    df_sim['signal'] = positions
    df_sim['position'] = df_sim['signal'].shift(1).fillna(0)
    df_sim['market_returns'] = df_sim['close'].pct_change().fillna(0)
    df_sim['strategy_returns'] = df_sim['position'] * df_sim['market_returns']
    
    df_sim['trade_trigger'] = df_sim['position'].diff().fillna(0).abs()
    df_sim['costs'] = df_sim['trade_trigger'] * cost_pct
    df_sim['net_returns'] = df_sim['strategy_returns'] - df_sim['costs']
    
    # Identify trades count
    trade_count = df_sim['trade_trigger'].sum() / 2
    
    return df_sim['net_returns'], trade_count

def main():
    if not HAS_UPSTOX:
        print("ERROR: Could not import UpstoxFetcher.")
        sys.exit(1)
        
    # Load selected stocks
    selected_path = Path("selected_stocks.json")
    if not selected_path.exists():
        print("ERROR: selected_stocks.json not found.")
        sys.exit(1)
        
    with open(selected_path) as f:
        selected_stocks = json.load(f)
        
    fetcher = UpstoxFetcher()
    backtest_days = 10
    cost_pct = 0.0003
    
    # Prefetch all dataframes
    data_cache = {}
    print("=== Prefetching 10 days of 5m Candles for Portfolio ===")
    for idx, item in enumerate(selected_stocks, 1):
        symbol = item['symbol']
        key = item['key']
        try:
            df = fetcher.get_historical(key, "5minute", days=backtest_days)
            if not df.empty and len(df) >= 150:
                data_cache[symbol] = df.sort_values("timestamps").reset_index(drop=True)
            time.sleep(0.12)
        except Exception as e:
            print(f"Error prefetching {symbol}: {e}")
            
    if not data_cache:
        print("ERROR: No data pre-fetched.")
        sys.exit(1)
        
    print(f"\nSuccessfully cached data for {len(data_cache)} stocks.")
    
    # Parameter Sweep Space
    windows = [20, 50]
    entry_zs = [2.0, 2.5, 3.0]
    exit_zs = [0.0, 0.2, 0.5]
    trend_filters = [False, True]
    
    combinations = list(itertools.product(windows, entry_zs, exit_zs, trend_filters))
    results = []
    
    print(f"\nRunning Optimization Sweep over {len(combinations)} combinations...")
    for idx, (window, entry_z, exit_z, use_trend) in enumerate(combinations, 1):
        portfolio_returns = []
        total_trades = 0
        
        for symbol, df in data_cache.items():
            net_ret_series, trade_count = run_simulation(df, window, entry_z, exit_z, stop_loss_z=3.5, use_trend_filter=use_trend, cost_pct=cost_pct)
            if net_ret_series is not None:
                # Rename series to symbol to align
                portfolio_returns.append(net_ret_series.to_frame(name=symbol))
                total_trades += trade_count
                
        if not portfolio_returns:
            continue
            
        # Merge portfolio returns and calculate portfolio aggregate return series
        portfolio_df = pd.concat(portfolio_returns, axis=1).fillna(0.0)
        portfolio_df['portfolio_return'] = portfolio_df.mean(axis=1)
        portfolio_df['equity'] = (1 + portfolio_df['portfolio_return']).cumprod()
        
        net_return = portfolio_df['equity'].iloc[-1] - 1
        max_dd = (portfolio_df['equity'] / portfolio_df['equity'].cummax() - 1).min()
        
        # Calculate Sharpe
        ret_std = portfolio_df['portfolio_return'].std()
        sharpe = (portfolio_df['portfolio_return'].mean() / ret_std * np.sqrt(75 * 252)) if ret_std > 0 else 0.0
        
        results.append({
            'window': window,
            'entry_z': entry_z,
            'exit_z': exit_z,
            'use_trend': use_trend,
            'net_return': net_return,
            'max_drawdown': max_dd,
            'total_trades': total_trades,
            'sharpe': sharpe
        })
        
        if idx % 10 == 0 or idx == len(combinations):
            print(f"  Completed {idx}/{len(combinations)} combinations...")
            
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No combinations tested.")
        return
        
    res_df = res_df.sort_values(by='net_return', ascending=False)
    
    print("\n=== TOP 5 OPTIMIZED PORTFOLIO CONFIGURATIONS ===")
    for rank, (i, row) in enumerate(res_df.head(5).iterrows(), 1):
        print(f"Rank {rank}:")
        print(f"  Params: Window={row['window']}, Entry Z={row['entry_z']}, Exit Z={row['exit_z']}, Trend={row['use_trend']}")
        print(f"  Metrics: Net Return={row['net_return']:.2%}, Max DD={row['max_drawdown']:.2%}, Total Trades={int(row['total_trades'])}, Sharpe={row['sharpe']:.2f}\n")
        
    # Re-run best configuration to plot it
    best = res_df.iloc[0]
    print(f"Generating performance curve plot for best setup (Net Return: {best['net_return']:.2%})...")
    
    portfolio_returns = []
    for symbol, df in data_cache.items():
        net_ret_series, _ = run_simulation(df, int(best['window']), best['entry_z'], best['exit_z'], stop_loss_z=3.5, use_trend_filter=best['use_trend'], cost_pct=cost_pct)
        if net_ret_series is not None:
            portfolio_returns.append(net_ret_series.to_frame(name=symbol))
            
    portfolio_df = pd.concat(portfolio_returns, axis=1).fillna(0.0)
    portfolio_df['portfolio_return'] = portfolio_df.mean(axis=1)
    portfolio_df['equity_curve'] = (1 + portfolio_df['portfolio_return']).cumprod()
    
    # Save best plot
    plt.style.use('dark_background' if plt.style.available else 'default')
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_df['equity_curve'], label='Optimized Stock Portfolio (Net)', color='#00FFFF', linewidth=2)
    plt.title(f"Optimized Multi-Stock Intraday Portfolio (Return={best['net_return']:.2%}, Max DD={best['max_drawdown']:.2%})", fontsize=12, fontweight='bold')
    plt.ylabel('Growth of 1 Unit')
    plt.grid(True, alpha=0.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig('portfolio_performance_optimized.png', dpi=150)
    plt.close()
    print("Saved optimized portfolio plot to portfolio_performance_optimized.png")

if __name__ == "__main__":
    main()
