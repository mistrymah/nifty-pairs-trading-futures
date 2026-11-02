import os
import sys
import itertools
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

def run_simulation(df, window, entry_z, exit_z, stop_loss_z=None, use_trend_filter=False, cost_pct=0.0003):
    """
    Runs a single strategy simulation and returns performance metrics.
    """
    df_sim = df.copy()
    
    # Calculate Z-Score indicators
    df_sim['mean'] = df_sim['close'].rolling(window=window, min_periods=window).mean()
    df_sim['std'] = df_sim['close'].rolling(window=window, min_periods=window).std(ddof=0)
    df_sim['z'] = (df_sim['close'] - df_sim['mean']) / df_sim['std']
    
    if use_trend_filter:
        df_sim['ema_trend'] = df_sim['close'].ewm(span=200, adjust=False).mean()
        
    df_sim = df_sim.dropna(subset=['z']).copy()
    
    if df_sim.empty:
        return None
        
    # State Machine Signal Generation
    position = 0
    positions = []
    
    for idx, row in df_sim.iterrows():
        z = row['z']
        price = row['close']
        
        # Stop-Loss Check first if we are in a position
        if position == 1 and stop_loss_z is not None and z < -stop_loss_z:
            position = 0  # Stop loss hit on Long
        elif position == -1 and stop_loss_z is not None and z > stop_loss_z:
            position = 0  # Stop loss hit on Short
            
        if position == 0:
            if z < -entry_z:
                # Long entry
                if use_trend_filter:
                    # Only buy if price is above trend EMA
                    if price > row['ema_trend']:
                        position = 1
                else:
                    position = 1
            elif z > entry_z:
                # Short entry
                if use_trend_filter:
                    # Only sell if price is below trend EMA
                    if price < row['ema_trend']:
                        position = -1
                else:
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
    
    # Shift position to prevent look-ahead bias
    df_sim['position'] = df_sim['signal'].shift(1).fillna(0)
    df_sim['market_returns'] = df_sim['close'].pct_change().fillna(0)
    df_sim['strategy_returns'] = df_sim['position'] * df_sim['market_returns']
    
    # Calculate costs
    df_sim['trade_trigger'] = df_sim['position'].diff().fillna(0).abs()
    df_sim['costs'] = df_sim['trade_trigger'] * cost_pct
    df_sim['net_returns'] = df_sim['strategy_returns'] - df_sim['costs']
    
    # Performance curves
    df_sim['equity_market'] = (1 + df_sim['market_returns']).cumprod()
    df_sim['equity_strategy_net'] = (1 + df_sim['net_returns']).cumprod()
    
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
            trades.append(trade_ret_net)
            
    num_trades = len(trades)
    if num_trades == 0:
        return None
        
    winning_trades = sum(1 for r in trades if r > 0)
    win_rate = winning_trades / num_trades
    avg_trade_ret = np.mean(trades)
    
    total_net_return = df_sim['equity_strategy_net'].iloc[-1] - 1
    max_dd = (df_sim['equity_strategy_net'] / df_sim['equity_strategy_net'].cummax() - 1).min()
    
    # Daily returns for Sharpe (approximate using strategy daily equivalent if possible,
    # or just use intraday std of net returns scaled to daily)
    # 5-min bars per day = ~75. Annualized std = std(returns) * sqrt(75 * 252)
    bars_per_day = 75 if "5" in str(df.index[1] - df.index[0]) else 25  # rough guess
    net_ret_std = df_sim['net_returns'].std()
    sharpe = (df_sim['net_returns'].mean() / net_ret_std * np.sqrt(bars_per_day * 252)) if net_ret_std > 0 else 0.0
    
    return {
        'net_return': total_net_return,
        'max_drawdown': max_dd,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_profit': avg_trade_ret,
        'sharpe': sharpe,
        'df': df_sim
    }

def main():
    if not HAS_UPSTOX:
        print("ERROR: Could not import UpstoxFetcher.")
        return
        
    fetcher = UpstoxFetcher()
    days = 30  # 30-day historical data
    
    intervals = ["5minute", "15minute", "30minute"]
    
    # Prefetch and cache all dataframes to avoid repeatedly hitting the API
    data_cache = {}
    print(f"Prefetching {days} days of historical data for Nifty 50 across intervals...")
    for interval in intervals:
        print(f"  Fetching {interval} data...")
        df = fetcher.get_historical("NSE_INDEX|Nifty 50", interval, days=days)
        if not df.empty:
            data_cache[interval] = df
            print(f"  Cached {len(df)} candles.")
        else:
            print(f"  WARNING: Failed to fetch {interval} data.")
            
    if not data_cache:
        print("ERROR: No data fetched. Exiting.")
        return
        
    # Parameter Search Space
    windows = [20, 50, 100]
    entry_zs = [1.8, 2.0, 2.5]
    exit_zs = [0.0, 0.2, 0.5]
    stop_losses = [3.5, 4.0, None]
    trend_filters = [False, True]
    
    results = []
    
    # Run Sweep
    print("\nRunning Parameter Sweeps...")
    combinations = list(itertools.product(
        data_cache.keys(), windows, entry_zs, exit_zs, stop_losses, trend_filters
    ))
    
    total_runs = len(combinations)
    print(f"Total Combinations to Test: {total_runs}")
    
    for idx, (interval, window, entry_z, exit_z, stop_z, use_trend) in enumerate(combinations):
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{total_runs} simulations completed...")
            
        df = data_cache[interval]
        sim_res = run_simulation(df, window, entry_z, exit_z, stop_z, use_trend)
        
        if sim_res is not None:
            results.append({
                'interval': interval,
                'window': window,
                'entry_z': entry_z,
                'exit_z': exit_z,
                'stop_z': stop_z,
                'use_trend': use_trend,
                'net_return': sim_res['net_return'],
                'max_drawdown': sim_res['max_drawdown'],
                'num_trades': sim_res['num_trades'],
                'win_rate': sim_res['win_rate'],
                'avg_profit': sim_res['avg_profit'],
                'sharpe': sim_res['sharpe']
            })
            
    # Sort results
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No trades triggered for any combination.")
        return
        
    # Sort by Net Return
    res_df = res_df.sort_values(by='net_return', ascending=False)
    
    print("\n=== TOP 5 OPTIMIZED STRATEGY CONFIGURATIONS ===")
    for rank, (i, row) in enumerate(res_df.head(5).iterrows(), 1):
        print(f"Rank {rank}:")
        print(f"  Params: Int={row['interval']}, Window={row['window']}, Entry Z={row['entry_z']}, Exit Z={row['exit_z']}, Stop Z={row['stop_z']}, Trend={row['use_trend']}")
        print(f"  Metrics: Net Return={row['net_return']:.2%}, Max DD={row['max_drawdown']:.2%}, Trades={int(row['num_trades'])}, Win Rate={row['win_rate']:.2%}, Sharpe={row['sharpe']:.2f}\n")
        
    # Re-run the absolute best configuration to get its dataframe and plot it
    best = res_df.iloc[0]
    best_df = data_cache[best['interval']]
    best_sim = run_simulation(
        best_df, int(best['window']), best['entry_z'], best['exit_z'], 
        best['stop_z'] if not pd.isna(best['stop_z']) else None, best['use_trend']
    )
    
    # Save best plot
    plt.style.use('dark_background' if plt.style.available else 'default')
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    sim_df = best_sim['df']
    
    # Price
    axes[0].plot(sim_df['timestamps'], sim_df['close'], label='Nifty 50 Close', color='#39FF14', alpha=0.9)
    axes[0].plot(sim_df['timestamps'], sim_df['mean'], label='Moving Mean', color='cyan', linestyle='--', alpha=0.7)
    upper = sim_df['mean'] + best['entry_z'] * sim_df['std']
    lower = sim_df['mean'] - best['entry_z'] * sim_df['std']
    axes[0].plot(sim_df['timestamps'], upper, color='red', linestyle=':', alpha=0.6)
    axes[0].plot(sim_df['timestamps'], lower, color='blue', linestyle=':', alpha=0.6)
    axes[0].fill_between(sim_df['timestamps'], lower, upper, color='gray', alpha=0.1)
    axes[0].set_title(f"Price & Bands (Interval={best['interval']}, Window={best['window']})", fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper left')
    
    # Z-Score
    axes[1].plot(sim_df['timestamps'], sim_df['z'], label='Z-Score', color='yellow', linewidth=1.2)
    axes[1].axhline(best['entry_z'], color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(-best['entry_z'], color='red', linestyle='--')
    axes[1].axhline(best['exit_z'], color='cyan', linestyle=':', alpha=0.5)
    axes[1].axhline(-best['exit_z'], color='cyan', linestyle=':')
    axes[1].axhline(0, color='white', alpha=0.3)
    if not pd.isna(best['stop_z']):
        axes[1].axhline(best['stop_z'], color='magenta', linestyle='-.', alpha=0.5, label=f"Stop Loss (+-{best['stop_z']})")
        axes[1].axhline(-best['stop_z'], color='magenta', linestyle='-.', alpha=0.5)
    axes[1].set_ylabel('Z-Score')
    axes[1].set_title(f"Z-Score & Levels (Entry={best['entry_z']}, Exit={best['exit_z']})", fontsize=11, fontweight='bold')
    axes[1].legend(loc='upper left')
    
    # Equity
    axes[2].plot(sim_df['timestamps'], sim_df['equity_market'], label='Market Buy & Hold', color='white', linestyle='--', alpha=0.5)
    axes[2].plot(sim_df['timestamps'], sim_df['equity_strategy_net'], label='Optimized Strategy Net', color='#00FFFF', linewidth=2)
    axes[2].set_ylabel('Growth of 1 Unit')
    axes[2].set_title(f"Net Equity Curve (Return={best['net_return']:.2%}, Max DD={best['max_drawdown']:.2%}, Sharpe={best['sharpe']:.2f})", fontsize=11, fontweight='bold')
    axes[2].legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('nifty_optimized.png', dpi=150)
    plt.close()
    print("Saved optimized strategy plot to nifty_optimized.png")

if __name__ == "__main__":
    main()
