import os
import sys
import itertools
from pathlib import Path
import pandas as pd
import numpy as np

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
    
    df_sim['equity_market'] = (1 + df_sim['market_returns']).cumprod()
    df_sim['equity_strategy_net'] = (1 + df_sim['net_returns']).cumprod()
    
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
    
    if len(df_sim) > 1:
        time_diff = (df_sim['timestamps'].iloc[1] - df_sim['timestamps'].iloc[0]).total_seconds()
        if time_diff <= 300: # 5min
            bars_per_day = 75
        elif time_diff <= 900: # 15min
            bars_per_day = 25
        elif time_diff <= 1800: # 30min
            bars_per_day = 13
        elif time_diff <= 3600: # 1hour
            bars_per_day = 7
        else: # 1day
            bars_per_day = 1
    else:
        bars_per_day = 25
        
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
    
    interval_limits = {
        "5minute": 30,
        "15minute": 30,
        "30minute": 90,
        "1hour": 90
    }
    
    data_cache = {}
    print("=== Prefetching Historical Data for Active Trading Intervals ===")
    for interval, days in interval_limits.items():
        df = fetcher.get_historical("NSE_INDEX|Nifty 50", interval, days=days)
        if not df.empty:
            data_cache[interval] = df
            
    if not data_cache:
        print("ERROR: No data fetched. Exiting.")
        return
        
    # Sweep configurations
    windows = [20, 50, 100]
    entry_zs = [1.5, 1.8, 2.0]  # Lower entry Z slightly to allow more trades
    exit_zs = [0.0, 0.2, 0.5]
    stop_losses = [3.5, 4.0, None]
    trend_filters = [False, True]
    
    results = []
    
    combinations = []
    for interval in data_cache.keys():
        for params in itertools.product(windows, entry_zs, exit_zs, stop_losses, trend_filters):
            combinations.append((interval, *params))
            
    for idx, (interval, window, entry_z, exit_z, stop_z, use_trend) in enumerate(combinations):
        df = data_cache[interval]
        sim_res = run_simulation(df, window, entry_z, exit_z, stop_z, use_trend)
        
        if sim_res is not None:
            # Normalize trade count to "trades per month" (approx 21 trading days)
            days_in_backtest = interval_limits[interval]
            trades_per_month = sim_res['num_trades'] / days_in_backtest * 21
            
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
                'trades_per_month': trades_per_month,
                'win_rate': sim_res['win_rate'],
                'avg_profit': sim_res['avg_profit'],
                'sharpe': sim_res['sharpe']
            })
            
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No results generated.")
        return
        
    # Save all results to CSV
    csv_path = Path("sweep_results_hf.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"Saved all results to {csv_path.resolve()}")
    
    # Filter for active trading frequency: at least 15 trades per month (about 3 trades every 4 days)
    # and positive net returns
    active_df = res_df[(res_df['trades_per_month'] >= 15) & (res_df['net_return'] > 0)]
    
    print("\n=== TOP 5 PROFITABLE HIGH-FREQUENCY CONFIGURATIONS (>= 15 Trades/Month) ===")
    if not active_df.empty:
        active_df_sorted = active_df.sort_values(by='net_return', ascending=False)
        for rank, (i, row) in enumerate(active_df_sorted.head(5).iterrows(), 1):
            stop_val = f"{row['stop_z']:.1f}" if not pd.isna(row['stop_z']) else "None"
            print(f"Rank {rank}:")
            print(f"  Params: Int={row['interval']}, Window={row['window']}, Entry Z={row['entry_z']}, Exit Z={row['exit_z']}, Stop Z={stop_val}, Trend={row['use_trend']}")
            print(f"  Metrics: Net Return={row['net_return']:.2%}, Max DD={row['max_drawdown']:.2%}, Trades/Month={row['trades_per_month']:.1f} (Total: {int(row['num_trades'])}), Win Rate={row['win_rate']:.2%}, Sharpe={row['sharpe']:.2f}\n")
    else:
        print("No configurations with >= 15 trades/month were profitable after transaction costs.")
        
    # Let's also look for a slightly lower threshold: at least 8 trades per month (about 2 trades per week)
    print("=== TOP 5 PROFITABLE MID-FREQUENCY CONFIGURATIONS (8 to 15 Trades/Month) ===")
    mid_df = res_df[(res_df['trades_per_month'] >= 8) & (res_df['trades_per_month'] < 15) & (res_df['net_return'] > 0)]
    if not mid_df.empty:
        mid_df_sorted = mid_df.sort_values(by='net_return', ascending=False)
        for rank, (i, row) in enumerate(mid_df_sorted.head(5).iterrows(), 1):
            stop_val = f"{row['stop_z']:.1f}" if not pd.isna(row['stop_z']) else "None"
            print(f"Rank {rank}:")
            print(f"  Params: Int={row['interval']}, Window={row['window']}, Entry Z={row['entry_z']}, Exit Z={row['exit_z']}, Stop Z={stop_val}, Trend={row['use_trend']}")
            print(f"  Metrics: Net Return={row['net_return']:.2%}, Max DD={row['max_drawdown']:.2%}, Trades/Month={row['trades_per_month']:.1f} (Total: {int(row['num_trades'])}), Win Rate={row['win_rate']:.2%}, Sharpe={row['sharpe']:.2f}\n")
    else:
        print("No configurations in this range were profitable.")

if __name__ == "__main__":
    main()
