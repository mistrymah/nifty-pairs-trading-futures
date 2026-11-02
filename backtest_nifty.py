import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup path to import UpstoxFetcher from nifty_mcp_upstox
UPSTOX_DIR = Path("E:/Trading_Agents/nifty_mcp_upstox")
if str(UPSTOX_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTOX_DIR))

# Try importing UpstoxFetcher
try:
    from fetcher import UpstoxFetcher
    HAS_UPSTOX = True
except ImportError:
    HAS_UPSTOX = False

def run_nifty_backtest(interval="5minute", days=10, window=20, entry_z=2.0, exit_z=0.5, cost_pct=0.0003):
    """
    Runs a Z-Score Mean Reversion backtest on real Nifty historical data.
    
    Parameters:
    - interval: Upstox candle interval (e.g. "5minute", "15minute", "1hour")
    - days: Number of days of historical data to fetch
    - window: Rolling window for Z-score moving average & std dev
    - entry_z: Z-score threshold to trigger entry (e.g. 2.0)
    - exit_z: Z-score threshold to exit trade (e.g. 0.5)
    - cost_pct: Transaction cost + slippage percentage per trade (e.g. 0.03% = 0.0003)
    """
    print(f"=== Z-Score Mean Reversion Backtest on Nifty ===")
    print(f"Interval: {interval} | Lookback: {days} days")
    print(f"Parameters: Window={window}, Entry Z={entry_z}, Exit Z={exit_z}")
    print(f"Transaction Cost: {cost_pct:.4%} per trade\n")
    
    # 2. Fetch Data
    if not HAS_UPSTOX:
        print("ERROR: Could not import UpstoxFetcher. Make sure 'nifty_mcp_upstox' folder is in E:/Trading_Agents/")
        return
        
    fetcher = UpstoxFetcher()
    print("Fetching historical data from Upstox API...")
    df = fetcher.get_nifty_historical(interval=interval, days=days)
    
    if df.empty:
        print("ERROR: No data fetched from Upstox. Please verify your Access Token and connection.")
        return
        
    print(f"Successfully fetched {len(df)} candles.")
    
    # 3. Calculate Indicators (Z-score)
    df['mean'] = df['close'].rolling(window=window, min_periods=window).mean()
    df['std'] = df['close'].rolling(window=window, min_periods=window).std(ddof=0)
    df['z'] = (df['close'] - df['mean']) / df['std']
    
    # Drop rows without enough data to warm up indicators
    df = df.dropna(subset=['z']).copy()
    
    # 4. Generate Signals (Corrected State Machine)
    position = 0
    positions = []
    
    for z in df['z']:
        if position == 0:
            if z < -entry_z:
                position = 1   # Long
            elif z > entry_z:
                position = -1  # Short
        else:
            # Exit when price reverts back inside exit threshold
            if abs(z) < exit_z:
                position = 0
        positions.append(position)
        
    df['signal'] = positions
    
    # 5. Backtest Simulation (execute on next candle to avoid look-ahead bias)
    df['position'] = df['signal'].shift(1).fillna(0)
    df['market_returns'] = df['close'].pct_change().fillna(0)
    df['strategy_returns'] = df['position'] * df['market_returns']
    
    # Calculate transaction costs when position changes
    # Position change = absolute diff of position
    df['trade_trigger'] = df['position'].diff().fillna(0).abs()
    df['costs'] = df['trade_trigger'] * cost_pct
    df['net_returns'] = df['strategy_returns'] - df['costs']
    
    # Cumulative Curves
    df['equity_market'] = (1 + df['market_returns']).cumprod()
    df['equity_strategy_raw'] = (1 + df['strategy_returns']).cumprod()
    df['equity_strategy_net'] = (1 + df['net_returns']).cumprod()
    
    # 6. Calculate Trade-Level Metrics
    df['trade_id'] = (df['position'] != df['position'].shift()).cumsum()
    
    trades = []
    # Filter for intervals when in a position
    active_positions = df[df['position'] != 0]
    if not active_positions.empty:
        for trade_id, group in active_positions.groupby('trade_id'):
            pos_type = "LONG" if group['position'].iloc[0] > 0 else "SHORT"
            entry_idx = group.index[0]
            exit_idx = group.index[-1]
            
            # Entry price is the close price of the bar BEFORE the trade starts
            # (since signal was shifted, position starts at entry_idx, meaning entry was at close of entry_idx - 1)
            # Find index in main df
            prev_idx = entry_idx - 1 if (entry_idx - 1) in df.index else entry_idx
            entry_price = df.loc[prev_idx, 'close']
            exit_price = group['close'].iloc[-1]
            
            # Simple trade return
            if pos_type == "LONG":
                trade_ret = (exit_price - entry_price) / entry_price
            else:
                trade_ret = (entry_price - exit_price) / entry_price
                
            # Apply round-trip cost (entry + exit)
            trade_ret_net = trade_ret - (2 * cost_pct)
            
            trades.append({
                'trade_id': trade_id,
                'type': pos_type,
                'entry_time': df.loc[prev_idx, 'timestamps'],
                'exit_time': group['timestamps'].iloc[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_raw': trade_ret,
                'return_net': trade_ret_net
            })
            
    trades_df = pd.DataFrame(trades)
    
    # 7. Print Performance Summary
    total_market_ret = df['equity_market'].iloc[-1] - 1
    total_strat_raw = df['equity_strategy_raw'].iloc[-1] - 1
    total_strat_net = df['equity_strategy_net'].iloc[-1] - 1
    
    max_dd_net = (df['equity_strategy_net'] / df['equity_strategy_net'].cummax() - 1).min()
    
    num_trades = len(trades)
    if num_trades > 0:
        winning_trades = sum(1 for t in trades if t['return_net'] > 0)
        win_rate = winning_trades / num_trades
        avg_trade_ret = trades_df['return_net'].mean()
    else:
        win_rate = 0.0
        avg_trade_ret = 0.0
        
    print("=== PERFORMANCE RESULTS ===")
    print(f"Total Market (Buy & Hold) Return: {total_market_ret:.2%}")
    print(f"Strategy Gross Return (No Costs): {total_strat_raw:.2%}")
    print(f"Strategy Net Return (After Costs): {total_strat_net:.2%}")
    print(f"Maximum Net Drawdown: {max_dd_net:.2%}")
    print(f"Total Executed Trades: {num_trades}")
    if num_trades > 0:
        print(f"Trade-Level Win Rate: {win_rate:.2%}")
        print(f"Average Profit per Trade: {avg_trade_ret:.3%}")
        
    # 8. Plot and Save Image
    plt.style.use('dark_background' if plt.style.available else 'default')
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Price and Bands
    axes[0].plot(df['timestamps'], df['close'], label='Nifty 50 Close', color='#39FF14', alpha=0.9, linewidth=1.5)
    axes[0].plot(df['timestamps'], df['mean'], label='Rolling Mean', color='cyan', linestyle='--', alpha=0.7)
    upper_band = df['mean'] + entry_z * df['std']
    lower_band = df['mean'] - entry_z * df['std']
    axes[0].plot(df['timestamps'], upper_band, color='red', linestyle=':', label=f'+{entry_z} Std')
    axes[0].plot(df['timestamps'], lower_band, color='blue', linestyle=':', label=f'-{entry_z} Std')
    axes[0].fill_between(df['timestamps'], lower_band, upper_band, color='gray', alpha=0.1)
    
    # Mark entry points on price plot
    long_entries = df[(df['signal'] == 1) & (df['signal'].shift(1) != 1)]
    short_entries = df[(df['signal'] == -1) & (df['signal'].shift(1) != -1)]
    
    axes[0].scatter(long_entries['timestamps'], long_entries['close'], marker='^', color='cyan', s=100, label='Long Entry', zorder=5)
    axes[0].scatter(short_entries['timestamps'], short_entries['close'], marker='v', color='magenta', s=100, label='Short Entry', zorder=5)
    
    axes[0].set_title('Nifty 50 Intraday Price & Z-Score Strategy Bands', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.2)
    
    # Plot 2: Z-Score
    axes[1].plot(df['timestamps'], df['z'], label='Z-Score', color='yellow', linewidth=1.2)
    axes[1].axhline(entry_z, color='red', linestyle='--', alpha=0.7, label=f'Entry threshold (+-{entry_z})')
    axes[1].axhline(-entry_z, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(exit_z, color='cyan', linestyle=':', alpha=0.5, label=f'Exit threshold (+-{exit_z})')
    axes[1].axhline(-exit_z, color='cyan', linestyle=':', alpha=0.5)
    axes[1].axhline(0, color='white', linestyle='-', alpha=0.3)
    axes[1].set_ylabel('Standard Deviations (Z)')
    axes[1].set_title('Z-Score & Thresholds', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.2)
    
    # Plot 3: Equity Curves
    axes[2].plot(df['timestamps'], df['equity_market'], label='Market Buy & Hold', color='white', alpha=0.5, linestyle='--')
    axes[2].plot(df['timestamps'], df['equity_strategy_raw'], label='Strategy Gross', color='#FF007F', alpha=0.7)
    axes[2].plot(df['timestamps'], df['equity_strategy_net'], label='Strategy Net (After Costs)', color='#00FFFF', linewidth=2)
    axes[2].set_ylabel('Growth of 1 Unit')
    axes[2].set_title('Cumulative Performance Comparison', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.2)
    
    plt.tight_layout()
    plot_path = Path("nifty_backtest.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved performance chart to {plot_path.resolve()}")
    plt.close()

if __name__ == "__main__":
    # Run backtest with default arguments
    # Interval "5minute" for last 10 calendar days
    run_nifty_backtest(interval="5minute", days=10)
