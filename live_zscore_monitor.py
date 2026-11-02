import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# 1. Setup path to import UpstoxFetcher from nifty_mcp_upstox
UPSTOX_DIR = Path("E:/Trading_Agents/nifty_mcp_upstox")
if str(UPSTOX_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTOX_DIR))

try:
    from fetcher import UpstoxFetcher
    HAS_UPSTOX = True
except ImportError:
    HAS_UPSTOX = False

# Strategy Parameters
WINDOW = 20
ENTRY_Z = 1.8
EXIT_Z = 0.0
INTERVAL = "30minute"

POSITION_FILE = Path("live_position.json")

# ANSI Terminal Colors
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"

def load_position():
    """Load active trade state from json file"""
    if POSITION_FILE.exists():
        try:
            with open(POSITION_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"position": 0, "entry_price": 0.0, "entry_time": ""}

def save_position(position, entry_price=0.0):
    """Save active trade state to json file"""
    state = {
        "position": position,
        "entry_price": entry_price,
        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if position != 0 else ""
    }
    with open(POSITION_FILE, 'w') as f:
        json.dump(state, f, indent=4)
    return state

def get_live_zscore(fetcher):
    """
    Fetches historical 30-min data, gets current spot price,
    and returns current spot, mean, std, z-score, and target bands.
    """
    # Fetch latest completed candles
    hist_df = fetcher.get_historical("NSE_INDEX|Nifty 50", INTERVAL, days=10)
    if hist_df.empty:
        raise Exception("Failed to fetch historical Nifty data.")
        
    spot = fetcher.get_nifty_spot()
    if spot.price == 0.0:
        raise Exception("Failed to fetch live Nifty spot price.")
        
    closes = list(hist_df['close'])
    last_hist_time = pd.to_datetime(hist_df['timestamps'].iloc[-1]).tz_localize(None)
    current_time = pd.to_datetime(spot.timestamp).tz_localize(None)
    
    # Append or update the current spot price in the candle series
    if current_time > last_hist_time:
        # If in the same 30-min bar, replace the last element, otherwise append
        if (current_time - last_hist_time).total_seconds() < 1800:
            closes[-1] = spot.price
        else:
            closes.append(spot.price)
    else:
        closes[-1] = spot.price
        
    # Calculate rolling metrics on the last WINDOW items
    active_window = closes[-WINDOW:]
    mean = np.mean(active_window)
    std = np.std(active_window, ddof=0)
    zscore = (spot.price - mean) / std if std > 0 else 0.0
    
    # Calculate entry and exit target price levels
    buy_band = mean - ENTRY_Z * std
    sell_band = mean + ENTRY_Z * std
    exit_target = mean
    
    return {
        'spot': spot.price,
        'change_pct': spot.change_pct,
        'mean': mean,
        'std': std,
        'zscore': zscore,
        'buy_band': buy_band,
        'sell_band': sell_band,
        'exit_target': exit_target,
        'timestamp': spot.timestamp
    }

def print_dashboard(state, metrics):
    """Prints a beautiful formatted dashboard to the console"""
    # Clear console (cross-platform)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    pos_val = state["position"]
    pos_text = f"{CYAN}FLAT{RESET}"
    if pos_val == 1:
        pos_text = f"{GREEN}LONG (Bought){RESET}"
    elif pos_val == -1:
        pos_text = f"{RED}SHORT (Sold){RESET}"
        
    # Calculate current trade profit/loss
    pnl_text = "N/A"
    if pos_val == 1:
        pnl = metrics['spot'] - state['entry_price']
        pnl_pct = pnl / state['entry_price'] * 100
        color = GREEN if pnl >= 0 else RED
        pnl_text = f"{color}{pnl:+.2f} ({pnl_pct:+.2f}%){RESET}"
    elif pos_val == -1:
        pnl = state['entry_price'] - metrics['spot']
        pnl_pct = pnl / state['entry_price'] * 100
        color = GREEN if pnl >= 0 else RED
        pnl_text = f"{color}{pnl:+.2f} ({pnl_pct:+.2f}%){RESET}"

    # Determine Z-Score Status color
    z_color = RESET
    if metrics['zscore'] < -ENTRY_Z:
        z_color = GREEN + BOLD
    elif metrics['zscore'] > ENTRY_Z:
        z_color = RED + BOLD
    elif abs(metrics['zscore']) < 0.2:
        z_color = CYAN
        
    # Formatting spot change
    change_color = GREEN if metrics['change_pct'] >= 0 else RED
    change_text = f"{change_color}{metrics['change_pct']:+.2f}%{RESET}"
    
    # Header
    print(f"{BOLD}============================================================{RESET}")
    print(f"{BOLD}       NIFTY 50 LIVE Z-SCORE MEAN REVERSION MONITOR        {RESET}")
    print(f"{BOLD}============================================================{RESET}")
    print(f" Last Updated : {metrics['timestamp']}")
    print(f" Nifty Spot   : {BOLD}{metrics['spot']:.2f}{RESET} ({change_text})")
    print(f" Current Z    : {z_color}{metrics['zscore']:+.2f}{RESET} (Thresholds: Buy < -{ENTRY_Z}, Sell > {ENTRY_Z})")
    print(f"------------------------------------------------------------")
    
    # Strategy Bands
    print(f" {GREEN}Long Entry (Z < -{ENTRY_Z})  : {metrics['buy_band']:.2f} {RESET}(Diff: {metrics['spot'] - metrics['buy_band']:+.2f})")
    print(f" {CYAN}Exit Target (Z = 0.0)   : {metrics['exit_target']:.2f} {RESET}(Diff: {metrics['spot'] - metrics['exit_target']:+.2f})")
    print(f" {RED}Short Entry (Z > {ENTRY_Z})   : {metrics['sell_band']:.2f} {RESET}(Diff: {metrics['spot'] - metrics['sell_band']:+.2f})")
    print(f"------------------------------------------------------------")
    
    # Position Information
    print(f" Current Position  : {pos_text}")
    if pos_val != 0:
        print(f" Entry Price       : {state['entry_price']:.2f} (Time: {state['entry_time']})")
        print(f" Live P&L          : {pnl_text}")
    print(f"------------------------------------------------------------")
    
    # Live Action Signals
    signal_text = f"{BOLD}HOLD POSITION / WAIT{RESET}"
    if pos_val == 0:
        if metrics['zscore'] < -ENTRY_Z:
            signal_text = f"{GREEN}{BOLD}★★★ BUY SIGNAL (LONG ENTRY) ★★★{RESET}"
        elif metrics['zscore'] > ENTRY_Z:
            signal_text = f"{RED}{BOLD}★★★ SELL SIGNAL (SHORT ENTRY) ★★★{RESET}"
    else:
        # Check exit
        if pos_val == 1 and metrics['zscore'] >= EXIT_Z:
            signal_text = f"{CYAN}{BOLD}⚡⚡⚡ EXIT SIGNAL (CLOSE LONG) ⚡⚡⚡{RESET}"
        elif pos_val == -1 and metrics['zscore'] <= EXIT_Z:
            signal_text = f"{CYAN}{BOLD}⚡⚡⚡ EXIT SIGNAL (CLOSE SHORT) ⚡⚡⚡{RESET}"
            
    print(f" Action Alert      : {signal_text}")
    print(f"{BOLD}============================================================{RESET}")
    print(f" Monitor is active. Updating automatically every 60s... (Ctrl+C to exit)")

def main():
    if not HAS_UPSTOX:
        print("ERROR: Could not import UpstoxFetcher. Please check paths.")
        return
        
    fetcher = UpstoxFetcher()
    state = load_position()
    
    print("Initializing Z-Score Live Monitor...")
    
    while True:
        try:
            # 1. Fetch metrics
            metrics = get_live_zscore(fetcher)
            
            # 2. Update Position State based on Signals
            pos_val = state["position"]
            if pos_val == 0:
                if metrics['zscore'] < -ENTRY_Z:
                    # Trigger Buy Long
                    state = save_position(1, metrics['spot'])
                elif metrics['zscore'] > ENTRY_Z:
                    # Trigger Sell Short
                    state = save_position(-1, metrics['spot'])
            else:
                # Trigger Exit
                if pos_val == 1 and metrics['zscore'] >= EXIT_Z:
                    state = save_position(0)
                elif pos_val == -1 and metrics['zscore'] <= EXIT_Z:
                    state = save_position(0)
                    
            # 3. Print Dashboard
            print_dashboard(state, metrics)
            
        except KeyboardInterrupt:
            print("\nExiting Live Monitor. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {YELLOW}WARNING: Connection or processing error: {e}{RESET}")
            print("Retrying in 60 seconds...")
            
        time.sleep(60)

if __name__ == "__main__":
    main()
