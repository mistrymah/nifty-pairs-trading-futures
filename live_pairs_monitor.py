import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# 1. Setup path to import UpstoxFetcher (local or external fallback)
try:
    from fetcher import UpstoxFetcher
    HAS_UPSTOX = True
except ImportError:
    UPSTOX_DIR = Path("E:/Trading_Agents/nifty_mcp_upstox")
    if str(UPSTOX_DIR) not in sys.path and UPSTOX_DIR.exists():
        sys.path.insert(0, str(UPSTOX_DIR))
    try:
        from fetcher import UpstoxFetcher
        HAS_UPSTOX = True
    except ImportError:
        HAS_UPSTOX = False

from rollover_engine import square_off_expired_positions, sync_pairs_config

# Strategy Parameters
WINDOW = 50
ENTRY_Z = 1.8
EXIT_Z = 0.2
LOTS_PER_LEG = 1           # Trade 1 lot of futures per leg
COST_PCT = 0.00005         # ~0.005% transaction fee rate (brokerage + STT for Futures)
SCRIPT_DIR = Path(__file__).parent.resolve()
POSITION_FILE = SCRIPT_DIR / "live_pairs_positions.json"

# ANSI Terminal Colors
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"

# Stock Pairs Map (Futures Expiry: 27-Oct-2026)
PAIRS = {
    "PFC-REC": {
        "symbol_a": "PFC", "symbol_b": "RECLTD",
        "key_a": "NSE_FO|48969", "key_b": "NSE_FO|48986",
        "lot_a": 1300, "lot_b": 1575, "expiry": "2026-10-27"
    },
    "TCS-INFY": {
        "symbol_a": "TCS", "symbol_b": "INFY",
        "key_a": "NSE_FO|49013", "key_b": "NSE_FO|48886",
        "lot_a": 225, "lot_b": 400, "expiry": "2026-10-27"
    },
    "ICICI-HDFC": {
        "symbol_a": "ICICIBANK", "symbol_b": "HDFCBANK",
        "key_a": "NSE_FO|48874", "key_b": "NSE_FO|48864",
        "lot_a": 700, "lot_b": 650, "expiry": "2026-10-27"
    },
    "TATASTEEL-JSW": {
        "symbol_a": "TATASTEEL", "symbol_b": "JSWSTEEL",
        "key_a": "NSE_FO|49012", "key_b": "NSE_FO|48900",
        "lot_a": 2750, "lot_b": 675, "expiry": "2026-10-27"
    },
    "BOB-CANBK": {
        "symbol_a": "BANKBARODA", "symbol_b": "CANBK",
        "key_a": "NSE_FO|48742", "key_b": "NSE_FO|48778",
        "lot_a": 2925, "lot_b": 6750, "expiry": "2026-10-27"
    },
    "NTPC-POWERGRID": {
        "symbol_a": "NTPC", "symbol_b": "POWERGRID",
        "key_a": "NSE_FO|48954", "key_b": "NSE_FO|48980",
        "lot_a": 1500, "lot_b": 1900, "expiry": "2026-10-27"
    },
    "HEROMOTO-BAJAJ": {
        "symbol_a": "HEROMOTOCO", "symbol_b": "BAJAJ-AUTO",
        "key_a": "NSE_FO|48866", "key_b": "NSE_FO|48737",
        "lot_a": 150, "lot_b": 75, "expiry": "2026-10-27"
    },
    "AXIS-KOTAK": {
        "symbol_a": "AXISBANK", "symbol_b": "KOTAKBANK",
        "key_a": "NSE_FO|48736", "key_b": "NSE_FO|48906",
        "lot_a": 625, "lot_b": 2000, "expiry": "2026-10-27"
    }
}

TRADES_CSV = SCRIPT_DIR / "live_pairs_trades.csv"

def log_pairs_trade(pair, pos_type, entry_time, exit_time, entry_a, entry_b, exit_a, exit_b, entry_spread, exit_spread, pnl_pct):
    # Retrieve expiry from PAIRS dict, default to 2026-08-25 if missing
    expiry_a = "2026-08-25" 
    expiry_b = "2026-08-25"
    if pair in PAIRS and 'expiry' in PAIRS[pair]:
        expiry_a = PAIRS[pair]['expiry']
        expiry_b = PAIRS[pair]['expiry']
        
    file_exists = TRADES_CSV.exists()
    with open(TRADES_CSV, "a", newline="") as f:
        import csv
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["pair", "type", "entry_time", "exit_time", "entry_price_a", "entry_price_b", "exit_price_a", "exit_price_b", "entry_spread", "exit_spread", "pnl_pct", "expiry_a", "expiry_b"])
        writer.writerow([pair, pos_type, entry_time, exit_time, entry_a, entry_b, exit_a, exit_b, entry_spread, exit_spread, f"{pnl_pct:.2f}", expiry_a, expiry_b])

def save_positions(positions):
    """Save active pairs trading positions to JSON file"""
    with open(POSITION_FILE, 'w') as f:
        json.dump(positions, f, indent=4)

def load_positions():
    """Load active pairs trading positions from JSON file"""
    positions = {}
    if POSITION_FILE.exists():
        try:
            with open(POSITION_FILE, 'r') as f:
                positions = json.load(f)
        except Exception:
            pass
            
    # Ensure all current pairs exist in the positions dict
    updated = False
    for pair in PAIRS:
        if pair not in positions:
            positions[pair] = {
                "position": 0,
                "entry_price_a": 0.0,
                "entry_price_b": 0.0,
                "entry_spread": 0.0,
                "entry_time": ""
            }
            updated = True
            
    if updated:
        save_positions(positions)
        
    return positions

def fetch_pair_prices(fetcher, pair_config):
    """Fetch live prices for both stocks in the pair using Upstox get_quote API"""
    try:
        keys_str = f"{pair_config['key_a']},{pair_config['key_b']}"
        quote_data = fetcher.get_quote(keys_str)
        data = quote_data.get("data", {})
        
        # Parse live price for stock A
        price_a = 0.0
        for k, v in data.items():
            if pair_config['key_a'].replace("|", ":") in k:
                price_a = float(v.get("last_price", 0))
                break
                
        # Parse live price for stock B
        price_b = 0.0
        for k, v in data.items():
            if pair_config['key_b'].replace("|", ":") in k:
                price_b = float(v.get("last_price", 0))
                break
                
        return price_a, price_b
    except Exception as e:
        # Fallback to get_option_quote
        try:
            qa = fetcher.get_option_quote(pair_config['key_a'])
            qb = fetcher.get_option_quote(pair_config['key_b'])
            price_a = qa.price if qa else 0.0
            price_b = qb.price if qb else 0.0
            return price_a, price_b
        except Exception as e_fallback:
            raise Exception(f"Failed to fetch live prices: {e} | Fallback: {e_fallback}")

def get_live_pairs_metrics(fetcher, pair_name, pair_config):
    """
    Fetches historical 5m candles, pulls live quotes, 
    and computes rolling mean, std, zscore, and current spread.
    """
    # 1. Fetch historical 5m candles for both stocks (last 10 days to handle weekends/holidays)
    df_a = fetcher.get_historical(pair_config['key_a'], "5minute", days=10)
    df_b = fetcher.get_historical(pair_config['key_b'], "5minute", days=10)
    
    if df_a.empty or df_b.empty:
        raise Exception("Failed to fetch historical data.")
        
    df_a_clean = df_a[['timestamps', 'close']].rename(columns={'close': 'close_a'})
    df_b_clean = df_b[['timestamps', 'close']].rename(columns={'close': 'close_b'})
    
    # 2. Align timestamps
    df = pd.merge(df_a_clean, df_b_clean, on='timestamps', how='inner')
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    if len(df) < WINDOW:
        raise Exception("Not enough aligned historical candles.")
        
    # 3. Pull live quotes
    price_a, price_b = fetch_pair_prices(fetcher, pair_config)
    
    if price_a == 0.0 or price_b == 0.0:
        # Fallback to last historical close if market is closed or quote fails
        price_a = df['close_a'].iloc[-1]
        price_b = df['close_b'].iloc[-1]
        
    # Append the live tick to the candles list
    closes_a = list(df['close_a'])
    closes_b = list(df['close_b'])
    
    closes_a.append(price_a)
    closes_b.append(price_b)
    
    # 4. Calculate spread metrics on the active window
    spreads = np.log(closes_a) - np.log(closes_b)
    active_spreads = spreads[-WINDOW:]
    
    current_spread = spreads[-1]
    mean = np.mean(active_spreads)
    std = np.std(active_spreads, ddof=0)
    zscore = (current_spread - mean) / std if std > 0 else 0.0
    
    return {
        "price_a": price_a,
        "price_b": price_b,
        "spread": current_spread,
        "mean": mean,
        "std": std,
        "zscore": zscore,
        "buy_z": -ENTRY_Z,
        "sell_z": ENTRY_Z,
        "exit_z": EXIT_Z
    }

def print_dashboard(positions, all_metrics):
    """Prints a beautiful formatted pairs trading console dashboard"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Header
    print(f"{BOLD}============================================================{RESET}")
    print(f"{BOLD}       NIFTY 50 LIVE PAIRS TRADING MONITOR (5m Z-SCORE)      {RESET}")
    print(f"{BOLD}============================================================{RESET}")
    print(f" Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Parameters: Window={WINDOW}, Entry Z={ENTRY_Z}, Exit Z={EXIT_Z}")
    print(f"============================================================{RESET}")
    
    # Loop pairs
    for idx, (pair, metrics) in enumerate(all_metrics.items(), 1):
        pos_data = positions[pair]
        pos_val = pos_data["position"]
        
        # Position label
        pos_text = f"{CYAN}FLAT{RESET}"
        if pos_val == 1:
            pos_text = f"{GREEN}LONG SPREAD (Long {PAIRS[pair]['symbol_a']} / Short {PAIRS[pair]['symbol_b']}){RESET}"
        elif pos_val == -1:
            pos_text = f"{RED}SHORT SPREAD (Short {PAIRS[pair]['symbol_a']} / Long {PAIRS[pair]['symbol_b']}){RESET}"
            
        # Zscore status color
        z_color = RESET
        if metrics['zscore'] < -ENTRY_Z:
            z_color = GREEN + BOLD
        elif metrics['zscore'] > ENTRY_Z:
            z_color = RED + BOLD
        elif abs(metrics['zscore']) < 0.2:
            z_color = CYAN
            
        # Calculate P&L (INR Rupees and %)
        pnl_text = "N/A"
        if pos_val != 0:
            qty_a = LOTS_PER_LEG * PAIRS[pair]['lot_a'] if pos_data['entry_price_a'] > 0 else 0.0
            qty_b = LOTS_PER_LEG * PAIRS[pair]['lot_b'] if pos_data['entry_price_b'] > 0 else 0.0
            if pos_val == 1:
                pnl_a = qty_a * (metrics['price_a'] - pos_data['entry_price_a'])
                pnl_b = qty_b * (pos_data['entry_price_b'] - metrics['price_b'])
            else:
                pnl_a = qty_a * (pos_data['entry_price_a'] - metrics['price_a'])
                pnl_b = qty_b * (metrics['price_b'] - pos_data['entry_price_b'])
            gross_pnl = pnl_a + pnl_b
            entry_value = (qty_a * pos_data['entry_price_a']) + (qty_b * pos_data['entry_price_b'])
            exit_value = (qty_a * metrics['price_a']) + (qty_b * metrics['price_b'])
            total_costs = (entry_value + exit_value) * COST_PCT
            net_pnl_inr = gross_pnl - total_costs
            pnl_pct = (net_pnl_inr / entry_value) * 100 if entry_value > 0 else 0.0
            pnl_color = GREEN if net_pnl_inr >= 0 else RED
            pnl_text = f"{pnl_color}Rs. {net_pnl_inr:+.2f} ({pnl_pct:+.2f}%){RESET}"
            
        # Action Alerts
        alert_text = f"{BOLD}HOLD / WAIT{RESET}"
        if pos_val == 0:
            if metrics['zscore'] < -ENTRY_Z:
                alert_text = f"{GREEN}{BOLD}★★★ BUY SPREAD SIGNAL (Z < -{ENTRY_Z}) ★★★{RESET}"
            elif metrics['zscore'] > ENTRY_Z:
                alert_text = f"{RED}{BOLD}★★★ SELL SPREAD SIGNAL (Z > {ENTRY_Z}) ★★★{RESET}"
        else:
            # Check exit
            if pos_val == 1 and metrics['zscore'] >= EXIT_Z:
                alert_text = f"{CYAN}{BOLD}⚡⚡⚡ EXIT SPREAD SIGNAL (Z >= {EXIT_Z}) ⚡⚡⚡{RESET}"
            elif pos_val == -1 and metrics['zscore'] <= -EXIT_Z:
                alert_text = f"{CYAN}{BOLD}⚡⚡⚡ EXIT SPREAD SIGNAL (Z <= -{EXIT_Z}) ⚡⚡⚡{RESET}"
                
        # Layout rendering
        print(f" {BOLD}PAIR {idx}: {pair} (FUTURES){RESET}")
        exp_display = PAIRS[pair].get('expiry', '29-Sep-2026')
        print(f"   Prices    : {PAIRS[pair]['symbol_a']} = {metrics['price_a']:.2f} | {PAIRS[pair]['symbol_b']} = {metrics['price_b']:.2f} ({exp_display} Expiry)")
        print(f"   Spread    : {metrics['spread']:.4f} (Rolling Mean: {metrics['mean']:.4f})")
        print(f"   Z-Score   : {z_color}{metrics['zscore']:+.2f}{RESET} (Entry Thresholds: {metrics['buy_z']:.1f} / {metrics['sell_z']:.1f})")
        print(f"   Position  : {pos_text}")
        if pos_val != 0:
            print(f"   Entry Info: Entry Spread = {pos_data['entry_spread']:.4f} (Time: {pos_data['entry_time']})")
            print(f"   Live Return: {pnl_text}")
        print(f"   Alert     : {alert_text}")
        print(f"------------------------------------------------------------")
        
    # Load completed trades stats
    realized_pnl_inr = 0.0
    trades_count = 0
    if TRADES_CSV.exists():
        try:
            df_t = pd.read_csv(TRADES_CSV)
            trades_count = len(df_t)
            for idx, row in df_t.iterrows():
                pair = row.get('pair', '')
                entry_a = float(row['entry_price_a'])
                exit_a = float(row['exit_price_a'])
                entry_b = float(row['entry_price_b'])
                exit_b = float(row['exit_price_b'])
                pos_type = row['type']
                
                # Fetch lots if pair exists in config, otherwise approximate
                if pair in PAIRS:
                    qty_a = LOTS_PER_LEG * PAIRS[pair]['lot_a']
                    qty_b = LOTS_PER_LEG * PAIRS[pair]['lot_b']
                else:
                    qty_a = 50000.0 / entry_a  # Fallback for old legacy trades
                    qty_b = 50000.0 / entry_b
                
                if "LONG" in pos_type:
                    pnl_a = qty_a * (exit_a - entry_a)
                    pnl_b = qty_b * (entry_b - exit_b)
                else:
                    pnl_a = qty_a * (entry_a - exit_a)
                    pnl_b = qty_b * (exit_b - entry_b)
                    
                gross_pnl = pnl_a + pnl_b
                entry_val = (qty_a * entry_a) + (qty_b * entry_b)
                exit_val = (qty_a * exit_a) + (qty_b * exit_b)
                costs = (entry_val + exit_val) * COST_PCT
                realized_pnl_inr += (gross_pnl - costs)
        except Exception:
            pass
            
    print(f" {BOLD}PORTFOLIO PERFORMANCE SUMMARY:{RESET}")
    print(f"   Completed Paper Trades : {trades_count}")
    print(f"   Cumulative Realized PNL: {GREEN if realized_pnl_inr >= 0 else RED}Rs. {realized_pnl_inr:+.2f}{RESET}")
    print(f"------------------------------------------------------------")
    print(f" Dashboard is active. Updating automatically every 60s... (Ctrl+C to exit)")
    print(f"============================================================{RESET}")

def main():
    # Enable ANSI colors in Windows Command Prompt/PowerShell
    if os.name == 'nt':
        os.system('')
        
    if not HAS_UPSTOX:
        print("ERROR: Could not import UpstoxFetcher.")
        sys.exit(1)
        
    fetcher = UpstoxFetcher()
    positions = load_positions()
    
    print("Initializing Live Pairs Trading Monitor...")
    
    # Initial automated expiry square-off & contract rollover
    square_off_expired_positions(positions, PAIRS, fetcher)
    sync_pairs_config(PAIRS)
    
    while True:
        try:
            # Check for expiry square-off & contract rollover every cycle
            square_off_expired_positions(positions, PAIRS, fetcher)
            sync_pairs_config(PAIRS)
            
            all_metrics = {}
            # 1. Fetch metrics for all pairs
            for pair_name, config in PAIRS.items():
                metrics = get_live_pairs_metrics(fetcher, pair_name, config)
                all_metrics[pair_name] = metrics
                
                # 2. Update Position State based on Signals
                pos_data = positions[pair_name]
                pos_val = pos_data["position"]
                
                if pos_val == 0:
                    if metrics['zscore'] < -ENTRY_Z:
                        # Enter Long Spread
                        positions[pair_name] = {
                            "position": 1,
                            "entry_price_a": metrics['price_a'],
                            "entry_price_b": metrics['price_b'],
                            "entry_spread": metrics['spread'],
                            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        save_positions(positions)
                    elif metrics['zscore'] > ENTRY_Z:
                        # Enter Short Spread
                        positions[pair_name] = {
                            "position": -1,
                            "entry_price_a": metrics['price_a'],
                            "entry_price_b": metrics['price_b'],
                            "entry_spread": metrics['spread'],
                            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        save_positions(positions)
                else:
                    # Check exits
                    if pos_val == 1 and metrics['zscore'] >= EXIT_Z:
                        # Exit Long Spread
                        qty_a = LOTS_PER_LEG * PAIRS[pair_name]['lot_a']
                        qty_b = LOTS_PER_LEG * PAIRS[pair_name]['lot_b']
                        pnl_a = qty_a * (metrics['price_a'] - pos_data['entry_price_a'])
                        pnl_b = qty_b * (pos_data['entry_price_b'] - metrics['price_b'])
                        gross_pnl = pnl_a + pnl_b
                        entry_value = (qty_a * pos_data['entry_price_a']) + (qty_b * pos_data['entry_price_b'])
                        exit_value = (qty_a * metrics['price_a']) + (qty_b * metrics['price_b'])
                        total_costs = (entry_value + exit_value) * COST_PCT
                        net_pnl_inr = gross_pnl - total_costs
                        pnl_pct = (net_pnl_inr / entry_value) * 100
                        log_pairs_trade(
                            pair_name, "LONG_SPREAD", pos_data["entry_time"],
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            pos_data["entry_price_a"], pos_data["entry_price_b"],
                            metrics["price_a"], metrics["price_b"],
                            pos_data["entry_spread"], metrics["spread"], pnl_pct
                        )
                        positions[pair_name] = {
                            "position": 0,
                            "entry_price_a": 0.0,
                            "entry_price_b": 0.0,
                            "entry_spread": 0.0,
                            "entry_time": ""
                        }
                        save_positions(positions)
                    elif pos_val == -1 and metrics['zscore'] <= -EXIT_Z:
                        # Exit Short Spread
                        qty_a = LOTS_PER_LEG * PAIRS[pair_name]['lot_a']
                        qty_b = LOTS_PER_LEG * PAIRS[pair_name]['lot_b']
                        pnl_a = qty_a * (pos_data['entry_price_a'] - metrics['price_a'])
                        pnl_b = qty_b * (metrics['price_b'] - pos_data['entry_price_b'])
                        gross_pnl = pnl_a + pnl_b
                        entry_value = (qty_a * pos_data['entry_price_a']) + (qty_b * pos_data['entry_price_b'])
                        exit_value = (qty_a * metrics['price_a']) + (qty_b * metrics['price_b'])
                        total_costs = (entry_value + exit_value) * COST_PCT
                        net_pnl_inr = gross_pnl - total_costs
                        pnl_pct = (net_pnl_inr / entry_value) * 100
                        log_pairs_trade(
                            pair_name, "SHORT_SPREAD", pos_data["entry_time"],
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            pos_data["entry_price_a"], pos_data["entry_price_b"],
                            metrics["price_a"], metrics["price_b"],
                            pos_data["entry_spread"], metrics["spread"], pnl_pct
                        )
                        positions[pair_name] = {
                            "position": 0,
                            "entry_price_a": 0.0,
                            "entry_price_b": 0.0,
                            "entry_spread": 0.0,
                            "entry_time": ""
                        }
                        save_positions(positions)
            
            # 3. Print Dashboard
            print_dashboard(positions, all_metrics)
            
        except KeyboardInterrupt:
            print("\nExiting Pairs Monitor. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {YELLOW}WARNING: Connection or data processing error: {e}{RESET}")
            print("Retrying in 60 seconds...")
            
        time.sleep(60)

if __name__ == "__main__":
    main()
