import os
import sys
import json
import gzip
import csv
import io
import re
import urllib.request
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
FUTURES_KEYS_FILE = SCRIPT_DIR / "futures_keys.json"
POSITION_FILE = SCRIPT_DIR / "live_pairs_positions.json"
TRADES_CSV = SCRIPT_DIR / "live_pairs_trades.csv"

# ANSI Colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"

def fetch_latest_active_futures():
    """Downloads Upstox instrument master and finds nearest active monthly futures"""
    url = 'https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz'
    print(f"{CYAN}[ROLLOVER] Downloading latest Upstox instrument master...{RESET}")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        data = io.BytesIO(resp.read())

    today = datetime.now().strftime('%Y-%m-%d')
    futures = {}

    with gzip.open(data, 'rt', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['instrument_type'] == 'FUTSTK' and row['exchange'] == 'NSE_FO':
                ts = row['tradingsymbol']
                m = re.match(r'^(.+?)(\d{2}[A-Z]{3}FUT)$', ts)
                if m:
                    sym = m.group(1)
                    if row['expiry'] >= today:
                        if sym not in futures:
                            futures[sym] = []
                        futures[sym].append({
                            'key': row['instrument_key'],
                            'expiry': row['expiry'],
                            'lot': int(row['lot_size']),
                            'symbol': ts
                        })

    out = {}
    for s, contracts in futures.items():
        contracts.sort(key=lambda x: x['expiry'])
        out[s] = contracts[0]

    with open(FUTURES_KEYS_FILE, "w") as f:
        json.dump(out, f, indent=4)
        
    print(f"{GREEN}[ROLLOVER] Successfully updated {FUTURES_KEYS_FILE} with active contracts.{RESET}")
    return out

def get_active_futures_map():
    """Reads futures_keys.json, and if expired or missing, refreshes from Upstox"""
    today = datetime.now().strftime('%Y-%m-%d')
    need_refresh = False
    
    if not FUTURES_KEYS_FILE.exists():
        need_refresh = True
    else:
        try:
            with open(FUTURES_KEYS_FILE, 'r') as f:
                data = json.load(f)
            # check if any expiry is in the past
            for sym, details in data.items():
                if details.get('expiry', '2000-01-01') <= today:
                    need_refresh = True
                    break
        except Exception:
            need_refresh = True
            
    if need_refresh:
        return fetch_latest_active_futures()
    else:
        with open(FUTURES_KEYS_FILE, 'r') as f:
            return json.load(f)

def square_off_expired_positions(positions, pairs_config, fetcher=None):
    """
    Checks all open positions. If current time is past expiry or on expiry day >= 15:15,
    automatically squares off the position and logs the trade to CSV.
    """
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    is_expiry_close_time = (now.hour > 15) or (now.hour == 15 and now.minute >= 15)
    
    positions_modified = False
    
    for pair_name, pos_data in list(positions.items()):
        pos_val = pos_data.get("position", 0)
        if pos_val == 0:
            continue
            
        pair_cfg = pairs_config.get(pair_name)
        if not pair_cfg:
            continue
            
        expiry_date = pair_cfg.get("expiry", "2000-01-01")
        is_expired = (today_str > expiry_date) or (today_str == expiry_date and is_expiry_close_time)
        
        if is_expired:
            print(f"\n{YELLOW}{BOLD}[EXPIRY AUTO-SQUAREOFF]{RESET} Position in {BOLD}{pair_name}{RESET} reached expiry ({expiry_date})! Squaring off...")
            
            # Fetch settlement / exit prices
            exit_price_a = pos_data["entry_price_a"]
            exit_price_b = pos_data["entry_price_b"]
            
            # Try fetching quote if still available
            if fetcher:
                try:
                    qa = fetcher.get_quote(pair_cfg['key_a'])
                    qb = fetcher.get_quote(pair_cfg['key_b'])
                    # extract LTP
                    for k, v in qa.get("data", {}).items():
                        exit_price_a = float(v.get("last_price", exit_price_a))
                    for k, v in qb.get("data", {}).items():
                        exit_price_b = float(v.get("last_price", exit_price_b))
                except Exception:
                    pass
            
            qty_a = pair_cfg.get('lot_a', 1)
            qty_b = pair_cfg.get('lot_b', 1)
            
            if pos_val == 1:
                # Long Spread: Buy A / Sell B at entry -> Sell A / Buy B at exit
                pnl_a = qty_a * (exit_price_a - pos_data['entry_price_a'])
                pnl_b = qty_b * (pos_data['entry_price_b'] - exit_price_b)
                trade_type = "LONG_SPREAD (EXPIRY_CLOSE)"
            else:
                # Short Spread: Sell A / Buy B at entry -> Buy A / Sell B at exit
                pnl_a = qty_a * (pos_data['entry_price_a'] - exit_price_a)
                pnl_b = qty_b * (exit_price_b - pos_data['entry_price_b'])
                trade_type = "SHORT_SPREAD (EXPIRY_CLOSE)"
                
            gross_pnl = pnl_a + pnl_b
            entry_val = (qty_a * pos_data['entry_price_a']) + (qty_b * pos_data['entry_price_b'])
            exit_val = (qty_a * exit_price_a) + (qty_b * exit_price_b)
            costs = (entry_val + exit_val) * 0.00005
            net_pnl = gross_pnl - costs
            pnl_pct = (net_pnl / entry_val) * 100 if entry_val > 0 else 0.0
            
            exit_spread = np.log(exit_price_a) - np.log(exit_price_b)
            exit_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            
            # Log to CSV
            file_exists = TRADES_CSV.exists()
            with open(TRADES_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["pair", "type", "entry_time", "exit_time", "entry_price_a", "entry_price_b", "exit_price_a", "exit_price_b", "entry_spread", "exit_spread", "pnl_pct", "expiry_a", "expiry_b"])
                writer.writerow([
                    pair_name, trade_type, pos_data["entry_time"], exit_time_str,
                    pos_data["entry_price_a"], pos_data["entry_price_b"],
                    exit_price_a, exit_price_b,
                    pos_data["entry_spread"], exit_spread, f"{pnl_pct:.2f}",
                    expiry_date, expiry_date
                ])
                
            print(f"  Closed {pair_name}: Net PnL = Rs. {net_pnl:+.2f} ({pnl_pct:+.2f}%) logged to tradebook.")
            
            # Reset position
            positions[pair_name] = {
                "position": 0,
                "entry_price_a": 0.0,
                "entry_price_b": 0.0,
                "entry_spread": 0.0,
                "entry_time": ""
            }
            positions_modified = True
            
    if positions_modified:
        with open(POSITION_FILE, 'w') as f:
            json.dump(positions, f, indent=4)
        print(f"{GREEN}[EXPIRY AUTO-SQUAREOFF] All expired positions reset to FLAT.{RESET}\n")

def sync_pairs_config(pairs_config):
    """
    Checks the PAIRS config and updates any expired keys with active futures.
    Returns the refreshed PAIRS dictionary and whether it changed.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    active_futures = get_active_futures_map()
    
    updated = False
    for pair_name, cfg in pairs_config.items():
        sym_a = cfg['symbol_a']
        sym_b = cfg['symbol_b']
        current_exp = cfg.get('expiry', '2000-01-01')
        
        if current_exp <= today or sym_a not in active_futures or sym_b not in active_futures:
            # Need rollover
            if sym_a in active_futures and sym_b in active_futures:
                new_a = active_futures[sym_a]
                new_b = active_futures[sym_b]
                
                cfg['key_a'] = new_a['key']
                cfg['key_b'] = new_b['key']
                cfg['lot_a'] = new_a['lot']
                cfg['lot_b'] = new_b['lot']
                cfg['expiry'] = new_a['expiry']
                updated = True
                print(f"{CYAN}[ROLLOVER] {pair_name} rolled over to expiry {new_a['expiry']} (Keys: {new_a['key']}, {new_b['key']}){RESET}")
                
    return pairs_config, updated
