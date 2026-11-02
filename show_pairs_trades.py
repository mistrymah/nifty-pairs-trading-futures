import os
import sys
from pathlib import Path
import pandas as pd
from live_pairs_monitor import PAIRS, LOTS_PER_LEG, COST_PCT

# ANSI Terminal Colors
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"

SCRIPT_DIR = Path(__file__).parent.resolve()
TRADES_CSV = SCRIPT_DIR / "live_pairs_trades.csv"

def main():
    # Enable ANSI colors in Windows Command Prompt/PowerShell
    if os.name == 'nt':
        os.system('')
        
    if not TRADES_CSV.exists():
        print(f"{YELLOW}No completed paper trades log found.{RESET}")
        print("Past trades will appear here once they are exited by the live monitor.")
        return
        
    try:
        df = pd.read_csv(TRADES_CSV)
    except Exception as e:
        print(f"{RED}Error reading trade logs: {e}{RESET}")
        return
        
    print(f"{BOLD}============================================================{RESET}")
    print(f"{BOLD}             COMPLETED PAIRS PAPER TRADES LOG (FUTURES)     {RESET}")
    print(f"{BOLD}============================================================{RESET}")
    print(f" Lots Allocation  : {LOTS_PER_LEG} lot(s) per leg")
    print(f" Transaction Costs: {COST_PCT*100:.3f}% per execution")
    print(f"============================================================{RESET}")
    
    total_net_pnl_inr = 0.0
    win_count = 0
    
    for idx, row in df.iterrows():
        pair = row.get('pair', '')
        if not pair:
            # Fallback in case old trades don't have 'pair' column reliably
            pair = f"Trade #{idx+1}"
            
        entry_a = float(row['entry_price_a'])
        exit_a = float(row['exit_price_a'])
        entry_b = float(row['entry_price_b'])
        exit_b = float(row['exit_price_b'])
        pos_type = row['type']
        
        # Calculate quantities based on lots
        if pair in PAIRS:
            qty_a = LOTS_PER_LEG * PAIRS[pair]['lot_a']
            qty_b = LOTS_PER_LEG * PAIRS[pair]['lot_b']
        else:
            # Fallback for old trades if pair removed
            qty_a = 50000.0 / entry_a
            qty_b = 50000.0 / entry_b
        
        # Calculate individual leg gross returns in Rupees
        if "LONG" in pos_type:
            # Long A / Short B
            pnl_a = qty_a * (exit_a - entry_a)
            pnl_b = qty_b * (entry_b - exit_b)
        else:
            # Short A / Long B
            pnl_a = qty_a * (entry_a - exit_a)
            pnl_b = qty_b * (exit_b - entry_b)
            
        gross_pnl = pnl_a + pnl_b
        
        # Compute exact transaction costs in Rupees (4 executions)
        entry_value = (qty_a * entry_a) + (qty_b * entry_b)
        exit_value = (qty_a * exit_a) + (qty_b * exit_b)
        total_costs = (entry_value + exit_value) * COST_PCT
        
        net_pnl_inr = gross_pnl - total_costs
        total_net_pnl_inr += net_pnl_inr
        if net_pnl_inr > 0:
            win_count += 1
            
        pnl_color = GREEN if net_pnl_inr >= 0 else RED
        pct_ret = (net_pnl_inr / entry_value) * 100 if entry_value > 0 else 0
        
        exp_a_str = f" [Exp: {row['expiry_a']}]" if 'expiry_a' in row else ""
        exp_b_str = f" [Exp: {row['expiry_b']}]" if 'expiry_b' in row else ""
        
        print(f" Trade #{idx+1} | {BOLD}{pair}{RESET} | {pos_type}")
        print(f"  Time  : Entry: {row['entry_time']} | Exit: {row['exit_time']}")
        print(f"  Leg A : Price {entry_a:.2f} -> {exit_a:.2f}{exp_a_str} (Qty: {qty_a:.1f} | Gross P&L: {GREEN if pnl_a >= 0 else RED}{pnl_a:+.2f} Rs{RESET})")
        print(f"  Leg B : Price {entry_b:.2f} -> {exit_b:.2f}{exp_b_str} (Qty: {qty_b:.1f} | Gross P&L: {GREEN if pnl_b >= 0 else RED}{pnl_b:+.2f} Rs{RESET})")
        print(f"  Spread: {row['entry_spread']:.4f} -> {row['exit_spread']:.4f}")
        print(f"  Costs : {total_costs:.2f} Rs | {BOLD}Net P&L: {pnl_color}{net_pnl_inr:+.2f} Rs ({pct_ret:+.2f}%){RESET}")
        print("-" * 60)
        
    win_rate = (win_count / len(df) * 100) if len(df) > 0 else 0.0
    
    print(f"{BOLD}============================================================{RESET}")
    print(f" Total Completed Trades: {len(df)}")
    print(f" Win Rate              : {win_count}/{len(df)} ({win_rate:.1f}%)")
    print(f" Cumulative Net P&L    : {GREEN if total_net_pnl_inr >= 0 else RED}{total_net_pnl_inr:+.2f} Rs{RESET}")
    print(f"{BOLD}============================================================{RESET}")

if __name__ == "__main__":
    main()
