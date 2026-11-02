import os
import sys
import json
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

def calculate_atr(df, period=14):
    """Calculate 14-day Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr

def main():
    if not HAS_UPSTOX:
        print("ERROR: Could not import UpstoxFetcher.")
        sys.exit(1)
        
    fetcher = UpstoxFetcher()
    
    # Load instrument map from nifty_mcp_upstox
    inst_path = UPSTOX_DIR / "market_data" / "nse_instruments.json"
    if not inst_path.exists():
        print(f"ERROR: Could not find nse_instruments.json at {inst_path.resolve()}")
        sys.exit(1)
        
    with open(inst_path) as f:
        inst_map = json.load(f)
        
    # Universe of 120 highly liquid NSE stocks (mostly F&O constituents)
    universe = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "LTIM", "ITC", "HINDUNILVR",
        "BAJFINANCE", "LT", "MM", "SUNPHARMA", "ADANIENT", "ADANIPORTS", "TATASTEEL", "POWERGRID", "NTPC", "TITAN",
        "TATAMOTORS", "AXISBANK", "HCLTECH", "KOTAKBANK", "ONGC", "COALINDIA", "ULTRACEMCO", "GRASIM", "MARUTI", "INDUSINDBK",
        "JSWSTEEL", "BPCL", "HINDALCO", "NESTLEIND", "APOLLOHOSP", "SHRIRAMFIN", "BAJAJFINSV", "BAJAJ-AUTO", "TATACONSUM", "DRREDDY",
        "WIPRO", "HEROMOTOCO", "TECHM", "EICHERMOT", "SBILIFE", "HDFCLIFE", "BEL", "ADANIPOWER", "BHEL", "JINDALSTEL",
        "TATACOMM", "ASHOKLEY", "BANKBARODA", "CANBK", "PNB", "TRENT", "DLF", "HAL", "GMRINFRA", "TATACHEM",
        "HINDPETRO", "REC", "PFC", "SAIL", "NMDC", "GAIL", "AMBUJACEM", "ACC", "SHREECEM", "MCDOWELL-N",
        "VOLTAS", "JUBLFOOD", "MUTHOOTFIN", "SRF", "UPL", "CONCOR", "LICHSGFIN", "TATAPOWER", "IEX", "ZEEL",
        "HDFCAMC", "WOCKPHARMA", "PIDILITIND", "COFORGE", "PERSISTENT", "MPHASIS", "KPITTECH", "OFSS", "DIVISLAB", "CIPLA",
        "GLAND", "BIOCON", "SYNGENE", "LUPIN", "AUROPHARMA", "JSWENERGY", "TATAELXSI", "BDL", "BHARATFORG", "ESCORTS",
        "CUMMINSIND", "AUBANK", "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "INDHOTEL", "GODREJCP", "DABUR", "COLPAL", "MARICO",
        "VBL", "BRITANNIA", "POLYCAB", "TATAMETALI", "IRCTC", "RVNL", "JIOFIN", "OBEROIRLTY", "HAVELLS", "PIDILITIND"
    ]
    
    universe = list(set(universe)) # De-duplicate
    
    print(f"=== Scanning {len(universe)} Stocks for Volume & ATR (Bhavcopy Mode) ===")
    
    stock_metrics = []
    
    for idx, symbol in enumerate(universe, 1):
        # Look up Upstox instrument key
        key = inst_map.get(symbol)
        if not key:
            # Try with alternate keys or skip
            continue
            
        print(f"[{idx}/{len(universe)}] Fetching EOD data for {symbol} ({key})...")
        try:
            # Fetch 30 days of daily data
            df = fetcher.get_historical(key, "1day", days=30)
            if df.empty or len(df) < 15:
                print(f"  WARNING: Insufficient data for {symbol}.")
                continue
                
            # Calculate 20-day Average Volume and 14-day relative ATR
            df = df.sort_values("timestamps").reset_index(drop=True)
            avg_volume = df['volume'].tail(20).mean()
            
            df['atr'] = calculate_atr(df, period=14)
            latest_close = df['close'].iloc[-1]
            latest_atr = df['atr'].iloc[-1]
            
            if pd.isna(latest_atr) or latest_close == 0:
                continue
                
            atr_pct = (latest_atr / latest_close) * 100
            
            stock_metrics.append({
                "symbol": symbol,
                "key": key,
                "close": latest_close,
                "avg_volume": avg_volume,
                "atr_pct": atr_pct
            })
        except Exception as e:
            print(f"  ERROR fetching {symbol}: {e}")
            
    if not stock_metrics:
        print("ERROR: No stock metrics calculated. Exiting.")
        sys.exit(1)
        
    metrics_df = pd.DataFrame(stock_metrics)
    
    # Ranking Logic
    # We rank descending (higher is better) for volume and atr_pct
    metrics_df['rank_volume'] = metrics_df['avg_volume'].rank(ascending=False)
    metrics_df['rank_atr_pct'] = metrics_df['atr_pct'].rank(ascending=False)
    
    # Combined score (lower rank sum is better)
    metrics_df['combined_rank'] = metrics_df['rank_volume'] + metrics_df['rank_atr_pct']
    
    # Sort and pick top 50
    sorted_df = metrics_df.sort_values(by='combined_rank')
    top_50 = sorted_df.head(50)
    
    print("\n=== TOP 10 SCANNER RESULTS ===")
    for rank, (i, row) in enumerate(top_50.head(10).iterrows(), 1):
        print(f"Rank {rank}: {row['symbol']} | Close: {row['close']:.2f} | Avg Vol: {int(row['avg_volume']):,} | ATR%: {row['atr_pct']:.2f}%")
        
    # Save the selected stocks to selected_stocks.json
    output_path = Path("selected_stocks.json")
    selected_list = top_50[['symbol', 'key', 'close', 'avg_volume', 'atr_pct']].to_dict(orient='records')
    
    with open(output_path, 'w') as f:
        json.dump(selected_list, f, indent=4)
        
    print(f"\nSaved {len(selected_list)} selected stocks to {output_path.resolve()}")

if __name__ == "__main__":
    main()
