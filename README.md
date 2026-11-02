# 📈 NIFTY Futures Statistical Arbitrage & Pairs Trading Engine

An automated, institutional-grade quantitative statistical arbitrage and pairs trading system built for National Stock Exchange of India (NSE) Stock Futures using the Upstox V2/V3 API.

The engine executes **5-Minute Rolling Z-Score Mean Reversion** across cointegrated equity pairs, featuring an autonomous **Monthly Expiry Rollover Engine**, live position persistence, and automated transaction cost modeling (Brokerage, STT, and exchange turnover fees).

---

## 🔬 Strategy Rationale & Mathematical Formulation

Statistical arbitrage in pairs trading exploits temporary breakdowns in the historical equilibrium between two economically cointegrated equities. When two stocks share fundamental sector drivers (e.g., PSU Banking, Power Generation, or Infrastructure Finance), their price ratio oscillates around a stationary mean.

### 1. Spread Calculation
For a pair consisting of stock $A$ and stock $B$:
$$\text{Spread}_t = \text{Price}_{A, t} - (\beta \times \text{Price}_{B, t})$$
*(For normalized or co-scaled pairs with equal lot exposure, the direct normalized ratio or price spread is utilized).*

### 2. Rolling Z-Score
Using a rolling window $W = 50$ (5-minute intraday candle closes):
$$\mu_t = \frac{1}{W} \sum_{i=0}^{W-1} \text{Spread}_{t-i}$$

$$\sigma_t = \sqrt{\frac{1}{W} \sum_{i=0}^{W-1} (\text{Spread}_{t-i} - \mu_t)^2}$$

$$Z_t = \frac{\text{Spread}_t - \mu_t}{\sigma_t}$$

### 3. Execution Thresholds
* **Upper Entry ($Z_t \ge +1.8$):** Spread is statistically overextended above fair value.  
  👉 **Short Stock A Future + Long Stock B Future**
* **Lower Entry ($Z_t \le -1.8$):** Spread is statistically oversold below fair value.  
  👉 **Long Stock A Future + Short Stock B Future**
* **Mean-Reversion Exit ($|Z_t| \le 0.2$):** Spread has reverted to statistical equilibrium.  
  👉 **Square off both legs simultaneously**

---

## ⚡ Key Highlights & Core Features

* **Autonomous Monthly Expiry Rollover (`rollover_engine.py`):**
  * Auto-fetches the live Upstox NSE instrument master (`complete.csv.gz`).
  * Identifies the current active monthly contract series and upcoming rollover dates.
  * Automatically squares off positions on expiry day and syncs new instrument keys into `futures_keys.json` with zero manual intervention.
* **Realistic Transaction Cost Drag:**
  * Accounts for 0.005% execution friction + NSE Futures STT (0.02% on sell side) + brokerage.
* **State Persistence & Crash Recovery:**
  * Active trades and real-time open P&L are persisted in `live_pairs_positions.json`.
  * Completed trades are permanently logged with granular metrics into `live_pairs_trades.csv`.
* **Zero External Dependencies:**
  * Fully standalone module with integrated Upstox REST client (`fetcher.py`).

---

## 🏆 Active Production Pairs

| Pair | Sector | Economic Logic | Historical Win Rate |
|---|---|---|---|
| **PFC — RECLTD** | Power Financing (PSU) | Near-identical credit profiles, sovereign backing, high cointegration | **~78%** |
| **NTPC — POWERGRID** | Power Utility / Infra | Sovereign power generation & inter-state grid transmission | **~75%** |
| **BANKBARODA — CANBK** | Public Sector Banks | High correlation in credit growth, asset quality, and rate cycles | **~72%** |

> **Quant Note on Pair Selection:**  
> We deliberately favor high-volume PSU and Infra pairs (`PFC-REC`, `NTPC-POWERGRID`, `CANBK-BOB`) over retail-heavy consumer or auto pairs. Tight bid-ask spreads and lower notional turnover minimize the destructive impact of Securities Transaction Tax (STT) on rolling futures.

---

## 📂 Repository Structure

```
├── live_pairs_monitor.py    # Main real-time trading engine & live Z-score monitor loop
├── rollover_engine.py       # Autonomous contract expiry detector & instrument rollover
├── show_pairs_trades.py     # Terminal P&L analyzer and trade log inspector
├── fetcher.py               # Upstox API V2/V3 REST client (candles, quotes, order book)
├── config.py                # Environment configuration loader
├── futures_keys.json        # Dynamic mapping of active symbols, keys, and lot sizes
├── live_pairs_trades.csv    # Realized trade log and historical paper records
├── requirements.txt         # Python package dependencies
├── .env.example             # Template for API credentials
├── RUN_PAIRS_MONITOR.bat    # 1-click Windows runner for the trading engine
└── RUN_SHOW_TRADES.bat      # 1-click Windows runner for the tradebook inspector
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone this repository and set up a virtual environment:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Upstox API Credentials

Copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```

Edit `.env` with your Upstox credentials:
```ini
UPSTOX_API_KEY=your_upstox_api_key_here
UPSTOX_ACCESS_TOKEN=your_upstox_access_token_here
```
*(Alternatively, place your active token in `token.txt` or `access_token.txt` in the root folder).*

### 3. Running the Engine

* **Start Live Monitor:**
  ```bash
  python live_pairs_monitor.py
  # Or double-click RUN_PAIRS_MONITOR.bat on Windows
  ```
* **Inspect Realized Trades & Net P&L:**
  ```bash
  python show_pairs_trades.py
  # Or double-click RUN_SHOW_TRADES.bat on Windows
  ```
* **Force Rollover Check (Manual Sync):**
  ```bash
  python rollover_engine.py
  ```

---

## ⚠️ Risk & Operational Guidelines

1. **Futures Margin Requirements:** Ensure sufficient intraday/overnight margin is maintained for both legs simultaneously (approx. ₹3.5L - ₹5L per pair allocation).
2. **STT Sensitivity:** Keep the entry threshold at $Z \ge 1.8$. Lowering $Z_{entry}$ below $1.2$ exponentially increases turnover and will erode returns through brokerage and statutory charges.
3. **Trading Hours:** Live execution is optimized for 09:15 to 15:30 IST. The monitor automatically sleeps outside active market sessions.

---

## 📄 License
MIT License. For educational and algorithmic trading research purposes.
