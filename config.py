# ============================================================
#   config.py  –  Global Configuration
# ============================================================

import os

# ── Asset Lists ────────────────────────────────────────────
STOCK_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX"]
CRYPTO_SYMBOLS = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT"]

# Crypto full names for CoinGecko
CRYPTO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "DOT": "polkadot",
}

# ── Data Settings ─────────────────────────────────────────
DEFAULT_PERIOD   = "2y"          # yfinance period string
DEFAULT_INTERVAL = "1d"          # yfinance interval
CRYPTO_DAYS      = 730           # days of crypto history

# ── Model Settings ────────────────────────────────────────
SEQUENCE_LENGTH   = 60           # LSTM look-back window (days)
TRAIN_SPLIT       = 0.80         # training / test ratio
PREDICTION_DAYS   = 30           # how many days to forecast
RANDOM_STATE      = 42

# ── Feature Engineering ───────────────────────────────────
RSI_PERIOD       = 14
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
BB_PERIOD        = 20
BB_STD           = 2
EMA_PERIODS      = [9, 21, 50, 200]
SMA_PERIODS      = [10, 20, 50, 200]

# ── Signals ───────────────────────────────────────────────
BUY_THRESHOLD    = 0.55          # probability threshold → BUY
SELL_THRESHOLD   = 0.45          # probability threshold → SELL

# ── Paths ─────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR       = os.path.join(BASE_DIR, "saved_models")
DATA_CACHE_DIR   = os.path.join(BASE_DIR, "data_cache")

os.makedirs(MODELS_DIR,    exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ── CoinGecko API ─────────────────────────────────────────
COINGECKO_BASE   = "https://api.coingecko.com/api/v3"

# ── Streamlit Theme ───────────────────────────────────────
PAGE_TITLE  = "AI Market Trend Analyzer"
PAGE_ICON   = "📈"
LAYOUT      = "wide"
