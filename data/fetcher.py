# ============================================================
#   data/fetcher.py  –  Stock & Crypto data ingestion
# ============================================================

import os, time, requests
import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COINGECKO_BASE, CRYPTO_IDS, DATA_CACHE_DIR, DEFAULT_PERIOD, DEFAULT_INTERVAL, CRYPTO_DAYS


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────
def _cache_path(symbol: str, kind: str) -> str:
    return os.path.join(DATA_CACHE_DIR, f"{symbol}_{kind}.csv")


def _load_cache(
    symbol: str, kind: str,
    max_age_hours: int = 24,
    allow_stale: bool  = False,
) -> pd.DataFrame | None:
    """
    Load cached OHLCV CSV.
    allow_stale=True returns the file even if older than max_age_hours
    (used as a rate-limit fallback).
    """
    path = _cache_path(symbol, kind)
    if not os.path.exists(path):
        return None
    age = (time.time() - os.path.getmtime(path)) / 3600
    if not allow_stale and age > max_age_hours:
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df if not df.empty else None


def _save_cache(df: pd.DataFrame, symbol: str, kind: str):
    df.to_csv(_cache_path(symbol, kind))


def _retry_fetch(fn, retries: int = 4, base_delay: float = 5.0):
    """
    Call fn(), retrying on rate-limit / network errors with
    exponential backoff: 5s, 10s, 20s, 40s …
    """
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            is_rate = any(k in msg for k in ["too many", "rate", "429", "limit", "throttl"])
            is_net  = any(k in msg for k in ["connection", "timeout", "network", "ssl"])
            if (is_rate or is_net) and attempt < retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  ⚠ Rate-limit / network error for attempt {attempt+1}. "
                      f"Retrying in {delay:.0f}s …")
                time.sleep(delay)
                last_err = e
            else:
                raise
    raise last_err


# ─────────────────────────────────────────────────────────
#  Stock Fetcher  (Yahoo Finance)
# ─────────────────────────────────────────────────────────
def fetch_stock_data(
    symbol: str,
    period: str     = DEFAULT_PERIOD,
    interval: str   = DEFAULT_INTERVAL,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data for a stock ticker via yfinance.
    Retries up to 4 times on rate-limit errors.
    Falls back to stale cache if all retries fail.
    """
    cache_key = f"{period}_{interval}"

    if use_cache:
        cached = _load_cache(symbol, cache_key, max_age_hours=24)
        if cached is not None:
            return cached

    def _do_fetch():
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = (
            df.index.tz_localize(None)
            if df.index.tzinfo is None
            else df.index.tz_convert(None)
        )
        df.index.name = "Date"
        df.dropna(inplace=True)
        return df

    try:
        df = _retry_fetch(_do_fetch)
        _save_cache(df, symbol, cache_key)
        return df
    except Exception as e:
        # Last resort: return stale cache if it exists
        stale = _load_cache(symbol, cache_key, allow_stale=True)
        if stale is not None:
            print(f"  ⚠ Using stale cache for {symbol} (live fetch failed: {e})")
            return stale
        raise RuntimeError(
            f"Failed to fetch stock data for {symbol}: {e}\n"
            "Tip: yfinance is rate-limited. Wait 60 seconds and try again."
        )


def fetch_stock_info(symbol: str) -> dict:
    """Return basic company info (name, sector, market cap …)."""
    try:
        def _do():
            return yf.Ticker(symbol).info
        info = _retry_fetch(_do)
        return {
            "name":       info.get("longName", symbol),
            "sector":     info.get("sector", "N/A"),
            "industry":   info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio":   info.get("trailingPE", None),
            "52w_high":   info.get("fiftyTwoWeekHigh", None),
            "52w_low":    info.get("fiftyTwoWeekLow", None),
            "currency":   info.get("currency", "USD"),
        }
    except Exception:
        return {"name": symbol}


# ─────────────────────────────────────────────────────────
#  Crypto Fetcher  (yfinance – proper daily OHLCV)
# ─────────────────────────────────────────────────────────
def fetch_crypto_data(
    symbol: str,
    days: int       = CRYPTO_DAYS,
    period: str     = "2y",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download daily OHLCV for a cryptocurrency via yfinance (e.g. BTC-USD).
    Falls back to CoinGecko market_chart if yfinance fails.
    Falls back to stale cache if both APIs fail.
    """
    cache_key = f"crypto_{period}"

    if use_cache:
        cached = _load_cache(symbol, cache_key, max_age_hours=24)
        if cached is not None:
            return cached

    # ── Primary: yfinance ─────────────────────────────────
    yf_ticker = f"{symbol.upper()}-USD"

    def _yf_fetch():
        ticker = yf.Ticker(yf_ticker)
        df = ticker.history(period=period, interval="1d")
        if df.empty or len(df) < 50:
            raise ValueError(f"Insufficient yfinance data for {yf_ticker}")
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = df.index.tz_localize(None) if df.index.tzinfo is None else df.index.tz_convert(None)
        df.index.name = "Date"
        df.dropna(inplace=True)
        return df

    try:
        df = _retry_fetch(_yf_fetch)
        _save_cache(df, symbol, cache_key)
        return df
    except Exception as yf_err:
        pass

    # ── Fallback: CoinGecko market_chart ──────────────────
    coin_id = CRYPTO_IDS.get(symbol.upper(), symbol.lower())
    url = (
        f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
        f"?vs_currency=usd&days={days}&interval=daily"
    )

    def _cg_fetch():
        resp = requests.get(url, timeout=15)
        if resp.status_code == 429:
            raise RuntimeError("429 Too Many Requests")
        resp.raise_for_status()
        data = resp.json()
        prices  = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        price_df = pd.DataFrame(prices,  columns=["ts", "Close"])
        vol_df   = pd.DataFrame(volumes, columns=["ts", "Volume"])
        price_df["Date"] = pd.to_datetime(price_df["ts"], unit="ms").dt.normalize()
        vol_df["Date"]   = pd.to_datetime(vol_df["ts"],   unit="ms").dt.normalize()
        df = price_df[["Date", "Close"]].merge(
            vol_df[["Date", "Volume"]], on="Date", how="inner"
        ).set_index("Date")
        df["Open"]  = df["Close"].shift(1).fillna(df["Close"])
        df["High"]  = df["Close"] * 1.005
        df["Low"]   = df["Close"] * 0.995
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        if df.empty:
            raise ValueError("Empty CoinGecko response")
        return df

    try:
        df = _retry_fetch(_cg_fetch)
        _save_cache(df, symbol, cache_key)
        return df
    except Exception as cg_err:
        pass

    # ── Last resort: stale cache ───────────────────────────
    stale = _load_cache(symbol, cache_key, allow_stale=True)
    if stale is not None:
        print(f"  ⚠ Using stale cache for {symbol} (all live sources rate-limited).")
        return stale

    raise RuntimeError(
        f"Failed to fetch crypto data for {symbol} from all sources.\n"
        "Tip: Both yfinance and CoinGecko are rate-limited. Wait 60 seconds and try again."
    )


def fetch_crypto_info(symbol: str) -> dict:
    """Return market cap, rank, supply info for a coin."""
    coin_id = CRYPTO_IDS.get(symbol.upper(), symbol.lower())
    url = f"{COINGECKO_BASE}/coins/{coin_id}?localization=false&tickers=false&community_data=false"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        md = data.get("market_data", {})
        return {
            "name":               data.get("name", symbol),
            "symbol":             data.get("symbol", "").upper(),
            "rank":               data.get("market_cap_rank", "N/A"),
            "market_cap":         md.get("market_cap", {}).get("usd", 0),
            "current_price":      md.get("current_price", {}).get("usd", 0),
            "ath":                md.get("ath", {}).get("usd", 0),
            "atl":                md.get("atl", {}).get("usd", 0),
            "circulating_supply": md.get("circulating_supply", 0),
        }
    except Exception:
        return {"name": symbol}


# ─────────────────────────────────────────────────────────
#  Universal Dispatcher
# ─────────────────────────────────────────────────────────
def fetch_data(symbol: str, asset_type: str = "stock", **kwargs) -> pd.DataFrame:
    """
    Unified entry point.
    asset_type: 'stock' | 'crypto'
    """
    if asset_type.lower() == "crypto":
        return fetch_crypto_data(symbol, **kwargs)
    return fetch_stock_data(symbol, **kwargs)


def fetch_info(symbol: str, asset_type: str = "stock") -> dict:
    if asset_type.lower() == "crypto":
        return fetch_crypto_info(symbol)
    return fetch_stock_info(symbol)
