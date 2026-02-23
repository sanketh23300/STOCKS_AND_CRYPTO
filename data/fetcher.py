# ============================================================
#   data/fetcher.py  –  Stock & Crypto data ingestion
# ============================================================

import os, time, json, requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COINGECKO_BASE, CRYPTO_IDS, DATA_CACHE_DIR, DEFAULT_PERIOD, DEFAULT_INTERVAL, CRYPTO_DAYS


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────
def _cache_path(symbol: str, kind: str) -> str:
    return os.path.join(DATA_CACHE_DIR, f"{symbol}_{kind}.csv")


def _load_cache(symbol: str, kind: str, max_age_hours: int = 6) -> pd.DataFrame | None:
    path = _cache_path(symbol, kind)
    if not os.path.exists(path):
        return None
    age = (time.time() - os.path.getmtime(path)) / 3600
    if age > max_age_hours:
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df if not df.empty else None


def _save_cache(df: pd.DataFrame, symbol: str, kind: str):
    df.to_csv(_cache_path(symbol, kind))


# ─────────────────────────────────────────────────────────
#  Stock Fetcher  (Yahoo Finance)
# ─────────────────────────────────────────────────────────
def fetch_stock_data(
    symbol: str,
    period: str    = DEFAULT_PERIOD,
    interval: str  = DEFAULT_INTERVAL,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data for a stock ticker via yfinance.
    Returns a clean DataFrame with columns:
        Open, High, Low, Close, Volume
    """
    cache_key = f"{period}_{interval}"
    if use_cache:
        cached = _load_cache(symbol, cache_key)
        if cached is not None:
            return cached

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = df.index.tz_localize(None) if df.index.tzinfo is None else df.index.tz_convert(None)
        df.index.name = "Date"
        df.dropna(inplace=True)

        _save_cache(df, symbol, cache_key)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch stock data for {symbol}: {e}")


def fetch_stock_info(symbol: str) -> dict:
    """Return basic company info (name, sector, market cap …)."""
    try:
        info = yf.Ticker(symbol).info
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
    Falls back to CoinGecko market_chart for daily Close+Volume if yfinance fails.
    Returns the same schema as fetch_stock_data.
    """
    cache_key = f"crypto_{period}"
    if use_cache:
        cached = _load_cache(symbol, cache_key)
        if cached is not None:
            return cached

    # ── Primary: yfinance (BTC-USD, ETH-USD …) ───────────
    yf_ticker = f"{symbol.upper()}-USD"
    try:
        ticker = yf.Ticker(yf_ticker)
        df = ticker.history(period=period, interval="1d")
        if not df.empty and len(df) > 50:
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = df.index.tz_localize(None)
            df.index.name = "Date"
            df.dropna(inplace=True)
            _save_cache(df, symbol, cache_key)
            return df
    except Exception:
        pass

    # ── Fallback: CoinGecko market_chart (daily Close+Volume) ──
    coin_id = CRYPTO_IDS.get(symbol.upper(), symbol.lower())
    url = (
        f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
        f"?vs_currency=usd&days={days}&interval=daily"
    )
    try:
        resp = requests.get(url, timeout=15)
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

        # Approximate Open/High/Low from Close (limited without real OHLC)
        df["Open"]  = df["Close"].shift(1).fillna(df["Close"])
        df["High"]  = df["Close"] * 1.005
        df["Low"]   = df["Close"] * 0.995
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)

        _save_cache(df, symbol, cache_key)
        return df

    except requests.exceptions.HTTPError as e:
        if "429" in str(e):
            raise RuntimeError("CoinGecko rate limit hit. Please wait a minute and retry.")
        raise RuntimeError(f"Failed to fetch crypto data for {symbol}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch crypto data for {symbol}: {e}")


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
            "name":             data.get("name", symbol),
            "symbol":           data.get("symbol", "").upper(),
            "rank":             data.get("market_cap_rank", "N/A"),
            "market_cap":       md.get("market_cap", {}).get("usd", 0),
            "current_price":    md.get("current_price", {}).get("usd", 0),
            "ath":              md.get("ath", {}).get("usd", 0),
            "atl":              md.get("atl", {}).get("usd", 0),
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
