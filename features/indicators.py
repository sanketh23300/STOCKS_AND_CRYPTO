# ============================================================
#   features/indicators.py  –  Technical Indicator Suite
# ============================================================

import numpy as np
import pandas as pd
import ta
from config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, EMA_PERIODS, SMA_PERIODS,
)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and attach all technical indicators to *df*.
    Requires columns: Open, High, Low, Close, Volume.
    Returns a new DataFrame with indicator columns appended.
    """
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Trend ─────────────────────────────────────────────
    for p in SMA_PERIODS:
        df[f"SMA_{p}"] = ta.trend.sma_indicator(close, window=p)

    for p in EMA_PERIODS:
        df[f"EMA_{p}"] = ta.trend.ema_indicator(close, window=p)

    macd_obj = ta.trend.MACD(close, window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df["MACD"]        = macd_obj.macd()
    df["MACD_Signal"] = macd_obj.macd_signal()
    df["MACD_Hist"]   = macd_obj.macd_diff()

    df["ADX"]  = ta.trend.adx(high, low, close, window=14)
    df["CCI"]  = ta.trend.cci(high, low, close, window=20)
    df["VORTEX_POS"] = ta.trend.vortex_indicator_pos(high, low, close, window=14)
    df["VORTEX_NEG"] = ta.trend.vortex_indicator_neg(high, low, close, window=14)

    # ── Momentum ──────────────────────────────────────────
    df["RSI"]       = ta.momentum.rsi(close, window=RSI_PERIOD)
    df["STOCH_K"]   = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
    df["STOCH_D"]   = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)
    df["Williams_R"] = ta.momentum.williams_r(high, low, close, lbp=14)
    df["ROC"]       = ta.momentum.roc(close, window=12)
    df["TRIX"]      = ta.trend.trix(close, window=15)

    # ── Volatility ────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=BB_PERIOD, window_dev=BB_STD)
    df["BB_Upper"]  = bb.bollinger_hband()
    df["BB_Mid"]    = bb.bollinger_mavg()
    df["BB_Lower"]  = bb.bollinger_lband()
    df["BB_Width"]  = bb.bollinger_wband()
    df["BB_Pct"]    = bb.bollinger_pband()

    df["ATR"]   = ta.volatility.average_true_range(high, low, close, window=14)
    df["DC_H"]  = ta.volatility.donchian_channel_hband(high, low, close, window=20)
    df["DC_L"]  = ta.volatility.donchian_channel_lband(high, low, close, window=20)

    # ── Volume ────────────────────────────────────────────
    df["OBV"]     = ta.volume.on_balance_volume(close, vol)
    df["CMF"]     = ta.volume.chaikin_money_flow(high, low, close, vol, window=20)
    df["MFI"]     = ta.volume.money_flow_index(high, low, close, vol, window=14)
    df["VWAP"]    = ta.volume.volume_weighted_average_price(high, low, close, vol, window=14)
    df["EOM"]     = ta.volume.ease_of_movement(high, low, vol, window=14)
    df["FI"]      = ta.volume.force_index(close, vol, window=13)
    df["NVI"]     = ta.volume.negative_volume_index(close, vol)

    # ── Price-derived ─────────────────────────────────────
    df["Price_Range"]     = high - low
    df["Price_Change"]    = close.diff(1)
    df["Pct_Change"]      = close.pct_change(1)
    df["Close_to_High"]   = close / high
    df["Close_to_Low"]    = close / low
    df["High_Low_Ratio"]  = high / low

    # ── Cross-signals ─────────────────────────────────────
    df["EMA9_above_EMA21"]   = (df["EMA_9"]  > df["EMA_21"]).astype(int)
    df["EMA21_above_EMA50"]  = (df["EMA_21"] > df["EMA_50"]).astype(int)
    df["Price_above_EMA50"]  = (close > df["EMA_50"]).astype(int)
    ma200_ref = df["SMA_200"] if "SMA_200" in df.columns else (df["EMA_200"] if "EMA_200" in df.columns else close)
    df["Price_above_SMA200"] = (close > ma200_ref).astype(int)
    df["MACD_positive"]      = (df["MACD"] > df["MACD_Signal"]).astype(int)
    df["RSI_overbought"]     = (df["RSI"] > 70).astype(int)
    df["RSI_oversold"]       = (df["RSI"] < 30).astype(int)

    return df


def get_signal_summary(df: pd.DataFrame) -> dict:
    """
    Derive a simple BUY / SELL / HOLD summary from the last row
    of an indicator-enriched DataFrame.
    """
    last = df.iloc[-1]
    signals = {}

    # RSI
    if last.get("RSI", 50) < 30:
        signals["RSI"] = ("BUY", "RSI oversold")
    elif last.get("RSI", 50) > 70:
        signals["RSI"] = ("SELL", "RSI overbought")
    else:
        signals["RSI"] = ("HOLD", "RSI neutral")

    # MACD
    if last.get("MACD", 0) > last.get("MACD_Signal", 0):
        signals["MACD"] = ("BUY", "MACD bullish crossover")
    else:
        signals["MACD"] = ("SELL", "MACD bearish crossover")

    # Bollinger Bands
    if last.get("BB_Pct", 0.5) < 0.05:
        signals["BB"] = ("BUY", "Price near lower band")
    elif last.get("BB_Pct", 0.5) > 0.95:
        signals["BB"] = ("SELL", "Price near upper band")
    else:
        signals["BB"] = ("HOLD", "Price within bands")

    # ADX
    if last.get("ADX", 20) > 25:
        signals["ADX"] = ("BUY", "Strong trend (ADX>25)")
    else:
        signals["ADX"] = ("HOLD", "Weak trend (ADX≤25)")

    # EMA trend
    if last.get("EMA9_above_EMA21", 0) and last.get("EMA21_above_EMA50", 0):
        signals["EMA"] = ("BUY", "Bullish EMA alignment")
    elif not last.get("EMA9_above_EMA21", 1) and not last.get("EMA21_above_EMA50", 1):
        signals["EMA"] = ("SELL", "Bearish EMA alignment")
    else:
        signals["EMA"] = ("HOLD", "Mixed EMA signals")

    # Composite
    buys  = sum(1 for s, _ in signals.values() if s == "BUY")
    sells = sum(1 for s, _ in signals.values() if s == "SELL")
    if   buys  > sells + 1: overall = "BUY"
    elif sells > buys  + 1: overall = "SELL"
    else:                    overall = "HOLD"

    signals["OVERALL"] = (overall, f"{buys} BUY, {sells} SELL signals")
    return signals
