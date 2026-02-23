# ============================================================
#   data/preprocessor.py  –  Feature-ready preprocessing
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import SEQUENCE_LENGTH, TRAIN_SPLIT


# ─────────────────────────────────────────────────────────
#  Label generation
# ─────────────────────────────────────────────────────────
def create_labels(df: pd.DataFrame, lookahead: int = 5, threshold: float = 0.015) -> pd.Series:
    """
    Create multi-class labels using forward return:
        2 → BUY   (price rises > threshold over lookahead days)
        0 → SELL  (price falls > threshold over lookahead days)
        1 → HOLD
    Default: 5-day lookahead, 1.5% threshold — gives balanced classes.
    """
    future_return = df["Close"].shift(-lookahead) / df["Close"] - 1
    labels = pd.Series(1, index=df.index, name="Label")
    labels[future_return >  threshold] = 2
    labels[future_return < -threshold] = 0
    labels.dropna(inplace=True)
    return labels


# ─────────────────────────────────────────────────────────
#  Scaler helpers
# ─────────────────────────────────────────────────────────
def fit_price_scaler(series: pd.Series | np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    """Scale a price series to [0, 1]. Returns (scaled, scaler)."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = np.array(series).reshape(-1, 1)
    scaled = scaler.fit_transform(values)
    return scaled, scaler


def transform_price(scaler: MinMaxScaler, series: pd.Series | np.ndarray) -> np.ndarray:
    values = np.array(series).reshape(-1, 1)
    return scaler.transform(values)


def inverse_transform_price(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).flatten()


# ─────────────────────────────────────────────────────────
#  LSTM sequence builder
# ─────────────────────────────────────────────────────────
def build_sequences(
    data: np.ndarray,
    seq_len: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1-D (or 2-D) scaled array into
    (X, y) sequences for LSTM training.
    """
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len : i])
        y.append(data[i, 0] if data.ndim > 1 else data[i])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────
#  ML feature matrix builder
# ─────────────────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame, target_col: str = "Label") -> tuple[pd.DataFrame, pd.Series]:
    """
    Drop rows with NaN, separate X (features) and y (labels).
    Assumes df already has indicator columns appended.
    Excludes raw OHLCV + Close from features to prevent data leakage.
    """
    df = df.copy().dropna()

    # Exclude raw OHLCV, Close (price leakage), and the label
    exclude = {"Open", "High", "Low", "Close", "Volume", target_col}
    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_cols]
    y = df[target_col] if target_col in df.columns else None
    return X, y


# ─────────────────────────────────────────────────────────
#  Train / test split  (time-aware – no shuffling)
# ─────────────────────────────────────────────────────────
def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    split_ratio: float = TRAIN_SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split = int(len(X) * split_ratio)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ─────────────────────────────────────────────────────────
#  Normalize feature matrix  (StandardScaler)
# ─────────────────────────────────────────────────────────
def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ─────────────────────────────────────────────────────────
#  Returns & volatility helpers
# ─────────────────────────────────────────────────────────
def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return_1d"]  = df["Close"].pct_change(1)
    df["Return_5d"]  = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["Return_20d"] = df["Close"].pct_change(20)
    df["Volatility"] = df["Return_1d"].rolling(20).std()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df
