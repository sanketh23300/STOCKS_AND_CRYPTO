# ============================================================
#   models/lstm_model.py  –  LSTM Price-Forecasting Model
# ============================================================

import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEQUENCE_LENGTH, PREDICTION_DAYS, RANDOM_STATE, MODELS_DIR

# ── Lazy-import TensorFlow so the app works even if TF is absent ──
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model as keras_load
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def _check_tf():
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed. "
            "Run:  pip install tensorflow"
        )


# ─────────────────────────────────────────────────────────
#  Architecture
# ─────────────────────────────────────────────────────────
def build_lstm_model(
    seq_len: int   = SEQUENCE_LENGTH,
    n_features: int = 1,
    units: int     = 64,
    dropout: float = 0.2,
) -> "tf.keras.Model":
    _check_tf()
    tf.random.set_seed(RANDOM_STATE)

    model = Sequential([
        Bidirectional(LSTM(units, return_sequences=True),
                      input_shape=(seq_len, n_features)),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────
def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray | None = None,
    y_val:   np.ndarray | None = None,
    units:   int   = 64,
    dropout: float = 0.2,
    epochs:  int   = 80,
    batch:   int   = 32,
    verbose: int   = 0,
) -> tuple["tf.keras.Model", dict]:
    _check_tf()

    # Reshape to (samples, seq_len, features) if needed
    if X_train.ndim == 2:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    seq_len    = X_train.shape[1]
    n_features = X_train.shape[2]
    model = build_lstm_model(seq_len, n_features, units, dropout)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    val_data = None
    if X_val is not None and y_val is not None:
        if X_val.ndim == 2:
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        val_data = (X_val, y_val)

    history = model.fit(
        X_train, y_train,
        validation_data = val_data,
        epochs          = epochs,
        batch_size      = batch,
        callbacks       = callbacks,
        verbose         = verbose,
    )
    return model, history.history


# ─────────────────────────────────────────────────────────
#  Forecasting
# ─────────────────────────────────────────────────────────
def predict_next_n_days(
    model,
    last_sequence: np.ndarray,
    scaler,
    n_days: int = PREDICTION_DAYS,
) -> np.ndarray:
    """
    Autoregressively predict the next *n_days* closing prices.
    last_sequence: shape (seq_len,) or (seq_len, 1) – scaled values
    Returns: array of shape (n_days,) in original price scale.
    """
    _check_tf()
    seq = last_sequence.reshape(-1, 1) if last_sequence.ndim == 1 else last_sequence
    seq = seq[-SEQUENCE_LENGTH:]  # ensure correct length

    preds_scaled = []
    current_seq  = list(seq.flatten())

    for _ in range(n_days):
        inp  = np.array(current_seq[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, 1)
        pred = model.predict(inp, verbose=0)[0, 0]
        preds_scaled.append(pred)
        current_seq.append(pred)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()


def evaluate_lstm(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
) -> dict:
    _check_tf()
    if X_test.ndim == 2:
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    preds_scaled = model.predict(X_test, verbose=0)
    preds        = scaler.inverse_transform(preds_scaled).flatten()
    actual       = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae   = np.mean(np.abs(preds - actual))
    rmse  = np.sqrt(np.mean((preds - actual) ** 2))
    mape  = np.mean(np.abs((actual - preds) / (actual + 1e-8))) * 100
    direction_acc = np.mean(
        np.sign(preds[1:] - preds[:-1]) == np.sign(actual[1:] - actual[:-1])
    )

    return {
        "MAE":           round(float(mae), 4),
        "RMSE":          round(float(rmse), 4),
        "MAPE(%)":       round(float(mape), 4),
        "Direction Acc": round(float(direction_acc), 4),
        "Predictions":   preds,
        "Actual":        actual,
    }


# ─────────────────────────────────────────────────────────
#  Persistence
# ─────────────────────────────────────────────────────────
def save_lstm(model, symbol: str):
    path = os.path.join(MODELS_DIR, f"{symbol}_lstm.keras")
    model.save(path)
    print(f"  LSTM saved → {path}")


def load_lstm(symbol: str):
    _check_tf()
    path = os.path.join(MODELS_DIR, f"{symbol}_lstm.keras")
    if os.path.exists(path):
        return keras_load(path)
    return None
