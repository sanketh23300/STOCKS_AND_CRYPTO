# ============================================================
# models/lstm_model.py — SAFE & CRASH-PROOF
# ============================================================

import os
import numpy as np

from config import SEQUENCE_LENGTH, PREDICTION_DAYS, RANDOM_STATE, MODELS_DIR

# ── Lazy TensorFlow import ────────────────────────────────
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def _check_tf():
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not installed")


# ─────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────
def build_lstm_model(seq_len: int, n_features: int):
    tf.random.set_seed(RANDOM_STATE)

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────────────
# Training (SAFE)
# ─────────────────────────────────────────────────────────
def train_lstm(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=50,
    batch_size=32,
):
    _check_tf()

    # 🚨 SAFETY CHECK
    if X_train is None or len(X_train) < 10:
        raise ValueError("Not enough data to train LSTM")

    # Ensure 3D shape
    if X_train.ndim == 2:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]

    model = build_lstm_model(seq_len, n_feat)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    return model, history.history


# ─────────────────────────────────────────────────────────
# Forecasting (SAFE)
# ─────────────────────────────────────────────────────────
def predict_next_n_days(model, last_sequence, scaler, n_days=PREDICTION_DAYS):
    _check_tf()

    if last_sequence is None or len(last_sequence) < SEQUENCE_LENGTH:
        return np.array([])

    seq = last_sequence[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)

    preds = []
    for _ in range(n_days):
        p = model.predict(seq, verbose=0)[0, 0]
        preds.append(p)
        seq = np.append(seq[:, 1:, :], [[[p]]], axis=1)

    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds).flatten()# ============================================================
# models/lstm_model.py — SAFE & CRASH-PROOF
# ============================================================

import os
import numpy as np

from config import SEQUENCE_LENGTH, PREDICTION_DAYS, RANDOM_STATE, MODELS_DIR

# ── Lazy TensorFlow import ────────────────────────────────
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def _check_tf():
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not installed")


# ─────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────
def build_lstm_model(seq_len: int, n_features: int):
    tf.random.set_seed(RANDOM_STATE)

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────────────
# Training (SAFE)
# ─────────────────────────────────────────────────────────
def train_lstm(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=50,
    batch_size=32,
):
    _check_tf()

    # 🚨 SAFETY CHECK
    if X_train is None or len(X_train) < 10:
        raise ValueError("Not enough data to train LSTM")

    # Ensure 3D shape
    if X_train.ndim == 2:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]

    model = build_lstm_model(seq_len, n_feat)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    return model, history.history


# ─────────────────────────────────────────────────────────
# Forecasting (SAFE)
# ─────────────────────────────────────────────────────────
def predict_next_n_days(model, last_sequence, scaler, n_days=PREDICTION_DAYS):
    _check_tf()

    if last_sequence is None or len(last_sequence) < SEQUENCE_LENGTH:
        return np.array([])

    seq = last_sequence[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)

    preds = []
    for _ in range(n_days):
        p = model.predict(seq, verbose=0)[0, 0]
        preds.append(p)
        seq = np.append(seq[:, 1:, :], [[[p]]], axis=1)

    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds).flatten()