# ============================================================
#   models/lstm_model.py  –  SAFE LSTM price forecasting
# ============================================================

import os
import numpy as np

from config import SEQUENCE_LENGTH, PREDICTION_DAYS, RANDOM_STATE

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def _check_tf():
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed")


# ── Model ────────────────────────────────────────────────
def build_lstm_model(seq_len, n_features, units=64, dropout=0.2):
    _check_tf()
    tf.random.set_seed(RANDOM_STATE)

    model = Sequential([
        Bidirectional(LSTM(units, return_sequences=True),
                      input_shape=(seq_len, n_features)),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ── Train ────────────────────────────────────────────────
def train_lstm(
    X_train, y_train,
    X_val=None, y_val=None,
    units=64, dropout=0.2,
    epochs=80, batch=32,
):
    _check_tf()

    if X_train.ndim != 3:
        raise ValueError(f"LSTM expects 3D input, got {X_train.shape}")

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]

    model = build_lstm_model(seq_len, n_features, units, dropout)

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        epochs=epochs,
        batch_size=batch,
        callbacks=callbacks,
        verbose=0,
    )

    return model, {}


# ── Predict future ───────────────────────────────────────
def predict_next_n_days(model, last_sequence, scaler, n_days=PREDICTION_DAYS):
    seq = last_sequence.reshape(-1, 1)
    preds = []
    history = list(seq.flatten())

    for _ in range(n_days):
        x = np.array(history[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, 1)
        pred = model.predict(x, verbose=0)[0, 0]
        preds.append(pred)
        history.append(pred)

    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds).flatten()


# ── Evaluate ─────────────────────────────────────────────
def evaluate_lstm(model, X_test, y_test, scaler):
    preds_scaled = model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = np.mean(np.abs(preds - actual))
    rmse = np.sqrt(np.mean((preds - actual) ** 2))
    direction = np.mean(
        np.sign(preds[1:] - preds[:-1]) == np.sign(actual[1:] - actual[:-1])
    )

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "Direction Acc": round(float(direction), 4),
        "Predictions": preds,
    }