def train_lstm_forecast(self, epochs: int = 60, units: int = 64):
    try:
        from models.lstm_model import train_lstm, predict_next_n_days, evaluate_lstm
    except ImportError:
        self.lstm_metrics = {"error": "TensorFlow not available"}
        self.forecast = None
        return self.lstm_metrics

    close = self.feat_df["Close"].values

    if len(close) < SEQUENCE_LENGTH * 2:
        self.lstm_metrics = {"error": "Not enough data for LSTM"}
        self.forecast = None
        return self.lstm_metrics

    scaled, self.price_scaler = fit_price_scaler(close)

    split = int(len(scaled) * TRAIN_SPLIT)
    train_scaled = scaled[:split]
    test_scaled = scaled[split:]

    X_tr, y_tr = build_sequences(train_scaled)
    X_te, y_te = build_sequences(test_scaled)

    if len(X_tr) < 10:
        self.lstm_metrics = {"error": "Insufficient LSTM samples"}
        self.forecast = None
        return self.lstm_metrics

    val_split = int(len(X_tr) * 0.8)
    X_val, y_val = X_tr[val_split:], y_tr[val_split:]
    X_tr, y_tr = X_tr[:val_split], y_tr[:val_split]

    self.lstm_model, _ = train_lstm(
        X_tr, y_tr,
        X_val if len(X_val) > 0 else None,
        y_val if len(X_val) > 0 else None,
        units=units,
        epochs=epochs,
    )

    if len(X_te) > 0:
        self.lstm_metrics = evaluate_lstm(
            self.lstm_model, X_te, y_te, self.price_scaler
        )
    else:
        self.lstm_metrics = {}

    last_seq = scaled[-SEQUENCE_LENGTH:]
    self.forecast = predict_next_n_days(
        self.lstm_model,
        last_seq,
        self.price_scaler,
        n_days=PREDICTION_DAYS,
    )

    return self.lstm_metrics