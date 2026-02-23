# ============================================================
# analysis/trend_analyzer.py — FINAL STABLE VERSION
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import SEQUENCE_LENGTH, PREDICTION_DAYS, TRAIN_SPLIT
from data.fetcher import fetch_data, fetch_info
from data.preprocessor import (
    create_labels,
    build_sequences,
    build_feature_matrix,
    time_split,
    normalize_features,
    fit_price_scaler,
    add_returns,
)
from features.indicators import add_all_indicators, get_signal_summary
from models.ml_model import (
    train_models,
    evaluate_models,
    get_best_model,
    predict_signal,
    label_to_signal,
    get_feature_importance,
)
from models.sentiment import get_market_sentiment


# ============================================================
# 🚨 REQUIRED CLASS — DO NOT RENAME
# ============================================================
class TrendAnalyzer:
    def __init__(self, symbol: str, asset_type: str = "stock"):
        self.symbol = symbol.upper()
        self.asset_type = asset_type.lower()

        self.info = {}
        self.raw_df = None
        self.feat_df = None

        self.trained = {}
        self.eval_df = None
        self.best_model = None
        self.best_model_name = ""

        self.signal_summary = {}
        self.sentiment = {}

        self.lstm_model = None
        self.price_scaler = None
        self.forecast = None
        self.lstm_metrics = {}

        self.feature_names = []

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------
    def load_data(self, **kwargs):
        self.raw_df = fetch_data(self.symbol, self.asset_type, **kwargs)
        self.info = fetch_info(self.symbol, self.asset_type)
        return self.raw_df

    # --------------------------------------------------------
    # 2. Feature engineering
    # --------------------------------------------------------
    def engineer_features(self):
        df = add_returns(self.raw_df)
        df = add_all_indicators(df)
        df["Label"] = create_labels(df, lookahead=1)
        df.dropna(inplace=True)
        self.feat_df = df
        return df

    # --------------------------------------------------------
    # 3. Train ML classifiers
    # --------------------------------------------------------
    def train_classifiers(self):
        X, y = build_feature_matrix(self.feat_df)
        self.feature_names = list(X.columns)

        X_tr, X_te, y_tr, y_te = time_split(X, y)
        X_tr_s, X_te_s, _ = normalize_features(X_tr, X_te)

        self.trained = train_models(X_tr_s, y_tr)
        self.eval_df = evaluate_models(self.trained, X_te_s, y_te)
        self.best_model_name, self.best_model = get_best_model(
            self.eval_df, self.trained
        )

        return self.eval_df

    # --------------------------------------------------------
    # 4. LSTM Forecast (SAFE)
    # --------------------------------------------------------
    def train_lstm_forecast(self, epochs: int = 60):
        try:
            from models.lstm_model import train_lstm, predict_next_n_days
        except Exception as e:
            self.lstm_metrics = {"error": str(e)}
            return self.lstm_metrics

        close_prices = self.feat_df["Close"].values

        # 🚨 HARD SAFETY CHECK
        if len(close_prices) < SEQUENCE_LENGTH * 2:
            self.lstm_metrics = {"error": "Not enough data for LSTM"}
            self.forecast = None
            return self.lstm_metrics

        scaled, self.price_scaler = fit_price_scaler(close_prices)

        split = int(len(scaled) * TRAIN_SPLIT)
        train_scaled = scaled[:split]
        test_scaled = scaled[split:]

        X_train, y_train = build_sequences(train_scaled)
        X_test, y_test = build_sequences(test_scaled)

        if len(X_train) == 0 or len(X_test) == 0:
            self.lstm_metrics = {"error": "LSTM sequence build failed"}
            self.forecast = None
            return self.lstm_metrics

        val_split = int(len(X_train) * 0.8)
        X_val, y_val = X_train[val_split:], y_train[val_split:]
        X_train, y_train = X_train[:val_split], y_train[:val_split]

        try:
            self.lstm_model, _ = train_lstm(
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=epochs,
            )
        except Exception as e:
            self.lstm_metrics = {"error": str(e)}
            self.forecast = None
            return self.lstm_metrics

        last_seq = scaled[-SEQUENCE_LENGTH:]
        self.forecast = predict_next_n_days(
            self.lstm_model,
            last_seq,
            self.price_scaler,
            n_days=PREDICTION_DAYS,
        )

        self.lstm_metrics = {"status": "success"}
        return self.lstm_metrics

    # --------------------------------------------------------
    # 5. Technical signals
    # --------------------------------------------------------
    def compute_signals(self):
        self.signal_summary = get_signal_summary(self.feat_df)
        return self.signal_summary

    # --------------------------------------------------------
    # 6. Sentiment
    # --------------------------------------------------------
    def compute_sentiment(self, news_api_key=None):
        self.sentiment = get_market_sentiment(
            self.symbol, self.asset_type, news_api_key
        )
        return self.sentiment

    # --------------------------------------------------------
    # 7. Latest ML signal
    # --------------------------------------------------------
    def latest_ml_signal(self):
        if self.best_model is None:
            return "NO MODEL"

        X, _ = build_feature_matrix(self.feat_df)
        X_last = X.iloc[[-1]]
        labels, _ = predict_signal(self.best_model, X_last)
        return label_to_signal(labels[0])

    # --------------------------------------------------------
    # 8. Feature importance
    # --------------------------------------------------------
    def feature_importance(self, top_n: int = 20):
        return get_feature_importance(
            self.best_model, self.feature_names, top_n
        )

    # --------------------------------------------------------
    # FULL PIPELINE
    # --------------------------------------------------------
    def run(
        self,
        train_lstm: bool = True,
        compute_sentiment: bool = True,
        lstm_epochs: int = 60,
        news_api_key: str | None = None,
        **fetch_kwargs,
    ):
        self.load_data(**fetch_kwargs)
        self.engineer_features()
        self.train_classifiers()
        self.compute_signals()

        if train_lstm:
            self.train_lstm_forecast(epochs=lstm_epochs)

        if compute_sentiment:
            self.compute_sentiment(news_api_key)

        return {
            "symbol": self.symbol,
            "asset_type": self.asset_type,
            "info": self.info,
            "raw_df": self.raw_df,
            "feat_df": self.feat_df,
            "eval_df": self.eval_df,
            "best_model": self.best_model_name,
            "ml_signal": self.latest_ml_signal(),
            "tech_signal": self.signal_summary.get("OVERALL", ("HOLD", ""))[0],
            "signal_summary": self.signal_summary,
            "sentiment": self.sentiment,
            "forecast": self.forecast,
            "lstm_metrics": self.lstm_metrics,
        }