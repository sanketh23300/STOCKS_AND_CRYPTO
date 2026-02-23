# ============================================================
#   analysis/trend_analyzer.py  –  End-to-end analysis pipeline
# ============================================================

import os, warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config              import SEQUENCE_LENGTH, PREDICTION_DAYS, TRAIN_SPLIT
from data.fetcher        import fetch_data, fetch_info
from data.preprocessor  import (
    create_labels, build_sequences,
    build_feature_matrix, time_split,
    normalize_features, fit_price_scaler,
    inverse_transform_price, transform_price,
    add_returns,
)
from features.indicators import add_all_indicators, get_signal_summary
from models.ml_model     import (
    train_models, evaluate_models, get_best_model,
    predict_signal, label_to_signal, get_feature_importance,
)
from models.sentiment    import get_market_sentiment


# ─────────────────────────────────────────────────────────
#  Main analysis pipeline
# ─────────────────────────────────────────────────────────
class TrendAnalyzer:
    def __init__(self, symbol: str, asset_type: str = "stock"):
        self.symbol     = symbol.upper()
        self.asset_type = asset_type.lower()
        self.info       = {}

        # Pipeline artefacts (filled by run())
        self.raw_df     : pd.DataFrame | None = None
        self.feat_df    : pd.DataFrame | None = None
        self.trained    : dict = {}
        self.best_model_name: str = ""
        self.best_model       = None
        self.eval_df    : pd.DataFrame | None = None
        self.feature_names: list[str] = []
        self.price_scaler = None
        self.lstm_model   = None
        self.lstm_metrics : dict = {}
        self.forecast     : np.ndarray | None = None
        self.signal_summary: dict = {}
        self.sentiment    : dict = {}

    # ── 1. Data ───────────────────────────────────────────
    def load_data(self, **kwargs) -> pd.DataFrame:
        self.raw_df = fetch_data(self.symbol, self.asset_type, **kwargs)
        self.info   = fetch_info(self.symbol, self.asset_type)
        return self.raw_df

    # ── 2. Feature engineering ────────────────────────────
    def engineer_features(self) -> pd.DataFrame:
        df = add_returns(self.raw_df)
        df = add_all_indicators(df)
        df["Label"] = create_labels(df, lookahead=1)
        df.dropna(inplace=True)
        self.feat_df = df
        return df

    # ── 3. Train ML classifiers ───────────────────────────
    def train_classifiers(self, model_names: list[str] | None = None) -> pd.DataFrame:
        X, y = build_feature_matrix(self.feat_df)
        self.feature_names = list(X.columns)

        X_tr, X_te, y_tr, y_te = time_split(X, y)
        X_tr_s, X_te_s, _      = normalize_features(X_tr, X_te)

        self.trained  = train_models(X_tr_s, y_tr, model_names)
        self.eval_df  = evaluate_models(self.trained, X_te_s, y_te)
        self.best_model_name, self.best_model = get_best_model(self.eval_df, self.trained)
        return self.eval_df

    # ── 4. LSTM price forecast ────────────────────────────
    def train_lstm_forecast(self, epochs: int = 60, units: int = 64) -> dict:
        try:
            from models.lstm_model import train_lstm, predict_next_n_days, evaluate_lstm
        except ImportError:
            self.lstm_metrics = {"error": "TensorFlow not available"}
            return self.lstm_metrics

        close        = self.feat_df["Close"].values
        scaled, self.price_scaler = fit_price_scaler(close)

        split        = int(len(scaled) * TRAIN_SPLIT)
        train_scaled = scaled[:split]
        test_scaled  = scaled[split:]

        X_tr, y_tr = build_sequences(train_scaled)
        X_te, y_te = build_sequences(test_scaled)

        # Validation split (last 20% of training)
        val_split = int(len(X_tr) * 0.8)
        X_val, y_val = X_tr[val_split:], y_tr[val_split:]
        X_tr,  y_tr  = X_tr[:val_split],  y_tr[:val_split]

        self.lstm_model, _ = train_lstm(
            X_tr, y_tr, X_val, y_val,
            units=units, epochs=epochs,
        )

        if len(X_te) > 0:
            self.lstm_metrics = evaluate_lstm(self.lstm_model, X_te, y_te, self.price_scaler)
        else:
            self.lstm_metrics = {}

        # Multi-day forecast
        last_seq     = scaled[-SEQUENCE_LENGTH:]
        self.forecast = predict_next_n_days(
            self.lstm_model, last_seq, self.price_scaler, n_days=PREDICTION_DAYS
        )
        return self.lstm_metrics

    # ── 5. Technical signal summary ───────────────────────
    def compute_signals(self) -> dict:
        self.signal_summary = get_signal_summary(self.feat_df)
        return self.signal_summary

    # ── 6. Sentiment ──────────────────────────────────────
    def compute_sentiment(self, news_api_key: str | None = None) -> dict:
        self.sentiment = get_market_sentiment(
            self.symbol, self.asset_type, news_api_key
        )
        return self.sentiment

    # ── 7. Latest prediction ──────────────────────────────
    def latest_ml_signal(self) -> str:
        if self.best_model is None:
            return "NO MODEL"
        X, _ = build_feature_matrix(self.feat_df)
        X_last = X.iloc[[-1]]
        labels, _ = predict_signal(self.best_model, X_last)
        return label_to_signal(labels[0])

    # ── 8. Feature importance ─────────────────────────────
    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        return get_feature_importance(self.best_model, self.feature_names, top_n)

    # ── Full pipeline ─────────────────────────────────────
    def run(
        self,
        train_lstm:      bool = True,
        compute_sentiment: bool = True,
        lstm_epochs:     int  = 60,
        news_api_key:    str | None = None,
        **fetch_kwargs,
    ) -> dict:
        """Run the complete analysis and return a results dict."""
        print(f"\n{'='*55}")
        print(f"  Analyzing: {self.symbol}  ({self.asset_type})")
        print(f"{'='*55}")

        print("\n[1/6] Loading market data …")
        self.load_data(**fetch_kwargs)

        print("[2/6] Engineering features & indicators …")
        self.engineer_features()

        print("[3/6] Training ML classifiers …")
        self.train_classifiers()

        print("[4/6] Computing technical signals …")
        self.compute_signals()

        if train_lstm:
            print(f"[5/6] Training LSTM forecast ({lstm_epochs} epochs) …")
            self.train_lstm_forecast(epochs=lstm_epochs)
        else:
            print("[5/6] Skipping LSTM (disabled).")

        if compute_sentiment:
            print("[6/6] Analysing sentiment …")
            self.compute_sentiment(news_api_key)
        else:
            print("[6/6] Skipping sentiment (disabled).")

        ml_signal  = self.latest_ml_signal()
        tech_signal = self.signal_summary.get("OVERALL", ("HOLD", ""))[0]
        sent_label  = self.sentiment.get("overall_label", "NEUTRAL") if self.sentiment else "NEUTRAL"

        print(f"\n{'─'*55}")
        print(f"  ML Signal    : {ml_signal}")
        print(f"  Tech Signal  : {tech_signal}")
        print(f"  Sentiment    : {sent_label}")
        if self.forecast is not None:
            print(f"  30d Forecast : ${self.forecast[-1]:.2f} (last predicted)")
        print(f"{'─'*55}\n")

        return {
            "symbol":         self.symbol,
            "asset_type":     self.asset_type,
            "info":           self.info,
            "raw_df":         self.raw_df,
            "feat_df":        self.feat_df,
            "eval_df":        self.eval_df,
            "best_model":     self.best_model_name,
            "ml_signal":      ml_signal,
            "tech_signal":    tech_signal,
            "signal_summary": self.signal_summary,
            "sentiment":      self.sentiment,
            "lstm_metrics":   self.lstm_metrics,
            "forecast":       self.forecast,
        }
