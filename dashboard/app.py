# ============================================================
# dashboard/app.py — FULL UI (STABLE BACKEND COMPATIBLE)
# ============================================================

import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    STOCK_SYMBOLS, CRYPTO_SYMBOLS,
    PAGE_TITLE, PAGE_ICON, LAYOUT,
    PREDICTION_DAYS,
)

from analysis.trend_analyzer import TrendAnalyzer
from visualization.charts import (
    candlestick_chart,
    indicator_chart,
    forecast_chart,
    model_comparison_chart,
    feature_importance_chart,
    sentiment_gauge,
    fear_greed_gauge,
    return_distribution,
    monthly_returns_heatmap,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 AI Market Analyzer")

    asset_type = st.selectbox("Asset Type", ["Stock", "Crypto"])
    symbols = STOCK_SYMBOLS if asset_type == "Stock" else CRYPTO_SYMBOLS

    symbol = st.selectbox("Select Symbol", symbols)

    period = st.selectbox("History", ["1y", "2y", "3y", "5y"])
    use_lstm = st.toggle("Enable LSTM Forecast", value=True)
    epochs = st.slider("LSTM Epochs", 20, 120, 60, 10)

    news_key = st.text_input("NewsAPI Key (optional)", type="password")

    run_btn = st.button("🚀 Run Analysis", type="primary")

# ─────────────────────────────────────────────
# Landing
# ─────────────────────────────────────────────
st.title(PAGE_TITLE)
st.caption("AI-powered technical analysis, ML signals, forecasting & sentiment")

if not run_btn:
    st.info("👈 Choose an asset and click **Run Analysis**")
    st.stop()

# ─────────────────────────────────────────────
# Run analysis
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def run_analysis(sym, atype, per, do_lstm, ep, key):
    analyzer = TrendAnalyzer(sym, atype.lower())
    kwargs = {"period": per} if atype == "Stock" else {}
    res = analyzer.run(
        train_lstm=do_lstm,
        lstm_epochs=ep,
        news_api_key=key or None,
        **kwargs,
    )
    res["_analyzer"] = analyzer
    return res

with st.spinner("Running analysis…"):
    result = run_analysis(
        symbol,
        asset_type,
        period,
        use_lstm,
        epochs,
        news_key,
    )

raw_df = result["raw_df"]
feat_df = result["feat_df"]
eval_df = result["eval_df"]
forecast = result["forecast"]
sentiment = result["sentiment"]
signals = result["signal_summary"]
analyzer = result["_analyzer"]

# ─────────────────────────────────────────────
# Header metrics
# ─────────────────────────────────────────────
latest = raw_df["Close"].iloc[-1]
prev = raw_df["Close"].iloc[-2]
chg = ((latest - prev) / prev) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Price", f"${latest:,.2f}", f"{chg:+.2f}%")
c2.metric("ML Signal", result["ml_signal"])
c3.metric("Tech Signal", result["tech_signal"])
c4.metric("Best Model", result["best_model"])

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📈 Price",
    "📉 Indicators",
    "🤖 Models",
    "🔮 Forecast",
    "💬 Sentiment",
    "📊 Stats",
    "⚙️ Signals",
])

# ── Price
with tabs[0]:
    st.plotly_chart(
        candlestick_chart(feat_df, symbol),
        use_container_width=True,
    )

# ── Indicators
with tabs[1]:
    st.plotly_chart(
        indicator_chart(feat_df, symbol),
        use_container_width=True,
    )

# ── Models
with tabs[2]:
    st.plotly_chart(
        model_comparison_chart(eval_df),
        use_container_width=True,
    )

    fi = analyzer.feature_importance()
    if not fi.empty:
        st.plotly_chart(
            feature_importance_chart(fi, result["best_model"]),
            use_container_width=True,
        )

# ── Forecast
with tabs[3]:
    if forecast is not None:
        st.plotly_chart(
            forecast_chart(raw_df, forecast, symbol),
            use_container_width=True,
        )
    else:
        st.info("LSTM forecast not available")

# ── Sentiment
with tabs[4]:
    if sentiment:
        st.plotly_chart(
            sentiment_gauge(
                sentiment.get("overall_score", 0),
                sentiment.get("overall_label", "NEUTRAL"),
            ),
            use_container_width=True,
        )
    else:
        st.info("Sentiment unavailable")

# ── Stats
with tabs[5]:
    st.plotly_chart(
        return_distribution(raw_df, symbol),
        use_container_width=True,
    )
    st.plotly_chart(
        monthly_returns_heatmap(raw_df, symbol),
        use_container_width=True,
    )

# ── Signals
with tabs[6]:
    df = pd.DataFrame([
        {"Indicator": k, "Signal": v[0], "Reason": v[1]}
        for k, v in signals.items()
    ])
    st.dataframe(df, use_container_width=True)