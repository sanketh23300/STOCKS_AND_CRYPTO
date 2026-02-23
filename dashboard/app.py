# ============================================================
#   dashboard/app.py  –  Streamlit AI Market Trend Analyzer
#   Run:  streamlit run dashboard/app.py
# ============================================================

import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy  as np
import pandas as pd
import streamlit as st

from config import (
    STOCK_SYMBOLS, CRYPTO_SYMBOLS, PAGE_TITLE, PAGE_ICON, LAYOUT,
    PREDICTION_DAYS,
)
from analysis.trend_analyzer import TrendAnalyzer
from visualization.charts   import (
    candlestick_chart, indicator_chart, forecast_chart,
    model_comparison_chart, feature_importance_chart,
    sentiment_gauge, fear_greed_gauge,
    return_distribution, monthly_returns_heatmap,
)

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)


# ─────────────────────────────────────────────────────────
#  CSS Styling
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
body, .stApp { background-color: #0E1117; color: #E2E8F0; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1A2035, #1F2937);
    border: 1px solid #2D3748;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
[data-testid="stMetricValue"]  { color: #00D4B4; font-size: 1.5rem !important; font-weight: 700; }
[data-testid="stMetricLabel"]  { color: #94A3B8; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricDelta"]  { font-size: 0.85rem !important; }

/* Signal badge */
.signal-buy  { background: linear-gradient(90deg,#00D4B4,#009B8A); color:#000; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1.1rem; }
.signal-sell { background: linear-gradient(90deg,#FF4B4B,#CC0000); color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1.1rem; }
.signal-hold { background: linear-gradient(90deg,#FFD700,#FFA500); color:#000; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1.1rem; }

/* Section headers */
.section-header {
    background: linear-gradient(90deg,#1A2035,#111827);
    border-left: 4px solid #00D4B4;
    padding: 10px 18px;
    border-radius: 6px;
    margin: 18px 0 10px;
    font-size: 1.05rem;
    font-weight: 600;
    color: #E2E8F0;
}

/* Sidebar */
[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1F2937; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  Helper: signal badge HTML
# ─────────────────────────────────────────────────────────
def signal_badge(signal: str) -> str:
    cls = {"BUY": "buy", "SELL": "sell"}.get(signal.upper(), "hold")
    icon = {"BUY": "▲", "SELL": "▼", "HOLD": "◆"}.get(signal.upper(), "◆")
    return f'<span class="signal-{cls}">{icon} {signal}</span>'


# ─────────────────────────────────────────────────────────
#  Sidebar – controls
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 AI Market Analyzer")
    st.divider()

    asset_type = st.selectbox("Asset Type", ["Stock", "Crypto"], index=0)
    symbols    = STOCK_SYMBOLS if asset_type == "Stock" else CRYPTO_SYMBOLS

    custom_sym = st.text_input("Custom Symbol (override list)", placeholder="e.g. AMD, AVAX").strip().upper()
    symbol     = custom_sym if custom_sym else st.selectbox("Select Asset", symbols)

    st.divider()
    period = st.selectbox("History", ["1y","2y","3y","5y"], index=1)
    train_lstm  = st.toggle("LSTM Forecast", value=True)
    lstm_epochs = st.slider("LSTM Epochs", 20, 150, 60, 10) if train_lstm else 60

    st.divider()
    news_api_key = st.text_input("NewsAPI Key (optional)", type="password",
                                  help="Leave blank to use demo headlines")
    run_btn = st.button("🚀  Run Analysis", width='stretch', type="primary")

    st.markdown("""
    <div style='margin-top:30px;color:#4B5563;font-size:0.72rem;text-align:center'>
    AI-Driven Market Analysis<br>Not financial advice
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  Main content
# ─────────────────────────────────────────────────────────
st.markdown(f"<h1 style='color:#00D4B4;'>📊 {PAGE_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94A3B8;'>AI-powered technical analysis, ML trend prediction & sentiment for stocks and crypto.</p>", unsafe_allow_html=True)

if not run_btn:
    # ── Landing screen ────────────────────────────────────
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        <div style='background:#1A2035;border:1px solid #2D3748;border-radius:12px;padding:20px;'>
        <h3 style='color:#00D4B4;'>🤖 AI Predictions</h3>
        <p style='color:#94A3B8;font-size:0.9rem;'>Random Forest, Gradient Boosting, SVM & Logistic Regression classifiers trained on 50+ technical features.</p>
        </div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown("""
        <div style='background:#1A2035;border:1px solid #2D3748;border-radius:12px;padding:20px;'>
        <h3 style='color:#4BA3FF;'>📉 LSTM Forecasting</h3>
        <p style='color:#94A3B8;font-size:0.9rem;'>Bidirectional LSTM neural network forecasts price movement for the next 30 trading days.</p>
        </div>""", unsafe_allow_html=True)
    with cols[2]:
        st.markdown("""
        <div style='background:#1A2035;border:1px solid #2D3748;border-radius:12px;padding:20px;'>
        <h3 style='color:#FFD700;'>💬 Sentiment Analysis</h3>
        <p style='color:#94A3B8;font-size:0.9rem;'>VADER NLP + Fear & Greed Index blend into an actionable market sentiment score.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈  Select an asset in the sidebar and click **Run Analysis** to begin.")
    st.stop()


# ─────────────────────────────────────────────────────────
#  Run analysis with caching (keyed by symbol + settings)
# ─────────────────────────────────────────────────────────
cache_key = f"{symbol}_{asset_type}_{period}_{train_lstm}_{lstm_epochs}"

@st.cache_resource(show_spinner=False)
def run_analysis(sym, atype, per, do_lstm, epochs, nkey):
    import importlib, sys
    # Ensure fresh imports on each unique key
    analyzer = TrendAnalyzer(sym, atype.lower())
    fetch_kw = {}
    if atype == "Stock":
        fetch_kw = {"period": per}
    elif atype == "Crypto":
        fetch_kw = {"period": per}
    result = analyzer.run(
        train_lstm        = do_lstm,
        compute_sentiment = True,
        lstm_epochs       = epochs,
        news_api_key      = nkey or None,
        **fetch_kw,
    )
    result["_analyzer"] = analyzer
    return result


with st.spinner(f"Analysing {symbol} … this may take a minute ⏳"):
    try:
        res = run_analysis(symbol, asset_type, period or "1y",
                           train_lstm, lstm_epochs, news_api_key or "")
    except Exception as exc:
        st.error(f"❌ Analysis failed: {exc}")
        st.stop()

raw_df    = res["raw_df"]
feat_df   = res["feat_df"]
info      = res["info"]
eval_df   = res["eval_df"]
ml_signal = res["ml_signal"]
tech_sig  = res["tech_signal"]
sentiment = res["sentiment"]
forecast  = res["forecast"]
lstm_met  = res["lstm_metrics"]
sig_sum   = res["signal_summary"]
analyzer  = res["_analyzer"]


# ─────────────────────────────────────────────────────────
#  ── HEADER: asset info + key metrics
# ─────────────────────────────────────────────────────────
name   = info.get("name", symbol)
latest = raw_df["Close"].iloc[-1]
prev   = raw_df["Close"].iloc[-2]
chg    = ((latest - prev) / prev) * 100
vol7   = raw_df["Close"].pct_change().tail(7).std() * 100

st.markdown(f"## {name} &nbsp; `{symbol}`")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("💲 Price",       f"${latest:,.2f}",   f"{chg:+.2f}%")
c2.metric("📊 7d Volatility", f"{vol7:.2f}%")
c3.metric("📅 Data Points",  f"{len(raw_df):,}")
c4.metric("🏆 Best Model",   res["best_model"],   f"{eval_df['Accuracy'].max()*100:.1f}% acc")
c5.metric("🔮 ML Signal",    ml_signal)
c6.metric("⚙️ Tech Signal",  tech_sig)

# Overall signal banner
overall_signal = ml_signal  # primary signal
sent_score = sentiment.get("overall_score", 0) if sentiment else 0
st.markdown(
    f"<div style='margin:18px 0;'>"
    f"<b style='color:#94A3B8;'>Combined AI Signal:</b>&nbsp;&nbsp;"
    f"{signal_badge(overall_signal)}"
    f"&nbsp;&nbsp;&nbsp;<span style='color:#4B5563;font-size:0.85rem;'>Sentiment: "
    f"{'🟢' if sent_score>0.05 else '🔴' if sent_score<-0.05 else '🟡'} "
    f"{sentiment.get('overall_label','N/A')}</span></div>",
    unsafe_allow_html=True,
)

st.divider()


# ─────────────────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Price & Volume",
    "📉 Indicators",
    "🤖 ML Models",
    "🔮 LSTM Forecast",
    "💬 Sentiment",
    "📊 Statistics",
    "⚙️ Signals",
])


# ── Tab 1: Candlestick ────────────────────────────────────
with tabs[0]:
    st.markdown("<div class='section-header'>Candlestick Chart with EMAs &amp; Bollinger Bands</div>",
                unsafe_allow_html=True)
    show_ema = st.toggle("Show EMA overlays", value=True, key="ema_tog")
    st.plotly_chart(candlestick_chart(feat_df, symbol, show_ema), width='stretch', key="candle")

    # Quick OHLCV table
    st.markdown("<div class='section-header'>Recent OHLCV Data</div>", unsafe_allow_html=True)
    disp = raw_df.tail(15)[["Open","High","Low","Close","Volume"]].copy()
    disp["Change%"] = disp["Close"].pct_change().mul(100).round(2)
    st.dataframe(
        disp.sort_index(ascending=False).style
            .background_gradient(subset=["Change%"], cmap="RdYlGn")
            .format({"Open":"${:.2f}","High":"${:.2f}","Low":"${:.2f}","Close":"${:.2f}",
                     "Volume":"{:,.0f}","Change%":"{:.2f}%"}),
        width='stretch',
    )


# ── Tab 2: Technical Indicators ──────────────────────────
with tabs[1]:
    st.markdown("<div class='section-header'>RSI · MACD · OBV</div>", unsafe_allow_html=True)
    st.plotly_chart(indicator_chart(feat_df, symbol), width='stretch', key="ind")

    st.markdown("<div class='section-header'>Latest Indicator Values</div>", unsafe_allow_html=True)
    last = feat_df.iloc[-1]
    cols_ind = st.columns(4)
    ind_items = [
        ("RSI",           f"{last.get('RSI',0):.1f}"),
        ("MACD",          f"{last.get('MACD',0):.4f}"),
        ("MACD Signal",   f"{last.get('MACD_Signal',0):.4f}"),
        ("ATR",           f"{last.get('ATR',0):.4f}"),
        ("ADX",           f"{last.get('ADX',0):.1f}"),
        ("OBV",           f"{last.get('OBV',0):,.0f}"),
        ("CMF",           f"{last.get('CMF',0):.4f}"),
        ("BB %",          f"{last.get('BB_Pct',0.5):.3f}"),
    ]
    for i, (label, val) in enumerate(ind_items):
        cols_ind[i % 4].metric(label, val)


# ── Tab 3: ML Models ─────────────────────────────────────
with tabs[2]:
    st.markdown("<div class='section-header'>Model Performance Comparison</div>",
                unsafe_allow_html=True)
    st.plotly_chart(model_comparison_chart(eval_df), width='stretch')

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("<div class='section-header'>Metrics Table</div>", unsafe_allow_html=True)
        disp_eval = eval_df[["Model","Accuracy","AUC-ROC"]].copy()
        disp_eval["Accuracy"] = disp_eval["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(disp_eval, width='stretch')

    with c2:
        st.markdown(f"<div class='section-header'>Feature Importance ({res['best_model']})</div>",
                    unsafe_allow_html=True)
        fi = analyzer.feature_importance(top_n=20)
        if not fi.empty:
            st.plotly_chart(feature_importance_chart(fi, res["best_model"]),
                            width='stretch')
        else:
            st.info("Feature importance not available for this model.")

    # Latest signal confidence
    best_pipe = analyzer.best_model
    if hasattr(best_pipe, "predict_proba"):
        from data.preprocessor import build_feature_matrix
        X, _ = build_feature_matrix(feat_df)
        probs = best_pipe.predict_proba(X.iloc[[-1]])[0]
        labels = ["SELL", "HOLD", "BUY"]
        st.markdown("<div class='section-header'>Latest Prediction Probabilities</div>",
                    unsafe_allow_html=True)
        pcols = st.columns(3)
        colors_map = {"SELL": "inverse", "HOLD": "off", "BUY": "normal"}
        for col, (lb, pr) in zip(pcols, zip(labels, probs)):
            col.metric(lb, f"{pr*100:.1f}%", delta_color=colors_map.get(lb, "off"))


# ── Tab 4: LSTM Forecast ─────────────────────────────────
with tabs[3]:
    if forecast is not None and len(forecast) > 0:
        st.markdown(f"<div class='section-header'>{PREDICTION_DAYS}-Day LSTM Price Forecast</div>",
                    unsafe_allow_html=True)

        actual_preds = lstm_met.get("Predictions", None) if lstm_met else None
        st.plotly_chart(
            forecast_chart(raw_df, forecast, symbol, actual_preds),
            width='stretch',
        )

        # Metrics
        if lstm_met and "MAE" in lstm_met:
            mc = st.columns(4)
            mc[0].metric("MAE",          f"${lstm_met['MAE']:,.2f}")
            mc[1].metric("RMSE",         f"${lstm_met['RMSE']:,.2f}")
            mc[2].metric("MAPE",         f"{lstm_met['MAPE(%)']:.2f}%")
            mc[3].metric("Direction Acc",f"{lstm_met['Direction Acc']*100:.1f}%")

        st.markdown("<div class='section-header'>Forecast Values</div>", unsafe_allow_html=True)
        from datetime import timedelta
        last_date    = raw_df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                     periods=len(forecast), freq="B")
        fcast_df = pd.DataFrame({
            "Date":     future_dates,
            "Forecast": forecast.round(2),
        })
        fcast_df["Change%"] = fcast_df["Forecast"].pct_change().mul(100).round(2)
        fcast_df.set_index("Date", inplace=True)
        st.dataframe(
            fcast_df.style.background_gradient(subset=["Change%"], cmap="RdYlGn")
                          .format({"Forecast":"${:.2f}","Change%":"{:.2f}%"}),
            width='stretch',
        )
    else:
        if lstm_met and "error" in lstm_met:
            st.warning(f"LSTM unavailable: {lstm_met['error']}")
        else:
            st.info("Enable **LSTM Forecast** in the sidebar to see predictions.")


# ── Tab 5: Sentiment ─────────────────────────────────────
with tabs[4]:
    if sentiment:
        news_s = sentiment.get("news_sentiment", {})
        fg     = sentiment.get("fear_greed", None)
        score  = sentiment.get("overall_score", 0)
        label  = sentiment.get("overall_label",  "NEUTRAL")

        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(sentiment_gauge(score, label), width='stretch')
        with g2:
            if fg:
                st.plotly_chart(fear_greed_gauge(fg["value"], fg["label"]), width='stretch')
            else:
                st.info("Fear & Greed Index is only available for crypto assets.")

        # Breakdown
        st.markdown("<div class='section-header'>Sentiment Breakdown</div>",
                    unsafe_allow_html=True)
        sc = st.columns(4)
        sc[0].metric("Positive",  f"{news_s.get('pos',0)*100:.1f}%")
        sc[1].metric("Neutral",   f"{news_s.get('neu',0)*100:.1f}%")
        sc[2].metric("Negative",  f"{news_s.get('neg',0)*100:.1f}%")
        sc[3].metric("Compound",  f"{news_s.get('compound',0):.4f}")

        # Headlines
        headlines = news_s.get("headlines", [])
        if headlines:
            st.markdown("<div class='section-header'>Recent Headlines Analysed</div>",
                        unsafe_allow_html=True)
            from models.sentiment import analyze_text
            for h in headlines:
                s = analyze_text(h)
                c = s["compound"]
                icon = "🟢" if c > 0.05 else "🔴" if c < -0.05 else "🟡"
                st.markdown(
                    f"{icon} `{c:+.3f}` &nbsp; {h[:120]}…" if len(h) > 120 else f"{icon} `{c:+.3f}` &nbsp; {h}",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Sentiment analysis unavailable.")


# ── Tab 6: Statistics ────────────────────────────────────
with tabs[5]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Daily Return Distribution</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(return_distribution(raw_df, symbol), width='stretch')
    with c2:
        st.markdown("<div class='section-header'>Monthly Returns Heatmap</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(monthly_returns_heatmap(raw_df, symbol), width='stretch')

    # Summary stats
    st.markdown("<div class='section-header'>Descriptive Statistics</div>",
                unsafe_allow_html=True)
    stats = raw_df["Close"].describe().round(4)
    returns_s = raw_df["Close"].pct_change().dropna()
    extra = pd.Series({
        "Skewness":  round(float(returns_s.skew()), 4),
        "Kurtosis":  round(float(returns_s.kurtosis()), 4),
        "Sharpe(~)": round(float(returns_s.mean() / returns_s.std() * (252**0.5)), 4),
        "Max DD %":  round(float(
            ((raw_df["Close"] / raw_df["Close"].cummax()) - 1).min() * 100
        ), 4),
    })
    full_stats = pd.concat([stats, extra])
    st.dataframe(full_stats.to_frame("Value"), width='stretch')


# ── Tab 7: Signals ───────────────────────────────────────
with tabs[6]:
    st.markdown("<div class='section-header'>Technical Signal Summary</div>",
                unsafe_allow_html=True)

    rows = []
    for indicator, (sig, reason) in sig_sum.items():
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(sig, "⚪")
        rows.append({"Indicator": indicator, "Signal": sig, "Reason": reason, "": emoji})

    sig_df = pd.DataFrame(rows)

    overall_row = sig_df[sig_df["Indicator"] == "OVERALL"]
    detail_rows = sig_df[sig_df["Indicator"] != "OVERALL"]

    if not overall_row.empty:
        ov = overall_row.iloc[0]
        st.markdown(
            f"<h3>Overall Signal: {signal_badge(ov['Signal'])} "
            f"<small style='color:#94A3B8;font-size:0.9rem;'>{ov['Reason']}</small></h3>",
            unsafe_allow_html=True,
        )

    st.dataframe(detail_rows.set_index("Indicator"), width='stretch')

    # Buy/Sell/Hold distribution pie
    import plotly.express as px
    counts = detail_rows["Signal"].value_counts().reset_index()
    counts.columns = ["Signal", "Count"]
    pie = px.pie(
        counts, names="Signal", values="Count",
        color="Signal",
        color_discrete_map={"BUY": "#00D4B4", "SELL": "#FF4B4B", "HOLD": "#FFD700"},
        title="Signal Distribution",
    )
    pie.update_layout(paper_bgcolor="#0E1117", font=dict(color="#E2E8F0"),
                      margin=dict(t=50, b=20))
    st.plotly_chart(pie, width='stretch')

