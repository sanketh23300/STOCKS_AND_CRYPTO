# ============================================================
#   visualization/charts.py  –  Plotly chart generators
# ============================================================

import numpy  as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express       as px
from plotly.subplots import make_subplots
from datetime import timedelta


COLORS = {
    "green":   "#00D4B4",
    "red":     "#FF4B4B",
    "blue":    "#4BA3FF",
    "yellow":  "#FFD700",
    "purple":  "#B44BFF",
    "orange":  "#FF8C00",
    "bg":      "#0E1117",
    "grid":    "#1F2937",
    "text":    "#E2E8F0",
}

_LAYOUT = dict(
    paper_bgcolor = COLORS["bg"],
    plot_bgcolor  = COLORS["bg"],
    font          = dict(color=COLORS["text"], size=12),
    xaxis         = dict(gridcolor=COLORS["grid"], showgrid=True),
    yaxis         = dict(gridcolor=COLORS["grid"], showgrid=True),
    legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin        = dict(l=40, r=20, t=50, b=40),
)


def _apply_layout(fig: go.Figure, title: str = "") -> go.Figure:
    layout = dict(_LAYOUT)
    layout["title"] = dict(text=title, font=dict(size=16))
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────
#  Candlestick + volume + overlays
# ─────────────────────────────────────────────────────────
def candlestick_chart(df: pd.DataFrame, symbol: str, show_ema: bool = True) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.02,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
    ), row=1, col=1)

    # EMAs
    if show_ema:
        for col, color in [("EMA_9", COLORS["yellow"]), ("EMA_21", COLORS["blue"]),
                           ("EMA_50", COLORS["orange"]),("EMA_200", COLORS["purple"])]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                              line=dict(color=color, width=1.2)), row=1, col=1)

    # Bollinger Bands
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"],
                      name="BB Upper", line=dict(color="rgba(150,150,150,0.5)", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"],
                      name="BB Lower", line=dict(color="rgba(150,150,150,0.5)", dash="dot"),
                      fill="tonexty", fillcolor="rgba(150,150,150,0.05)"), row=1, col=1)

    # Volume
    colors = [COLORS["green"] if c >= o else COLORS["red"]
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                  marker_color=colors, opacity=0.7), row=2, col=1)

    fig.update_layout(**_LAYOUT,
                      title=dict(text=f"{symbol} – Price & Volume", font=dict(size=16)))
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# ─────────────────────────────────────────────────────────
#  Indicator sub-plots  (RSI, MACD, OBV)
# ─────────────────────────────────────────────────────────
def indicator_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3], vertical_spacing=0.04,
        subplot_titles=("RSI", "MACD", "OBV"),
    )

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                      line=dict(color=COLORS["blue"], width=1.5)), row=1, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"],   row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], row=1, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(100,100,255,0.05)",
                      line_width=0, row=1, col=1)

    # MACD
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                      line=dict(color=COLORS["blue"], width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                      line=dict(color=COLORS["orange"], width=1.5)), row=2, col=1)
        hist_colors = [COLORS["green"] if v >= 0 else COLORS["red"]
                       for v in df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram",
                     marker_color=hist_colors, opacity=0.6), row=2, col=1)

    # OBV
    if "OBV" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV",
                      line=dict(color=COLORS["purple"], width=1.5)), row=3, col=1)

    fig.update_layout(**_LAYOUT,
                      title=dict(text=f"{symbol} – Technical Indicators", font=dict(size=16)))
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# ─────────────────────────────────────────────────────────
#  LSTM forecast chart
# ─────────────────────────────────────────────────────────
def forecast_chart(
    df:       pd.DataFrame,
    forecast: np.ndarray,
    symbol:   str,
    actual_preds: np.ndarray | None = None,
) -> go.Figure:
    fig = go.Figure()

    # Historical close
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="Historical",
        line=dict(color=COLORS["blue"], width=2),
    ))

    # LSTM back-test predictions
    if actual_preds is not None and len(actual_preds) > 0:
        pred_dates = df.index[-len(actual_preds):]
        fig.add_trace(go.Scatter(
            x=pred_dates, y=actual_preds, name="LSTM (test)",
            line=dict(color=COLORS["orange"], width=1.5, dash="dot"),
        ))

    # Future forecast
    if forecast is not None and len(forecast) > 0:
        last_date    = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=len(forecast), freq="B",
        )
        # Confidence band (±2 % of std)
        std = np.std(forecast) * 0.5
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(reversed(future_dates)),
            y=list(forecast + std) + list(reversed(forecast - std)),
            fill="toself", fillcolor="rgba(0,212,180,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence Band",
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=forecast, name="Forecast",
            line=dict(color=COLORS["green"], width=2.5),
            mode="lines+markers", marker=dict(size=4),
        ))

    _apply_layout(fig, f"{symbol} – LSTM Price Forecast ({len(forecast) if forecast is not None else 0} days)")
    return fig


# ─────────────────────────────────────────────────────────
#  Model comparison bar chart
# ─────────────────────────────────────────────────────────
def model_comparison_chart(eval_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    df = eval_df.copy()
    auc_vals = pd.to_numeric(df["AUC-ROC"], errors="coerce")

    fig.add_trace(go.Bar(
        x=df["Model"], y=df["Accuracy"],
        name="Accuracy", marker_color=COLORS["blue"],
    ))

    if auc_vals.notna().any():
        fig.add_trace(go.Bar(
            x=df["Model"], y=auc_vals,
            name="AUC-ROC", marker_color=COLORS["green"],
        ))

    layout = {k: v for k, v in _LAYOUT.items() if k != "yaxis"}
    fig.update_layout(**layout,
                      title=dict(text="Model Performance Comparison", font=dict(size=16)),
                      barmode="group",
                      yaxis=dict(
                          tickformat=".2%", range=[0, 1],
                          gridcolor=COLORS["grid"], showgrid=True,
                      ),
                      )
    return fig


# ─────────────────────────────────────────────────────────
#  Feature importance horizontal bar
# ─────────────────────────────────────────────────────────
def feature_importance_chart(fi_df: pd.DataFrame, model_name: str) -> go.Figure:
    fig = px.bar(
        fi_df.sort_values("Importance"),
        x="Importance", y="Feature",
        orientation="h",
        title=f"Feature Importance – {model_name}",
        color="Importance",
        color_continuous_scale=["#4BA3FF", "#00D4B4"],
    )
    fig.update_layout(**_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────
#  Sentiment gauge
# ─────────────────────────────────────────────────────────
def sentiment_gauge(score: float, label: str) -> go.Figure:
    color = COLORS["green"] if score > 0.05 else COLORS["red"] if score < -0.05 else COLORS["yellow"]
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = score,
        delta = {"reference": 0, "valueformat": ".3f"},
        number= {"valueformat": ".3f", "font": {"size": 32, "color": color}},
        title = {"text": f"Sentiment Score<br><b>{label}</b>", "font": {"size": 14, "color": COLORS["text"]}},
        gauge = {
            "axis": {"range": [-1, 1], "tickwidth": 1, "tickcolor": COLORS["text"]},
            "bar":  {"color": color},
            "steps": [
                {"range": [-1, -0.05], "color": "rgba(255,75,75,0.2)"},
                {"range": [-0.05, 0.05], "color": "rgba(255,215,0,0.1)"},
                {"range": [0.05, 1],  "color": "rgba(0,212,180,0.2)"},
            ],
            "threshold": {"value": score, "line": {"color": color, "width": 4}},
        },
    ))
    fig.update_layout(paper_bgcolor=COLORS["bg"], font=dict(color=COLORS["text"]),
                      margin=dict(l=20, r=20, t=60, b=20), height=280)
    return fig


# ─────────────────────────────────────────────────────────
#  Fear & Greed gauge (crypto)
# ─────────────────────────────────────────────────────────
def fear_greed_gauge(value: int, label: str) -> go.Figure:
    colors = ["#FF4B4B", "#FF8C00", "#FFD700", "#A8D5A2", "#00D4B4"]
    c = colors[min(int(value // 20), 4)]
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        title = {"text": f"Fear & Greed Index<br><b>{label}</b>", "font": {"size": 14, "color": COLORS["text"]}},
        gauge = {
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": c},
            "steps": [
                {"range": [0,  25], "color": "rgba(255,75,75,0.2)"},
                {"range": [25, 50], "color": "rgba(255,140,0,0.15)"},
                {"range": [50, 75], "color": "rgba(255,215,0,0.15)"},
                {"range": [75,100], "color": "rgba(0,212,180,0.2)"},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor=COLORS["bg"], font=dict(color=COLORS["text"]),
                      margin=dict(l=20, r=20, t=60, b=20), height=280)
    return fig


# ─────────────────────────────────────────────────────────
#  Price return distribution
# ─────────────────────────────────────────────────────────
def return_distribution(df: pd.DataFrame, symbol: str) -> go.Figure:
    returns = df["Close"].pct_change().dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=50, name="Daily Return %",
        marker_color=COLORS["blue"], opacity=0.75,
    ))
    mean_r = returns.mean()
    fig.add_vline(x=mean_r, line_dash="dash", line_color=COLORS["yellow"],
                  annotation_text=f"Mean: {mean_r:.2f}%")
    _apply_layout(fig, f"{symbol} – Daily Return Distribution")
    return fig


# ─────────────────────────────────────────────────────────
#  Heatmap: monthly returns
# ─────────────────────────────────────────────────────────
def monthly_returns_heatmap(df: pd.DataFrame, symbol: str) -> go.Figure:
    df2 = df[["Close"]].copy()
    df2["Return"] = df2["Close"].pct_change() * 100
    df2["Year"]   = df2.index.year
    df2["Month"]  = df2.index.month

    pivot = df2.groupby(["Year", "Month"])["Return"].sum().unstack(fill_value=0)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    x_labels = [month_labels[m-1] for m in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=x_labels, y=[str(y) for y in pivot.index],
        colorscale=[
            [0.0, "#FF4B4B"], [0.5, "#1F2937"], [1.0, "#00D4B4"]
        ],
        zmid=0, text=pivot.values.round(1),
        texttemplate="%{text}%", textfont={"size": 9},
        showscale=True, colorbar=dict(title="Return %"),
    ))
    _apply_layout(fig, f"{symbol} – Monthly Returns Heatmap (%)")
    return fig
