# ============================================================
#   models/sentiment.py  –  News / Social Sentiment Analysis
# ============================================================

import os, requests
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── VADER (no API key needed) ────────────────────────────
_analyzer = SentimentIntensityAnalyzer()


def analyze_text(text: str) -> dict:
    """Return VADER scores for a single text snippet."""
    scores = _analyzer.polarity_scores(str(text))
    label  = (
        "POSITIVE" if scores["compound"] >= 0.05  else
        "NEGATIVE" if scores["compound"] <= -0.05 else
        "NEUTRAL"
    )
    return {**scores, "label": label}


def analyze_batch(texts: list[str]) -> pd.DataFrame:
    """Analyze a list of texts. Returns a per-text DataFrame."""
    rows = [analyze_text(t) for t in texts]
    return pd.DataFrame(rows)


def aggregate_sentiment(texts: list[str]) -> dict:
    """Aggregate sentiment scores over a list of texts."""
    if not texts:
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0,
                "label": "NEUTRAL", "count": 0}

    df = analyze_batch(texts)
    mean_comp = df["compound"].mean()
    label = (
        "POSITIVE" if mean_comp >= 0.05 else
        "NEGATIVE" if mean_comp <= -0.05 else
        "NEUTRAL"
    )
    return {
        "compound": round(float(mean_comp), 4),
        "pos":      round(float(df["pos"].mean()), 4),
        "neu":      round(float(df["neu"].mean()), 4),
        "neg":      round(float(df["neg"].mean()), 4),
        "label":    label,
        "count":    len(texts),
    }


# ─────────────────────────────────────────────────────────
#  Fetch live news via NewsAPI (free tier, optional)
# ─────────────────────────────────────────────────────────
def fetch_news_headlines(
    query: str,
    api_key: str | None = None,
    max_articles: int   = 20,
) -> list[str]:
    """
    Fetch recent news headlines.
    Falls back to demo headlines if no API key is provided.
    """
    if api_key:
        url = (
            "https://newsapi.org/v2/everything"
            f"?q={query}&sortBy=publishedAt&pageSize={max_articles}"
            f"&language=en&apiKey={api_key}"
        )
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                articles = resp.json().get("articles", [])
                headlines = []
                for a in articles:
                    title = a.get("title", "")
                    desc  = a.get("description", "")
                    if title:
                        headlines.append(f"{title}. {desc or ''}")
                return headlines
        except Exception:
            pass

    # ── Demo headlines if no key ──────────────────────────
    return _demo_headlines(query)


def _demo_headlines(query: str) -> list[str]:
    """Synthetic balanced demo headlines for any asset."""
    q = query.upper()
    return [
        f"{q} shows strong bullish momentum as institutional investors increase positions.",
        f"Analysts upgrade {q} to outperform with raised price targets.",
        f"Market volatility creates uncertainty around {q} future outlook.",
        f"{q} breaks key resistance level, technical analysis points higher.",
        f"Bearish divergence signals potential correction in {q} near term.",
        f"Strong earnings report boosts confidence in {q} long-term growth.",
        f"Regulatory concerns weigh heavily on {q} market sentiment today.",
        f"{q} adoption grows as new partnerships are announced by major players.",
        f"Profit-taking observed in {q} after recent strong performance rally.",
        f"Macro headwinds pose risk to {q} short-term price action.",
        f"Technical indicators suggest {q} is in a consolidation phase now.",
        f"Positive on-chain data signals accumulation phase for {q}.",
    ]


# ─────────────────────────────────────────────────────────
#  Fear & Greed index (crypto – from alternative.me)
# ─────────────────────────────────────────────────────────
def fetch_fear_greed_index() -> dict:
    """Fetch the Crypto Fear & Greed Index (free, no key)."""
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=7", timeout=8)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            latest = data[0] if data else {}
            history = [
                {
                    "value":       int(d.get("value", 50)),
                    "label":       d.get("value_classification", "Neutral"),
                    "timestamp":   d.get("timestamp", ""),
                }
                for d in data
            ]
            return {
                "value":   int(latest.get("value", 50)),
                "label":   latest.get("value_classification", "Neutral"),
                "history": history,
            }
    except Exception:
        pass

    return {"value": 50, "label": "Neutral", "history": []}


# ─────────────────────────────────────────────────────────
#  Combined sentiment score for dashboard
# ─────────────────────────────────────────────────────────
def get_market_sentiment(
    symbol: str,
    asset_type: str     = "stock",
    news_api_key: str | None = None,
) -> dict:
    """
    Return a unified sentiment dict with:
    - news_sentiment (vader over headlines)
    - fear_greed     (only for crypto)
    - overall_score  [-1 …+1]
    - overall_label
    """
    query     = symbol if asset_type == "stock" else f"{symbol} cryptocurrency"
    headlines = fetch_news_headlines(query, news_api_key)
    news_sent = aggregate_sentiment(headlines)
    news_sent["headlines"] = headlines[:10]

    result = {
        "news_sentiment": news_sent,
        "fear_greed":     None,
        "overall_score":  news_sent["compound"],
        "overall_label":  news_sent["label"],
    }

    if asset_type == "crypto":
        fg = fetch_fear_greed_index()
        result["fear_greed"] = fg
        # Blend news (60%) + fear/greed (40%, normalised to [-1,+1])
        fg_norm = (fg["value"] - 50) / 50
        blended = 0.6 * news_sent["compound"] + 0.4 * fg_norm
        result["overall_score"] = round(float(blended), 4)
        result["overall_label"] = (
            "POSITIVE" if blended >= 0.05 else
            "NEGATIVE" if blended <= -0.05 else
            "NEUTRAL"
        )

    return result
