# ============================================================
#   main.py  –  CLI entry point (headless analysis)
#   Usage:
#       python main.py --symbol AAPL --type stock
#       python main.py --symbol BTC  --type crypto --no-lstm
# ============================================================

import argparse, warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from analysis.trend_analyzer import TrendAnalyzer


def parse_args():
    p = argparse.ArgumentParser(description="AI Market Trend Analyzer – CLI")
    p.add_argument("--symbol",  "-s", default="AAPL",  help="Ticker / coin symbol")
    p.add_argument("--type",    "-t", default="stock",
                   choices=["stock", "crypto"], help="Asset type")
    p.add_argument("--period",  "-p", default="1y",    help="History period (stocks only)")
    p.add_argument("--no-lstm", action="store_true",   help="Skip LSTM forecasting")
    p.add_argument("--epochs",  "-e", default=60, type=int, help="LSTM epochs")
    p.add_argument("--news-key",       default=None,
                   help="NewsAPI key (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    analyzer = TrendAnalyzer(args.symbol, args.type)
    fetch_kw = {}
    if args.type == "stock":
        fetch_kw = {"period": args.period}

    result = analyzer.run(
        train_lstm        = not args.no_lstm,
        compute_sentiment = True,
        lstm_epochs       = args.epochs,
        news_api_key      = args.news_key,
        **fetch_kw,
    )

    # ── Print summary ──────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  SUMMARY — {result['symbol']} ({result['asset_type'].upper()})")
    print("=" * 55)

    info = result["info"]
    print(f"  Name         : {info.get('name', args.symbol)}")
    print(f"  Latest Close : ${result['raw_df']['Close'].iloc[-1]:,.2f}")
    print()

    print("  ML Model Accuracy:")
    for _, row in result["eval_df"].iterrows():
        print(f"    {row['Model']:<22} {row['Accuracy']*100:5.2f}%")
    print(f"  → Best: {result['best_model']}")
    print()

    print(f"  ML Signal    : {result['ml_signal']}")
    print(f"  Tech Signal  : {result['tech_signal']}")

    if result.get("sentiment"):
        s = result["sentiment"]
        print(f"  Sentiment    : {s.get('overall_label','N/A')} ({s.get('overall_score',0):.4f})")

    if result.get("forecast") is not None:
        fc = result["forecast"]
        print(f"  30d Forecast : ${fc[-1]:,.2f} (final predicted price)")
        if result.get("lstm_metrics"):
            lm = result["lstm_metrics"]
            print(f"  LSTM RMSE    : ${lm.get('RMSE',0):,.2f}")
            print(f"  LSTM Dir Acc : {lm.get('Direction Acc',0)*100:.1f}%")

    print("\n" + "=" * 55)
    print("  Run  `streamlit run dashboard/app.py`  for the")
    print("  full interactive dashboard.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
