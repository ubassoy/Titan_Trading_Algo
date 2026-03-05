"""
main.py — Entry Point: Live Signal Scanner + Report

HOW TO RUN
──────────
Terminal:
    python main.py                  → live signal scan
    python main.py --backtest       → backtest on full watchlist
    python main.py --backtest --start 2021-01-01 --end 2024-01-01

Google Colab / Jupyter:
    Set RUN_MODE below, then run the cell.
    RUN_MODE = "live"       → live signal scan
    RUN_MODE = "backtest"   → backtest

PIPELINE STAGES (live mode)
────────────────────────────
Stage 0 : Regime detection + interest rate loading + macro data
Stage 1 : Data fetch + liquidity filter + earnings check (per ticker)
Stage 2 : Feature engineering (46 features)
Stage 3 : Walk-forward XGBoost + Platt calibration (per ticker)
Stage 4 : Wilson CI + Half-Kelly position sizing
Stage 5 : HybridScorer rule-based validation
Stage 6 : Elite filter (7 hard gates) + composite scoring
Stage 7 : Signal report + CSV export
"""

import argparse
from datetime import datetime
from typing import Optional

import pandas as pd

from config import CFG, log
from regime import RegimeClassifier, MacroDataLoader, InterestRateLoader
from features import FeatureEngine, FEATURE_COLS, build_labels
from model import AlphaModel
from sizing import PositionSizer
from scoring import HybridScorer
from filters import DataManager, SignalFilter


# ==============================================================================
# SIGNAL REPORT
# ==============================================================================
def print_report(
    signals:  list[dict],
    rejected: list[dict],
    regime:   RegimeClassifier,
    rates:    InterestRateLoader,
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rf  = regime.regime_features

    print()
    print("=" * 125)
    print(f"  TITAN v7.1 — MACRO-QUANT HYBRID SIGNAL REPORT    {now}")
    print("=" * 125)

    rl = {
        "BULL": "🟢 BULL", "NEUTRAL": "🟡 NEUTRAL",
        "BEAR": "🔴 BEAR", "CRASH": "🚨 CRASH",
    }.get(regime.regime, "⚪ UNKNOWN")

    print(
        f"\n  REGIME : {rl}  |  VIX: {rf.get('vix','?'):.1f} ({rf.get('vix_tier','?')})  |  "
        f"SPY: {rf.get('spy_trend','?')}  |  "
        f"Credit: {'STRESS ⚠' if rf.get('credit_stress') else 'OK ✓'}  |  "
        f"Sizing: {regime.sizing_multiplier*100:.0f}%"
    )

    if rates.rate_data is not None and not rates.rate_data.empty:
        current_rate = float(rates.rate_data["treasury_10y"].iloc[-1])
        rate_3m_chg  = float(rates.rate_data["rate_3m_change"].iloc[-1])
        rate_status  = (
            "📈 RISING"  if rate_3m_chg > 0.25 else
            "📉 FALLING" if rate_3m_chg < -0.25 else
            "➡️  STABLE"
        )
        print(
            f"  RATES  : 10Y Treasury: {current_rate:.2f}%  "
            f"({rate_status}, Δ3m: {rate_3m_chg:+.2f}%)"
        )

    print(
        f"  TARGET : +{CFG.TARGET_GAIN*100:.0f}%  |  STOP: -{CFG.STOP_LOSS*100:.0f}%  |  "
        f"R/R: {CFG.TARGET_GAIN/CFG.STOP_LOSS:.1f}:1  |  "
        f"Account: ${CFG.ACCOUNT_SIZE:,.0f}\n"
    )

    if regime.regime == "CRASH":
        print("  🚨 CRASH — VIX>35. NO NEW LONGS.\n")

    if not signals:
        print("  ❌  No signals passed all filters today.\n" + "=" * 125)
        return

    # ── Signal table ──────────────────────────────────────────────────────────
    hdr = (
        f"  {'#':<3} | {'TICKER':<7} | {'SCORE':>6} | {'PRICE':>7} | "
        f"{'ML%':>6} | {'OOS':>6} | {'W-LB':>5} | {'BRIER':>5} | "
        f"{'ALLOC':>6} | {'$SIZE':>8} | {'HYB':>4} | "
        f"{'52WH':>5} | {'RS_SPY':>7} | {'GLD':>5}"
    )
    print(hdr)
    print("  " + "-" * 115)

    for rank, s in enumerate(signals, 1):
        sz  = s["sizing"]
        hyb = s.get("hybrid", {})
        rs  = s.get("rel_str_spy_pct", 0.0)
        gc  = s.get("corr_gld_20d", 0.0)
        p52 = s.get("price_to_52w_high", 0.0)
        ma  = "✓" if s.get("above_200ma") else "✗"
        rsf = "🟢" if rs > 0 else "🔴"
        print(
            f"  [{rank}] | {s['ticker']:<7} | {s.get('elite_score',0):>6.3f} | "
            f"{s['price']:>7.2f} | {s['signal_prob']*100:>5.1f}% | "
            f"{s['oos_precision']*100:>5.1f}% | {sz['win_rate_ci_lb']:>4.1f}% | "
            f"{s['oos_brier']:>5.3f} | {sz['alloc_pct']:>5.2f}% | "
            f"${sz['dollar_size']:>7,.0f} | {hyb.get('hybrid_norm',0):>3}% | "
            f"{p52*100:>4.0f}% | {rsf}{rs:>+5.2f}% | "
            f"{gc:>5.2f}  200MA:{ma}"
        )

    # ── Trade sheet ───────────────────────────────────────────────────────────
    print("\n  " + "─" * 115)
    print("  TRADE SHEET")
    print("  " + "─" * 115)
    total = 0.0
    for i, s in enumerate(signals, 1):
        sz = s["sizing"]
        if sz["dollar_size"] < CFG.MIN_POSITION_DOLLAR:
            continue
        shares = int(sz["dollar_size"] / s["price"])
        stop   = round(s["price"] * (1 - CFG.STOP_LOSS), 2)
        tgt    = round(s["price"] * (1 + CFG.TARGET_GAIN), 2)
        total += sz["alloc_pct"]
        print(
            f"  [{i}] {s['ticker']:<6}  BUY {shares} @ ~${s['price']:.2f}"
            f"  │  Stop: ${stop}  │  Target: ${tgt}"
            f"  │  Risk: ${round(shares * s['price'] * CFG.STOP_LOSS):,.0f}"
        )
    print(
        f"\n  ALLOCATED: {total:.1f}%  "
        f"(${total/100*CFG.ACCOUNT_SIZE:,.0f})  |  CASH: {100-total:.1f}%\n"
    )

    # ── Hybrid score breakdown ────────────────────────────────────────────────
    print("  HYBRID SCORE BREAKDOWN")
    print("  " + "─" * 115)
    for s in signals:
        hyb = s.get("hybrid", {})
        print(
            f"  {s['ticker']:<7} [{hyb.get('hybrid_norm',0):>3}%] : "
            f"{' · '.join(hyb.get('reasons', [])[:6])}"
        )
    print()

    # ── Feature importance (top 3 signals) ───────────────────────────────────
    print("  TOP PREDICTIVE FEATURES (top 3 signals)")
    print("  " + "─" * 115)
    for i, s in enumerate(signals[:3], 1):
        feat_imp = s.get("feature_importance")
        if feat_imp is not None:
            print(f"\n  [{i}] {s['ticker']}:")
            for feat, imp in feat_imp.head(5).items():
                print(f"      {feat:<25} : {imp:.4f}")
    print()

    # ── Rejected signals ──────────────────────────────────────────────────────
    if rejected:
        print(f"  REJECTED ({len(rejected)} failed elite filter):")
        for s in sorted(rejected, key=lambda x: x["oos_precision"], reverse=True)[:5]:
            print(
                f"    {s['ticker']:<7} OOS:{s['oos_precision']*100:.1f}%  "
                f"ML:{s['signal_prob']*100:.1f}%  "
                f"Hyb:{s.get('hybrid',{}).get('hybrid_norm',0)}%  "
                f"→ {' | '.join(s.get('reject_reasons', ['?']))}"
            )
    print("\n" + "=" * 125)

    # ── CSV export ────────────────────────────────────────────────────────────
    rows = [{
        "date":              now,
        "ticker":            s["ticker"],
        "price":             s["price"],
        "signal_prob_pct":   round(s["signal_prob"] * 100, 2),
        "oos_precision_pct": round(s["oos_precision"] * 100, 2),
        "win_rate_ci_lb":    s["sizing"]["win_rate_ci_lb"],
        "brier_score":       s["oos_brier"],
        "alloc_pct":         s["sizing"]["alloc_pct"],
        "dollar_size":       s["sizing"]["dollar_size"],
        "stop_price":        round(s["price"] * (1 - CFG.STOP_LOSS), 2),
        "target_price":      round(s["price"] * (1 + CFG.TARGET_GAIN), 2),
        "hybrid_score":      s.get("hybrid", {}).get("hybrid_norm", 0),
        "hybrid_reasons":    " | ".join(s.get("hybrid", {}).get("reasons", [])),
        "price_to_52w_high": round(s.get("price_to_52w_high", 0.0), 4),
        "rel_str_spy_pct":   round(s.get("rel_str_spy_pct", 0.0), 4),
        "corr_gld_20d":      round(s.get("corr_gld_20d", 0.0), 4),
        "regime":            regime.regime,
        "vix":               rf.get("vix", ""),
        "above_200ma":       s.get("above_200ma", False),
        "elite_score":       s.get("elite_score", 0.0),
    } for s in signals]
    pd.DataFrame(rows).to_csv(CFG.OUTPUT_CSV, index=False)
    log.info(f"Signals saved → {CFG.OUTPUT_CSV}")


# ==============================================================================
# LIVE SCAN PIPELINE
# ==============================================================================
def run_live():
    print("\n" + "=" * 125)
    print("  TITAN v7.1 — Macro-Quant Hybrid Signal Engine")
    print("=" * 125)

    # Stage 0
    regime = RegimeClassifier()
    regime.run()
    macro = MacroDataLoader()
    macro.load()
    rates = InterestRateLoader()
    rates.load()

    watchlist   = DataManager.load_watchlist()
    raw_signals = []

    for ticker in watchlist:
        log.info(f"Processing {ticker}...")

        # Stage 1
        df = DataManager.fetch(ticker)
        if df is None or not DataManager.passes_liquidity(df):
            continue
        earnings_ok = DataManager.earnings_clear(ticker)

        # Stage 2
        df_feat = FeatureEngine.build(
            df, regime.regime_features, macro.macro_returns,
            rate_data=rates.rate_data,
        )
        labels = build_labels(df)

        # Stage 3
        model = AlphaModel(ticker)
        if not model.run(df_feat, labels):
            continue
        signal_prob = model.predict_latest(df_feat)
        if signal_prob is None or signal_prob < 0.52:
            continue

        # Stage 4
        sizing = PositionSizer.compute(
            oos_precision=model.oos_precision,
            oos_n=model.n_oos_samples,
            current_prob=signal_prob,
            regime_multiplier=regime.sizing_multiplier,
        )
        if sizing["alloc_pct"] < CFG.MIN_KELLY_AFTER_CI * 100:
            continue

        # Stage 5
        latest = df_feat.dropna(subset=FEATURE_COLS).iloc[-1]
        hybrid = HybridScorer.compute(latest, regime.regime_features)

        raw_signals.append({
            "ticker":             ticker,
            "price":              float(df["Close"].iloc[-1]),
            "signal_prob":        signal_prob,
            "oos_precision":      model.oos_precision,
            "oos_brier":          model.oos_brier,
            "oos_n":              model.n_oos_samples,
            "n_folds":            model.n_folds,
            "sizing":             sizing,
            "earnings_close":     not earnings_ok,
            "above_200ma":        SignalFilter.above_200ma(df["Close"]),
            "hybrid":             hybrid,
            "rel_str_spy_pct":    float(latest.get("rel_str_spy", 0.0)) * 100,
            "corr_gld_20d":       float(latest.get("corr_gld_20d", 0.0)),
            "price_to_52w_high":  float(latest.get("price_to_52w_high", 0.0)),
            "feature_importance": model.feature_importance,
        })

        log.info(
            f"  ✓ {ticker}: ML={signal_prob*100:.1f}%  "
            f"OOS={model.oos_precision*100:.1f}%  "
            f"Hyb={hybrid['hybrid_norm']}%  "
            f"alloc={sizing['alloc_pct']:.1f}%"
        )

    log.info(f"Scan done. {len(raw_signals)}/{len(watchlist)} passed pre-filters.")

    # Stage 6
    elite, rejected = SignalFilter.run(raw_signals)
    log.info(f"Elite: {len(elite)} | Rejected: {len(rejected)}")

    # Stage 7
    print_report(elite, rejected, regime, rates)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

# ── Colab / Jupyter: edit these two lines ─────────────────────────────────────
RUN_MODE = "live"        # "live" or "backtest"
BT_START = "2021-01-01"
BT_END   = datetime.today().strftime("%Y-%m-%d")
# ─────────────────────────────────────────────────────────────────────────────


def _is_notebook() -> bool:
    """Detect Colab / Jupyter environment to bypass argparse."""
    try:
        return get_ipython().__class__.__name__ in (  # type: ignore
            "ZMQInteractiveShell", "Shell", "TerminalInteractiveShell"
        )
    except NameError:
        return False


if __name__ == "__main__":
    if _is_notebook():
        # Colab/Jupyter: use RUN_MODE variable above
        if RUN_MODE == "backtest":
            from backtest import run_backtest
            run_backtest(BT_START, BT_END)
        else:
            run_live()
    else:
        # Terminal: use argparse flags
        parser = argparse.ArgumentParser(description="TITAN v7.1 Signal Engine")
        parser.add_argument("--backtest", action="store_true", help="Run backtester")
        parser.add_argument("--start", type=str, default="2021-01-01",
                            help="Backtest start (YYYY-MM-DD)")
        parser.add_argument("--end", type=str,
                            default=datetime.today().strftime("%Y-%m-%d"),
                            help="Backtest end (YYYY-MM-DD)")
        args = parser.parse_args()

        if args.backtest:
            from backtest import run_backtest
            run_backtest(args.start, args.end)
        else:
            run_live()
