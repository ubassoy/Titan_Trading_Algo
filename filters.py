"""
filters.py — Data Manager, Elite Filter, Composite Scorer

TWO-STAGE FILTERING
────────────────────
Stage 1 (pre-filter, inside run_live): cheap checks run before the expensive
  ML model. Liquidity, earnings proximity. Rejects bad candidates early.

Stage 2 (elite filter, SignalFilter): hard gates on the ML model outputs.
  ALL five conditions must pass simultaneously. Any single failure rejects.

COMPOSITE SCORE
───────────────
Signals that pass all hard gates are ranked by a weighted composite score
combining five normalised dimensions:

  Precision  (30%): OOS precision normalised to [0.54, 0.80] range
  Signal prob(20%): ML probability normalised to [0.60, 0.95] range
  Allocation (15%): Kelly allocation normalised to [3%, 15%] range
  Brier score(15%): Calibration quality normalised to [0.25, 0.10] range
  Hybrid     (20%): Rule-based score 0–100 normalised to [0, 1]

The composite score determines ranking and report ordering, not the trade
decision itself — the hard gates do that.
"""

import os
from typing import Optional
import pandas as pd
import yfinance as yf

from config import CFG, log


# ==============================================================================
# DATA MANAGER
# ==============================================================================
class DataManager:
    """Handles watchlist loading, data fetching, and pre-filters."""

    @staticmethod
    def load_watchlist() -> list[str]:
        if os.path.exists(CFG.WATCHLIST_CSV):
            try:
                df  = pd.read_csv(CFG.WATCHLIST_CSV)
                col = next(
                    (c for c in df.columns if c.lower() in ("ticker", "symbol", "tickers")),
                    df.columns[0],
                )
                tickers = df[col].dropna().astype(str).str.upper().tolist()
                log.info(f"Loaded {len(tickers)} tickers from {CFG.WATCHLIST_CSV}")
                return list(set(tickers))
            except Exception as e:
                log.warning(f"CSV read failed: {e}. Using built-in watchlist.")
        return CFG.WATCHLIST

    @staticmethod
    def fetch(
        ticker: str,
        period: str = None,
        start:  str = None,
        end:    str = None,
    ) -> Optional[pd.DataFrame]:
        try:
            if start and end:
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            else:
                df = yf.download(ticker, period=period or CFG.DATA_PERIOD, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.dropna(subset=["Close", "Volume"], inplace=True)
            return df if len(df) >= 400 else None
        except Exception:
            return None

    @staticmethod
    def passes_liquidity(df: pd.DataFrame) -> bool:
        """
        Two-part liquidity check:
          1. Average daily share volume ≥ 1.5M (enough shares to fill small orders)
          2. Average daily dollar volume ≥ $5M (enough liquidity to affect slippage)
        Both must pass. Dollar volume catches high-priced low-share-count names.
        """
        avg_vol        = df["Volume"].rolling(30).mean().iloc[-1]
        avg_dollar_vol = (df["Close"] * df["Volume"]).rolling(30).mean().iloc[-1]
        return avg_vol >= CFG.MIN_VOLUME_30D and avg_dollar_vol >= 5_000_000

    @staticmethod
    def earnings_clear(ticker: str) -> bool:
        """
        Skip stocks with earnings within EARNINGS_BUFFER_DAYS.
        Earnings cause gap risk that invalidates stop-loss assumptions —
        a 4% stop is meaningless if a stock gaps 15% on earnings.
        """
        try:
            cal = yf.Ticker(ticker).calendar
            if isinstance(cal, dict) and "Earnings Date" in cal:
                dates = cal["Earnings Date"]
                if not isinstance(dates, list):
                    dates = [dates]
                now = pd.Timestamp.now().normalize()
                for d in dates:
                    if pd.isna(d):
                        continue
                    d = pd.to_datetime(d).tz_localize(None)
                    if -2 <= (d - now).days <= CFG.EARNINGS_BUFFER_DAYS:
                        log.info(f"  ⚠ {ticker}: earnings proximity — skipping")
                        return False
            return True
        except Exception:
            return True


# ==============================================================================
# SIGNAL FILTER + COMPOSITE SCORER
# ==============================================================================
class SignalFilter:

    @staticmethod
    def above_200ma(close: pd.Series) -> bool:
        return (
            len(close) >= 200
            and float(close.iloc[-1]) > float(close.rolling(200).mean().iloc[-1])
        )

    @classmethod
    def apply_hard_filters(cls, signal: dict) -> tuple[bool, list[str]]:
        """
        Five hard gates — ALL must pass.
        A single failure rejects the signal regardless of other metrics.

        Gate 1 — OOS precision: model must be historically accurate
        Gate 2 — R/R ratio: minimum reward-to-risk (target/stop)
        Gate 3 — Kelly allocation: sizing must be meaningful (≥3%)
        Gate 4 — Above 200MA: structural trend requirement
        Gate 5 — ML probability: model must be sufficiently confident today
        Gate 6 — No earnings within buffer period
        Gate 7 — Hybrid rule score must clear minimum threshold
        """
        fails = []
        sz    = signal["sizing"]

        if signal["oos_precision"] < CFG.ELITE_MIN_OOS_PRECISION:
            fails.append(f"OOS {signal['oos_precision']*100:.1f}%<{CFG.ELITE_MIN_OOS_PRECISION*100:.0f}%")
        if (CFG.TARGET_GAIN / CFG.STOP_LOSS) < CFG.ELITE_MIN_RR:
            fails.append("R/R below min")
        if sz["alloc_pct"] < CFG.ELITE_MIN_ALLOC:
            fails.append(f"Alloc {sz['alloc_pct']:.1f}%<{CFG.ELITE_MIN_ALLOC:.0f}%")
        if CFG.ELITE_ABOVE_200MA and not signal.get("above_200ma", True):
            fails.append("Below 200MA")
        if signal["signal_prob"] < CFG.ELITE_MIN_SIGNAL_PROB:
            fails.append(f"ML {signal['signal_prob']*100:.1f}%<{CFG.ELITE_MIN_SIGNAL_PROB*100:.0f}%")
        if signal.get("earnings_close"):
            fails.append("Earnings proximity")
        if signal.get("hybrid", {}).get("hybrid_norm", 0) < CFG.ELITE_MIN_HYBRID_SCORE:
            fails.append(f"Hybrid {signal['hybrid']['hybrid_norm']}<{CFG.ELITE_MIN_HYBRID_SCORE}")

        return len(fails) == 0, fails

    @classmethod
    def compute_score(cls, signal: dict) -> float:
        """
        Weighted composite score for ranking signals that passed all gates.
        Each dimension normalised to [0, 1] before weighting.
        Weights defined in config.py and must sum to 1.0.
        """
        sz      = signal["sizing"]
        prec_n  = max(0.0, min(1.0, (signal["oos_precision"] - 0.54) / 0.26))
        prob_n  = max(0.0, min(1.0, (signal["signal_prob"] - 0.60) / 0.35))
        alloc_n = max(0.0, min(1.0, (sz["alloc_pct"] - 3.0) / 12.0))
        brier_n = max(0.0, min(1.0, (0.25 - signal["oos_brier"]) / 0.15))
        hyb_n   = signal.get("hybrid", {}).get("hybrid_norm", 50) / 100.0

        return round(
            CFG.SCORE_WEIGHT_PRECISION   * prec_n
            + CFG.SCORE_WEIGHT_SIGNAL_PROB * prob_n
            + CFG.SCORE_WEIGHT_ALLOC       * alloc_n
            + CFG.SCORE_WEIGHT_BRIER       * brier_n
            + CFG.SCORE_WEIGHT_HYBRID      * hyb_n,
            4,
        )

    @classmethod
    def run(cls, signals: list[dict]) -> tuple[list[dict], list[dict]]:
        """Filter signals into elite/rejected. Rank elite by composite score."""
        elite, rejected = [], []
        for s in signals:
            passed, reasons = cls.apply_hard_filters(s)
            if passed:
                s["elite_score"] = cls.compute_score(s)
                elite.append(s)
            else:
                s["reject_reasons"] = reasons
                rejected.append(s)
        elite.sort(key=lambda x: x["elite_score"], reverse=True)
        return elite[:CFG.MAX_SIGNALS], rejected
