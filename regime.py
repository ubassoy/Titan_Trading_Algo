"""
regime.py — Market Regime Classifier + Interest Rate Loader + Macro Data Loader

DESIGN DECISION: Rules-based, NOT machine-learned.
  Rationale: there are only ~30-40 genuine regime shifts in 20 years of data.
  That is far too few observations to train an ML model without catastrophic
  overfitting. Rules derived from economic principles are transparent,
  auditable, and do not overfit.

Regime tiers and their sizing multipliers:
  BULL    → 1.00x  (all systems go)
  NEUTRAL → 0.60x  (proceed with caution)
  BEAR    → 0.25x  (very selective, minimal sizing)
  CRASH   → 0.00x  (no new longs — VIX > 35)

The key function compute_regime_from_slice() is separated from the class
so it can be called per bar during backtesting without look-ahead bias.
"""

from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf

from config import CFG, log


# ==============================================================================
# PURE FUNCTION — called by both live RegimeClassifier and BacktestRegimeEngine
# ==============================================================================
def compute_regime_from_slice(
    spy_slice: pd.Series,
    vix_slice: pd.Series,
    hyg_slice: pd.Series,
    qqq_slice: pd.Series,
    iwm_slice: pd.Series,
) -> dict:
    """
    Reconstruct regime features using only data available up to a given bar.
    All inputs are pd.Series sliced to [:current_bar] — no future data.

    Scoring logic (max 5 bullish signals):
      VIX < CALM   → +2  (low volatility is a premium environment)
      VIX < NORMAL → +1  (normal conditions)
      SPY > 200MA  → +1  (structural uptrend)
      SPY positive 1m momentum → +1
      HYG > 20MA (no credit stress) → +1

    Regime assignment:
      VIX >= CRASH  → CRASH   (veto — sizing = 0)
      bullish >= 4  → BULL
      bullish >= 2  → NEUTRAL
      else          → BEAR
    """
    try:
        vix       = float(vix_slice.iloc[-1])
        spy_200ma = spy_slice.rolling(200).mean().iloc[-1]
        spy_trend = "ABOVE" if float(spy_slice.iloc[-1]) > float(spy_200ma) else "BELOW"
        spy_mom   = float(spy_slice.iloc[-1] / spy_slice.iloc[-21] - 1) if len(spy_slice) > 21 else 0.0
        hyg_20ma  = hyg_slice.rolling(20).mean().iloc[-1]
        credit_stress = bool(float(hyg_slice.iloc[-1]) < float(hyg_20ma))
        qqq_ret   = float(qqq_slice.pct_change(5).iloc[-1]) if len(qqq_slice) > 5 else 0.0
        iwm_ret   = float(iwm_slice.pct_change(5).iloc[-1]) if len(iwm_slice) > 5 else 0.0
        risk_app  = (qqq_ret + iwm_ret) / 2

        bullish = 0
        if vix < CFG.VIX_CALM:     bullish += 2
        elif vix < CFG.VIX_NORMAL: bullish += 1
        if spy_trend == "ABOVE":   bullish += 1
        if spy_mom > 0:            bullish += 1
        if not credit_stress:      bullish += 1

        if vix >= CFG.VIX_CRASH:
            regime, mult = "CRASH",   0.0
        elif bullish >= 4:
            regime, mult = "BULL",    1.0
        elif bullish >= 2:
            regime, mult = "NEUTRAL", 0.6
        else:
            regime, mult = "BEAR",    0.25

        return {
            "vix":              round(vix, 2),
            "vix_5d_avg":       round(float(vix_slice.tail(5).mean()), 2),
            "spy_trend":        spy_trend,
            "spy_1m_ret":       round(spy_mom * 100, 2),
            "credit_stress":    credit_stress,
            "risk_appetite_5d": round(risk_app * 100, 2),
            "bullish_signals":  bullish,
            "regime":           regime,
            "sizing_mult":      mult,
            "vix_tier": (
                "CALM"   if vix < CFG.VIX_CALM   else
                "NORMAL" if vix < CFG.VIX_NORMAL  else
                "STRESS" if vix < CFG.VIX_STRESS  else
                "CRASH"
            ),
        }
    except Exception:
        return {
            "vix": 20.0, "vix_tier": "NORMAL", "spy_trend": "ABOVE",
            "credit_stress": False, "risk_appetite_5d": 0.0,
            "spy_1m_ret": 0.0, "bullish_signals": 2,
            "regime": "NEUTRAL", "sizing_mult": 0.6,
        }


# ==============================================================================
# LIVE REGIME CLASSIFIER
# ==============================================================================
class RegimeClassifier:
    """Fetches live market data and classifies the current regime."""

    def __init__(self):
        self.vix:               Optional[float] = None
        self.spy_trend:         Optional[str]   = None
        self.credit_stress:     Optional[bool]  = None
        self.regime:            str             = "UNKNOWN"
        self.sizing_multiplier: float           = 0.0
        self.regime_features:   dict            = {}

    def run(self) -> bool:
        log.info("Stage 0 ▶ Regime detection (live)...")
        try:
            raw   = yf.download(CFG.REGIME_ASSETS, period="1y", progress=False, auto_adjust=True)
            close = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            close.dropna(how="all", inplace=True)

            feats = compute_regime_from_slice(
                spy_slice=close["SPY"].dropna(),
                vix_slice=close["^VIX"].dropna(),
                hyg_slice=close["HYG"].dropna(),
                qqq_slice=close["QQQ"].dropna(),
                iwm_slice=close["IWM"].dropna(),
            )

            self.vix               = feats["vix"]
            self.spy_trend         = feats["spy_trend"]
            self.credit_stress     = feats["credit_stress"]
            self.regime            = feats["regime"]
            self.sizing_multiplier = feats["sizing_mult"]
            self.regime_features   = feats

            log.info(
                f"  Regime: {self.regime} | VIX: {self.vix:.1f} "
                f"({feats['vix_tier']}) | Sizing: {self.sizing_multiplier*100:.0f}%"
            )
            return True

        except Exception as e:
            log.error(f"Regime check failed: {e}")
            self.regime, self.sizing_multiplier = "UNKNOWN", 0.3
            self.regime_features = compute_regime_from_slice(
                pd.Series([400.0]), pd.Series([20.0]),
                pd.Series([80.0]), pd.Series([350.0]), pd.Series([200.0]),
            )
            return True


# ==============================================================================
# INTEREST RATE LOADER
# ==============================================================================
class InterestRateLoader:
    """
    Fetches 10Y Treasury yield (^TNX).

    Why 10Y Treasury matters for equities:
      Rising 10Y = higher discount rate = lower present value of future earnings.
      This is the risk-free rate in DCF models. When it rises fast, growth stocks
      get repriced most severely (longer duration assets hurt more).
      Falling 10Y = easing cycle = expansion of valuation multiples.

    Features generated:
      rate_10y        : Current yield level (%)
      rate_1m_change  : 21-day change
      rate_3m_change  : 63-day change (primary signal)
      rate_6m_change  : 126-day change (structural trend)
      rates_rising    : Binary — 1 if 3m change > 0.25%
      rates_high      : Binary — 1 if yield > 4.5%
    """

    def __init__(self):
        self.rate_data: Optional[pd.DataFrame] = None

    def load(self, period: str = None) -> bool:
        p = period or CFG.DATA_PERIOD
        log.info(f"Rates ▶ Loading 10Y Treasury yield (^TNX, period={p})...")
        try:
            raw = yf.download("^TNX", period=p, progress=False, auto_adjust=True)
            if raw.empty:
                log.warning("  No rate data — features will be zero-filled.")
                return False
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            rates = pd.DataFrame(index=raw.index)
            rates["treasury_10y"]   = raw["Close"]
            rates["rate_1m_change"] = rates["treasury_10y"].diff(21)
            rates["rate_3m_change"] = rates["treasury_10y"].diff(63)
            rates["rate_6m_change"] = rates["treasury_10y"].diff(126)
            rates["rates_rising"]   = (rates["rate_3m_change"] > 0.25).astype(int)
            rates["rates_high"]     = (rates["treasury_10y"] > 4.5).astype(int)

            self.rate_data = rates
            log.info(
                f"  Rates loaded: {len(rates)} days | "
                f"Current 10Y: {rates['treasury_10y'].iloc[-1]:.2f}% | "
                f"Δ3m: {rates['rate_3m_change'].iloc[-1]:+.2f}%"
            )
            return True
        except Exception as e:
            log.warning(f"  Rate load failed: {e}. Zeroing rate features.")
            self.rate_data = None
            return False


# ==============================================================================
# MACRO DATA LOADER
# ==============================================================================
class MacroDataLoader:
    """
    Loads cross-asset macro returns: Euro Stoxx 50, SPY, GLD.

    Why cross-asset features matter:
      EU_Return    : EU opens before US → lead-lag signal for global risk appetite
      SPY_Return   : Used to compute relative strength per stock
      GLD_Return   : Used to compute correlation — high corr = risk-off behaviour
    """

    def __init__(self):
        self.macro_returns: Optional[pd.DataFrame] = None

    def load(self, period: str = None) -> bool:
        p = period or CFG.DATA_PERIOD
        log.info(f"Macro ▶ Loading cross-asset data (period={p})...")
        try:
            tickers = list(CFG.MACRO_TICKERS.values())
            raw     = yf.download(tickers, period=p, progress=False, auto_adjust=True)
            prices  = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            prices.dropna(how="all", inplace=True)
            prices.rename(
                columns={v: CFG.MACRO_COLS[v] for v in tickers if v in CFG.MACRO_COLS},
                inplace=True,
            )
            self.macro_returns = np.log(prices / prices.shift(1))
            log.info(f"  Macro loaded: {len(self.macro_returns)} days")
            return True
        except Exception as e:
            log.warning(f"  Macro load failed: {e}. Zeroing macro features.")
            self.macro_returns = None
            return False
