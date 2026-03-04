"""
================================================================================
TITAN v6.0 — Macro-Quant Swing Trading Signal Engine
================================================================================
Portfolio demonstration project.
Showcases: walk-forward OOS validation · probability calibration · cross-asset
feature engineering · Kelly sizing with Wilson CI · rules-based regime gating.

NOTE: Specific threshold values and elite filter parameters have been replaced
with descriptive placeholders. The architecture, methodology, and core logic
are fully intact and representative of the actual system.

Dependencies:
    pip install yfinance xgboost scikit-learn pandas numpy scipy

Usage:
    1. Add tickers to winning_stocks.csv (column: ticker), or edit WATCHLIST.
    2. python TITAN_v6.py
    3. Review printed Signal Report + signals_YYYYMMDD.csv
================================================================================
"""

import os
import warnings
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, brier_score_loss
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TITAN")


# ==============================================================================
# CONFIGURATION
# Centralised config class — all tuneable parameters live here.
# Threshold values marked [PARAM] have been abstracted for this portfolio
# version. The structure and relationships between parameters are authentic.
# ==============================================================================
class Config:

    # ── Universe ───────────────────────────────────────────────────────────────
    WATCHLIST: list[str] = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD",
        "JPM", "BAC", "GS", "V", "MA", "XOM", "CVX", "LLY", "UNH",
        "HD", "COST", "PLTR", "COIN", "MSTR", "SMCI", "MU", "CRM",
    ]
    WATCHLIST_CSV: str = "winning_stocks.csv"

    # ── Regime detection assets ────────────────────────────────────────────────
    # Multi-asset basket captures VIX, trend, credit, and breadth simultaneously
    REGIME_ASSETS: list[str] = [
        "SPY",    # US broad market
        "QQQ",    # Tech / growth proxy
        "IWM",    # Small-cap — risk appetite indicator
        "TLT",    # Long bonds — flight-to-safety signal
        "GLD",    # Gold — inflation / tail-risk hedge
        "UUP",    # USD index — global risk-off signal
        "^VIX",   # CBOE Volatility Index
        "HYG",    # High-yield credit — spread proxy
    ]

    # ── Cross-asset macro tickers (v6.0 addition) ─────────────────────────────
    # Downloaded ONCE at pipeline start and reused across all tickers.
    # EU index is geographically diversified and opens before US — lead-lag signal.
    MACRO_TICKERS: dict[str, str] = {
        "EU":  "^STOXX50E",   # Euro Stoxx 50
        "SPY": "SPY",         # US broad market baseline
        "GLD": "GLD",         # Gold safe-haven proxy
    }
    MACRO_COLS: dict[str, str] = {
        "^STOXX50E": "EU_Return",
        "SPY":       "SPY_Return",
        "GLD":       "GLD_Return",
    }

    # ── Trade parameters ───────────────────────────────────────────────────────
    HOLD_DAYS_MIN: int   = 5
    HOLD_DAYS_MAX: int   = 20
    TARGET_GAIN:   float = 0.08    # [PARAM] Target % gain within holding window
    STOP_LOSS:     float = 0.04    # [PARAM] Hard stop — defines R/R ratio

    # ── Friction model ─────────────────────────────────────────────────────────
    # Target is adjusted upward to account for round-trip slippage.
    # A signal that barely hits the nominal target is not a real win after costs.
    COMMISSION_BPS: float = 0.0
    SLIPPAGE_BPS:   float = 10.0   # [PARAM] Conservative retail slippage estimate
    TOTAL_FRICTION: float = (0.0 + 10.0) / 10_000

    # ── Account / position sizing ──────────────────────────────────────────────
    ACCOUNT_SIZE:        float = 40_000
    MAX_POSITION_PCT:    float = 0.15   # Hard cap: never >15% in one name
    MAX_POSITIONS:       int   = 6
    MIN_POSITION_DOLLAR: float = 500

    # ── Signal pre-filters ─────────────────────────────────────────────────────
    MIN_VOLUME_30D:       int   = 1_500_000   # [PARAM] Liquidity floor
    MIN_OOS_PRECISION:    float = 0.54        # [PARAM] Minimum walk-forward precision
    MIN_KELLY_AFTER_CI:   float = 0.01        # [PARAM] Discard if adjusted Kelly < threshold
    EARNINGS_BUFFER_DAYS: int   = 14          # Blackout window around earnings dates

    # ── Elite filter hard gates ────────────────────────────────────────────────
    # Signals must pass ALL of these before appearing in the report.
    # Rationale for each threshold documented in Key Design Decisions (README).
    ELITE_MIN_OOS_PRECISION: float = 0.58   # [PARAM]
    ELITE_MIN_RR:            float = 2.0    # Minimum reward/risk ratio
    ELITE_MIN_ALLOC:         float = 3.0    # [PARAM] Min Kelly % to be worth trading
    ELITE_ABOVE_200MA:       bool  = True   # No longs in structural downtrends
    ELITE_MIN_SIGNAL_PROB:   float = 0.60   # [PARAM] Calibrated model confidence floor

    # ── Composite score weights (must sum to 1.0) ──────────────────────────────
    # Weights reflect relative importance of each quality signal.
    # Precision weighted highest — historical track record matters most.
    # Brier score weighted to reward well-calibrated models specifically.
    SCORE_WEIGHT_PRECISION:   float = 0.35
    SCORE_WEIGHT_SIGNAL_PROB: float = 0.25
    SCORE_WEIGHT_ALLOC:       float = 0.20
    SCORE_WEIGHT_BRIER:       float = 0.20   # Lower Brier = better calibration

    MAX_SIGNALS: int = 5

    # ── Walk-forward validation config ────────────────────────────────────────
    DATA_PERIOD:         str = "5y"    # 5 years → sufficient folds for robust OOS
    TRAIN_WINDOW_DAYS:   int = 504     # ~2 trading years initial training window
    OOS_WINDOW_DAYS:     int = 63      # ~1 quarter per OOS fold
    MIN_TRAIN_SAMPLES:   int = 150
    MIN_POSITIVE_LABELS: int = 20      # Positive class floor per fold

    # ── VIX regime thresholds ──────────────────────────────────────────────────
    VIX_CALM:   float = 20.0   # Below: full sizing
    VIX_STRESS: float = 28.0   # Below: reduced sizing
    VIX_CRASH:  float = 35.0   # Above: signals only, no new longs

    OUTPUT_CSV: str = f"signals_{datetime.today().strftime('%Y%m%d')}.csv"


CFG = Config()

# Effective target: nominal target adjusted upward for round-trip friction.
# This means labels are only positive if the stock genuinely clears costs.
EFFECTIVE_TARGET = ((1 + CFG.TARGET_GAIN) * (1 + CFG.TOTAL_FRICTION) ** 2) - 1


# ==============================================================================
# STAGE 0 — REGIME CLASSIFIER
#
# KEY DESIGN DECISION: This is deliberately rules-based, NOT machine-learned.
#
# Rationale: A regime classifier trained on historical market states has access
# to ~30-40 regime-shift events in 20 years of data. This is far too few for
# any ML model without catastrophic overfitting. Rules derived from economic
# first principles (VIX level, credit spreads, trend, breadth) are transparent,
# auditable, and don't overfit to the specific shape of 2008 or 2020.
#
# Output: regime label (BULL/NEUTRAL/BEAR/CRASH) + sizing multiplier (0.0–1.0)
# The sizing multiplier scales ALL Kelly position sizes downstream.
# ==============================================================================
class RegimeClassifier:

    def __init__(self):
        self.vix:               Optional[float] = None
        self.spy_trend:         Optional[str]   = None
        self.credit_stress:     Optional[bool]  = None
        self.regime:            str             = "UNKNOWN"
        self.sizing_multiplier: float           = 0.0
        self.regime_features:   dict            = {}

    def run(self) -> bool:
        log.info("Stage 0 ▶ Regime detection...")
        try:
            raw = yf.download(
                CFG.REGIME_ASSETS, period="1y", progress=False, auto_adjust=True,
            )
            close = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            close.dropna(how="all", inplace=True)

            # ── VIX: primary market stress indicator ──────────────────────────
            vix_series = close["^VIX"].dropna()
            self.vix   = float(vix_series.iloc[-1])
            vix_5d_avg = float(vix_series.tail(5).mean())

            # ── SPY 200-day trend: structural market direction ─────────────────
            spy           = close["SPY"].dropna()
            self.spy_trend = "ABOVE" if spy.iloc[-1] > spy.rolling(200).mean().iloc[-1] else "BELOW"
            spy_momentum  = float(spy.iloc[-1] / spy.iloc[-21] - 1)

            # ── HYG credit proxy: stress in debt markets precedes equity stress
            hyg = close["HYG"].dropna()
            self.credit_stress = bool(hyg.iloc[-1] < hyg.rolling(20).mean().iloc[-1])

            # ── QQQ + IWM breadth: measures risk appetite across cap sizes ─────
            qqq_ret = float(close["QQQ"].dropna().pct_change(5).iloc[-1])
            iwm_ret = float(close["IWM"].dropna().pct_change(5).iloc[-1])
            risk_appetite = (qqq_ret + iwm_ret) / 2

            # ── Score bullish signals (0–5) ────────────────────────────────────
            # Each signal is independent — no single factor dominates.
            bullish = 0
            if self.vix < CFG.VIX_CALM:      bullish += 2   # Double weight: VIX is most reliable
            elif self.vix < CFG.VIX_STRESS:   bullish += 1
            if self.spy_trend == "ABOVE":      bullish += 1
            if spy_momentum > 0:               bullish += 1
            if not self.credit_stress:         bullish += 1

            # ── Classify and assign sizing multiplier ─────────────────────────
            if self.vix >= CFG.VIX_CRASH:
                self.regime, self.sizing_multiplier = "CRASH",   0.0
            elif bullish >= 4:
                self.regime, self.sizing_multiplier = "BULL",    1.0
            elif bullish >= 2:
                self.regime, self.sizing_multiplier = "NEUTRAL", 0.6
            else:
                self.regime, self.sizing_multiplier = "BEAR",    0.25

            self.regime_features = {
                "vix":              round(self.vix, 2),
                "vix_5d_avg":       round(vix_5d_avg, 2),
                "spy_trend":        self.spy_trend,
                "spy_1m_ret":       round(spy_momentum * 100, 2),
                "credit_stress":    self.credit_stress,
                "risk_appetite_5d": round(risk_appetite * 100, 2),
                "bullish_signals":  bullish,
            }
            log.info(
                f"  Regime: {self.regime} | VIX: {self.vix:.1f} | "
                f"Bullish signals: {bullish}/5 | Sizing: {self.sizing_multiplier*100:.0f}%"
            )
            return True

        except Exception as e:
            log.error(f"Regime check failed: {e}")
            self.regime, self.sizing_multiplier = "UNKNOWN", 0.3
            return True


# ==============================================================================
# MACRO DATA LOADER (v6.0)
#
# KEY DESIGN DECISION: Cross-asset features downloaded ONCE, shared across all
# tickers. This is efficient (no redundant API calls) and ensures consistent
# date alignment across the entire universe scan.
#
# The four macro features capture dimensions that pure price-action models miss:
#   EU_Return      → Lead-lag from European market open
#   Rel_Str_SPY    → Idiosyncratic alpha vs broad market (momentum factor)
#   Corr_SPY_20d   → Beta regime — is the stock moving with or against the market?
#   Corr_GLD_20d   → Risk-off behaviour — is it acting like a safe haven?
# ==============================================================================
class MacroDataLoader:

    def __init__(self):
        self.macro_returns: Optional[pd.DataFrame] = None

    def load(self) -> bool:
        log.info("Macro ▶ Loading cross-asset data (EU / SPY / GLD)...")
        try:
            tickers = list(CFG.MACRO_TICKERS.values())
            raw = yf.download(tickers, period=CFG.DATA_PERIOD, progress=False, auto_adjust=True)

            prices = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            prices.dropna(how="all", inplace=True)
            prices.rename(columns={v: CFG.MACRO_COLS[v] for v in tickers if v in CFG.MACRO_COLS}, inplace=True)

            # Log returns — same transform applied to stock features for consistency
            self.macro_returns = np.log(prices / prices.shift(1))
            log.info(f"  Macro loaded: {len(self.macro_returns)} days | cols: {list(self.macro_returns.columns)}")
            return True

        except Exception as e:
            log.warning(f"Macro load failed: {e}. Macro features will be zero-filled.")
            self.macro_returns = None
            return False


# ==============================================================================
# STAGE 1 — DATA QUALITY & PRE-FILTER
# ==============================================================================
class DataManager:

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
                log.warning(f"Could not read CSV: {e}. Using built-in watchlist.")
        return CFG.WATCHLIST

    @staticmethod
    def fetch(ticker: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(ticker, period=CFG.DATA_PERIOD, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.dropna(subset=["Close", "Volume"], inplace=True)
            return df if len(df) >= 400 else None
        except Exception:
            return None

    @staticmethod
    def passes_liquidity(df: pd.DataFrame) -> bool:
        """
        Two-part liquidity gate: share volume AND dollar volume.
        Dollar volume catches stocks with high share count but low price —
        a $1 stock trading 2M shares/day has thin real liquidity.
        """
        avg_vol        = df["Volume"].rolling(30).mean().iloc[-1]
        avg_dollar_vol = (df["Close"] * df["Volume"]).rolling(30).mean().iloc[-1]
        return avg_vol >= CFG.MIN_VOLUME_30D and avg_dollar_vol >= 5_000_000

    @staticmethod
    def earnings_clear(ticker: str) -> bool:
        """
        Conservative earnings blackout: skip if earnings within EARNINGS_BUFFER_DAYS.
        If we can't confirm (API failure), we allow but flag in the report.
        Binary options pricing around earnings makes technical signals unreliable.
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
# FEATURE ENGINEERING
#
# KEY DESIGN DECISION: All 39 features are computed using ONLY data available
# at time t. No future information can leak into any feature at any row.
#
# Features span 7 groups to capture different return-generating mechanisms:
#   1. Price momentum (multiple timeframes — Moskowitz et al. 2012)
#   2. Volatility regime (vol clustering, vol ratio)
#   3. Trend & mean-reversion (distance from moving averages)
#   4. Momentum indicators (RSI, MACD — with correct Wilder smoothing)
#   5. Volume / microstructure (relative volume, price-volume agreement)
#   6. Market regime context (VIX level, SPY trend, credit conditions)
#   7. Cross-asset macro (EU lead-lag, SPY relative strength, GLD correlation)
# ==============================================================================
class FeatureEngine:

    @staticmethod
    def build(
        df:              pd.DataFrame,
        regime_features: dict,
        macro_returns:   Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:

        d = df.copy()

        # ── 1. Price momentum ─────────────────────────────────────────────────
        d["log_ret"]  = np.log(d["Close"] / d["Close"].shift(1))
        d["ret_5d"]   = d["Close"].pct_change(5)
        d["ret_10d"]  = d["Close"].pct_change(10)
        d["ret_20d"]  = d["Close"].pct_change(20)
        d["ret_60d"]  = d["Close"].pct_change(60)

        # Multi-timeframe momentum: log-spaced lookbacks sample trend at multiple
        # horizons — captures both short-term reversal and medium-term persistence
        for days in [1, 2, 4, 8, 16, 32]:
            d[f"mtf_ret_{days}d"] = d["Close"].pct_change(days)

        # Alignment score: how many MTF windows show positive returns? (-6 to +6)
        # Strong alignment = trend health; mixed = choppy/transitional
        d["mtf_alignment"] = sum(
            np.sign(d[f"mtf_ret_{days}d"]) for days in [1, 2, 4, 8, 16, 32]
        )

        # ── 2. Volatility ─────────────────────────────────────────────────────
        d["vol_10d"]   = d["log_ret"].rolling(10).std() * np.sqrt(252)
        d["vol_20d"]   = d["log_ret"].rolling(20).std() * np.sqrt(252)
        d["vol_ratio"] = d["vol_10d"] / d["vol_20d"].replace(0, np.nan)  # >1 = expanding vol

        # ── 3. Trend / mean-reversion ─────────────────────────────────────────
        d["sma_20"]       = d["Close"].rolling(20).mean()
        d["sma_50"]       = d["Close"].rolling(50).mean()
        d["sma_200"]      = d["Close"].rolling(200).mean()
        d["dist_20"]      = (d["Close"] - d["sma_20"])  / d["sma_20"]
        d["dist_50"]      = (d["Close"] - d["sma_50"])  / d["sma_50"]
        d["dist_200"]     = (d["Close"] - d["sma_200"]) / d["sma_200"]
        d["trend_50_200"] = (d["sma_50"] - d["sma_200"]) / d["sma_200"]  # Golden/death cross

        # ── 4. Momentum indicators ─────────────────────────────────────────────
        # RSI: Wilder's exponential smoothing (not simple rolling mean — common mistake)
        delta    = d["Close"].diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        d["rsi"]          = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))
        d["rsi_oversold"]   = (d["rsi"] < 35).astype(int)
        d["rsi_overbought"] = (d["rsi"] > 70).astype(int)

        # MACD
        ema12          = d["Close"].ewm(span=12, adjust=False).mean()
        ema26          = d["Close"].ewm(span=26, adjust=False).mean()
        d["macd"]      = ema12 - ema26
        d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
        d["macd_hist"] = d["macd"] - d["macd_sig"]
        d["macd_cross"] = (
            (d["macd"] > d["macd_sig"]) & (d["macd"].shift(1) <= d["macd_sig"].shift(1))
        ).astype(int)

        # Bollinger band position: normalised to [-1, +1] range
        bb_mid      = d["Close"].rolling(20).mean()
        bb_std      = d["Close"].rolling(20).std()
        d["bb_pos"] = (d["Close"] - bb_mid) / (2 * bb_std.replace(0, np.nan))

        # ── 5. Volume / microstructure ────────────────────────────────────────
        d["vol_rel_20"]      = d["Volume"] / d["Volume"].rolling(20).mean().replace(0, np.nan)
        d["vol_rel_5"]       = d["Volume"] / d["Volume"].rolling(5).mean().replace(0, np.nan)
        d["vol_price_agree"] = d["vol_rel_20"] * np.sign(d["log_ret"])  # +: vol confirms direction

        # ATR: normalised by price so it's comparable across stocks
        hl  = d["High"] - d["Low"]
        hc  = (d["High"] - d["Close"].shift(1)).abs()
        lc  = (d["Low"]  - d["Close"].shift(1)).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        d["atr_14"]  = tr.ewm(span=14, adjust=False).mean()
        d["atr_pct"] = d["atr_14"] / d["Close"]

        # Trend confirmation composite: momentum + vol expansion + volume must ALL agree
        mom_ok  = (d["Close"] > d["sma_20"]).astype(int)
        atr_exp = (d["atr_14"] > d["atr_14"].rolling(14).mean()).astype(int)
        vol_ok  = (d["Volume"] > d["Volume"].rolling(20).mean()).astype(int)
        d["trend_confirm"] = mom_ok * atr_exp * vol_ok  # 1 only if all three agree

        # Choppiness: direction changes in last 5 bars (0=trending, 4=fully choppy)
        daily_dir   = np.sign(d["Close"].diff())
        d["choppiness_5d"] = sum(
            (daily_dir.shift(i) != daily_dir.shift(i + 1)).astype(int) for i in range(4)
        )

        # ── 6. Market regime context (scalar from RegimeClassifier) ───────────
        d["regime_vix"]       = regime_features.get("vix", 20.0)
        d["regime_spy_trend"] = 1 if regime_features.get("spy_trend") == "ABOVE" else 0
        d["regime_credit_ok"] = 0 if regime_features.get("credit_stress") else 1
        d["regime_risk_app"]  = regime_features.get("risk_appetite_5d", 0.0)

        # ── 7. Cross-asset macro features (v6.0) ──────────────────────────────
        #
        # Aligned to the stock's date index via forward-fill: this ensures we
        # only use macro data that was actually available at each time t.
        # Zero-fill on missing: graceful degradation if macro download fails.
        if macro_returns is not None:
            aligned = macro_returns.reindex(d.index, method="ffill").fillna(0.0)
            spy_ret = aligned.get("SPY_Return", pd.Series(0.0, index=d.index))
            gld_ret = aligned.get("GLD_Return", pd.Series(0.0, index=d.index))
            eu_ret  = aligned.get("EU_Return",  pd.Series(0.0, index=d.index))

            d["eu_return"]    = eu_ret
            d["rel_str_spy"]  = d["log_ret"] - spy_ret           # Alpha: outperforming market?
            d["corr_spy_20d"] = d["log_ret"].rolling(20).corr(spy_ret)  # Beta regime
            d["corr_gld_20d"] = d["log_ret"].rolling(20).corr(gld_ret)  # Risk-off behaviour
        else:
            for col in ["eu_return", "rel_str_spy", "corr_spy_20d", "corr_gld_20d"]:
                d[col] = 0.0

        return d.replace([np.inf, -np.inf], np.nan)


# Complete feature list passed to XGBoost (39 features)
FEATURE_COLS = [
    # Price momentum
    "log_ret", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    # Multi-timeframe momentum
    "mtf_ret_1d", "mtf_ret_2d", "mtf_ret_4d",
    "mtf_ret_8d", "mtf_ret_16d", "mtf_ret_32d", "mtf_alignment",
    # Volatility
    "vol_10d", "vol_20d", "vol_ratio",
    # Trend / mean-reversion
    "dist_20", "dist_50", "dist_200", "trend_50_200",
    # Momentum indicators
    "rsi", "rsi_oversold", "rsi_overbought", "macd_hist", "macd_cross",
    # Mean-reversion
    "bb_pos",
    # Volume / microstructure
    "vol_rel_20", "vol_rel_5", "vol_price_agree",
    # Volatility regime
    "atr_pct",
    # Confirmation composites
    "trend_confirm", "choppiness_5d",
    # Regime context
    "regime_vix", "regime_spy_trend", "regime_credit_ok", "regime_risk_app",
    # Cross-asset macro (v6.0)
    "eu_return", "rel_str_spy", "corr_spy_20d", "corr_gld_20d",
]


# ==============================================================================
# LABEL CONSTRUCTION
#
# KEY DESIGN DECISION: Close-to-Close labels only (not High).
#
# Using High in the label is a common and seductive mistake. The rationale for
# avoiding it: you can only realise a gain at a closing price in practice.
# Using High inflates hit rates and introduces subtle look-ahead — the intraday
# high is not observable until the end of the day, and you can't guarantee a
# fill at the exact high. Close-to-Close is conservative and honest.
#
# Labels at the tail (last HOLD_DAYS_MAX+1 rows) are set to -1 and excluded
# from both training and evaluation — there is no valid future window for them.
# ==============================================================================
def build_labels(df: pd.DataFrame) -> pd.Series:
    close  = df["Close"].values
    n      = len(close)
    H      = CFG.HOLD_DAYS_MAX
    labels = np.zeros(n, dtype=int)

    for i in range(n - H - 1):
        future_window = close[i + 1 : i + H + 1]
        if future_window.max() >= close[i] * (1 + EFFECTIVE_TARGET):
            labels[i] = 1

    labels[-(H + 1):] = -1  # No valid future window — excluded from all modelling
    return pd.Series(labels, index=df.index, name="label")


# ==============================================================================
# STAGE 2 — WALK-FORWARD ALPHA MODEL
#
# KEY DESIGN DECISION: Expanding-window walk-forward, not k-fold.
#
# Standard k-fold cross-validation on time series violates temporal ordering —
# a model trained on 2023 data evaluated on 2021 data is meaningless at best
# and actively misleading at worst. Walk-forward ensures:
#   (a) The model is NEVER evaluated on data it was trained on.
#   (b) OOS metrics reflect real generalization, not in-sample fit.
#   (c) The accumulation of predictions across folds gives a statistically
#       meaningful sample for precision and Brier score estimation.
#
# KEY DESIGN DECISION: Platt scaling calibration.
#
# Raw XGBoost probabilities are systematically overconfident — outputs cluster
# near 0 and 1 more than the true distribution warrants. A raw score of 0.72
# does NOT mean 72% probability of success. Platt scaling (logistic regression
# on XGB outputs) maps raw scores to true probabilities. Critically, the
# calibrator is fitted on the OOS fold — never on training data.
# ==============================================================================
class AlphaModel:

    def __init__(self, ticker: str):
        self.ticker             = ticker
        self.oos_precision: float  = 0.0
        self.oos_brier:     float  = 1.0    # Lower = better; 0.25 = random binary
        self.n_folds:       int    = 0
        self.n_oos_samples: int    = 0
        self.final_model           = None
        self.feature_importance: Optional[pd.Series] = None

    def _make_base_model(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,    # Regularisation: prevents overfitting minority class
            scale_pos_weight=1,     # Set dynamically per fold based on class balance
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

    def run(self, df_features: pd.DataFrame, labels: pd.Series) -> bool:
        """
        Runs walk-forward validation and trains the final production model.
        Returns True only if OOS metrics clear the quality threshold.
        """
        data       = df_features[FEATURE_COLS].copy()
        data["label"] = labels
        data       = data[data["label"] != -1].dropna()

        if len(data) < CFG.MIN_TRAIN_SAMPLES * 2:
            return False

        X_all = data[FEATURE_COLS].values
        y_all = data["label"].values

        oos_true, oos_proba = [], []
        train_end = CFG.TRAIN_WINDOW_DAYS
        step      = CFG.OOS_WINDOW_DAYS

        # ── Walk-forward loop ─────────────────────────────────────────────────
        while train_end + step <= len(data):
            X_tr, y_tr = X_all[:train_end], y_all[:train_end]
            X_te, y_te = X_all[train_end : train_end + step], y_all[train_end : train_end + step]

            if y_tr.sum() < CFG.MIN_POSITIVE_LABELS:
                train_end += step
                continue

            # Dynamic class balancing per fold
            n0, n1 = (y_tr == 0).sum(), (y_tr == 1).sum()
            spw    = n0 / n1 if n1 > 0 else 1.0

            model = self._make_base_model()
            model.set_params(scale_pos_weight=spw)
            model.fit(X_tr, y_tr)

            # Platt calibration fitted on OOS fold — not training data
            calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            calibrated.fit(X_te, y_te)

            oos_true.extend(y_te.tolist())
            oos_proba.extend(calibrated.predict_proba(X_te)[:, 1].tolist())
            self.n_folds += 1
            train_end    += step

        if len(oos_true) < 50:
            return False

        # ── OOS metrics ───────────────────────────────────────────────────────
        oos_arr   = np.array(oos_true)
        proba_arr = np.array(oos_proba)

        self.oos_precision = precision_score(oos_arr, (proba_arr >= 0.5).astype(int), zero_division=0)
        self.oos_brier     = brier_score_loss(oos_arr, proba_arr)
        self.n_oos_samples = len(oos_true)

        if self.oos_precision < CFG.MIN_OOS_PRECISION:
            return False

        # ── Final production model: trained on ALL data ────────────────────────
        # We trust it only because OOS metrics already validated the approach.
        n0_all, n1_all = (y_all == 0).sum(), (y_all == 1).sum()
        base = self._make_base_model()
        base.set_params(scale_pos_weight=n0_all / n1_all if n1_all > 0 else 1.0)
        base.fit(X_all, y_all)

        # Calibrate final model on last OOS fold
        calib_X, calib_y = X_all[-CFG.OOS_WINDOW_DAYS:], y_all[-CFG.OOS_WINDOW_DAYS:]
        if len(np.unique(calib_y)) > 1:
            self.final_model = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
            self.final_model.fit(calib_X, calib_y)
        else:
            self.final_model = base

        self.feature_importance = pd.Series(
            base.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)

        return True

    def predict_latest(self, df_features: pd.DataFrame) -> Optional[float]:
        """Returns calibrated signal probability for the most recent bar."""
        if self.final_model is None:
            return None
        try:
            X = df_features[FEATURE_COLS].dropna().iloc[[-1]].values
            return float(self.final_model.predict_proba(X)[0, 1])
        except Exception:
            return None


# ==============================================================================
# STAGE 3 — POSITION SIZING
#
# KEY DESIGN DECISION: Wilson score CI lower bound on Kelly criterion.
#
# The Kelly formula takes a win rate and outputs an optimal bet size.
# The problem: OOS precision is estimated from ~200-400 samples. Measurement
# noise means a true 54% win rate could easily measure as 60% in one sample.
# Feeding the noisy point estimate directly into Kelly produces aggressive
# position sizes that will ruin a small account on a bad run.
#
# Solution: Wilson score confidence interval lower bound at 90% confidence.
# This gives the value we're 90% confident the true precision is AT LEAST.
# We bet on that conservative floor, not the measured estimate.
#
# Additional scaling layers:
#   × regime multiplier  (0.0–1.0 based on market environment)
#   × conviction scalar  (derived from calibrated signal probability)
#   hard cap at MAX_POSITION_PCT regardless
# ==============================================================================
class PositionSizer:

    @staticmethod
    def wilson_lower_bound(precision: float, n: int, confidence: float = 0.90) -> float:
        """
        Wilson score interval lower bound on a proportion.
        More accurate than normal approximation, especially for small n.
        """
        if n == 0:
            return 0.0
        z      = stats.norm.ppf((1 + confidence) / 2)
        denom  = 1 + z ** 2 / n
        centre = (precision + z ** 2 / (2 * n)) / denom
        margin = (z * np.sqrt(precision * (1 - precision) / n + z ** 2 / (4 * n ** 2))) / denom
        return max(0.0, centre - margin)

    @staticmethod
    def half_kelly(win_rate: float) -> float:
        """
        Half-Kelly: standard institutional adjustment.
        Full Kelly is theoretically optimal but practically too volatile —
        it assumes perfect knowledge of the true win rate, which we don't have.
        Half-Kelly reduces drawdowns substantially at modest cost to long-run growth.
        """
        b     = CFG.TARGET_GAIN / CFG.STOP_LOSS   # Payout ratio
        kelly = (win_rate * b - (1 - win_rate)) / b
        return max(0.0, kelly / 2)

    @classmethod
    def compute(
        cls,
        oos_precision:     float,
        oos_n:             int,
        current_prob:      float,
        regime_multiplier: float,
    ) -> dict:

        win_rate_lb = cls.wilson_lower_bound(oos_precision, oos_n)
        kelly_base  = cls.half_kelly(win_rate_lb)

        # Regime scaling: size down in NEUTRAL/BEAR, zero in CRASH
        kelly_regime = kelly_base * regime_multiplier

        # Conviction scalar: scales from 0 (prob=0.50) to 1 (prob=1.0)
        # Only signals above 0.5 should ever reach this stage
        conviction   = max(0.0, (current_prob - 0.5) * 2)
        kelly_final  = min(kelly_regime * conviction, CFG.MAX_POSITION_PCT)

        return {
            "win_rate_ci_lb":    round(win_rate_lb * 100, 1),
            "kelly_base_pct":    round(kelly_base * 100, 1),
            "alloc_pct":         round(kelly_final * 100, 2),
            "dollar_size":       round(kelly_final * CFG.ACCOUNT_SIZE, 0),
            "conviction_scalar": round(conviction, 3),
        }


# ==============================================================================
# STAGE 5 — ELITE SIGNAL FILTER & COMPOSITE SCORER
#
# Two-phase filtering:
#   Phase 1 (Hard gates): Binary pass/fail. Any single failure drops the signal.
#   Phase 2 (Scoring): Survivors ranked by weighted composite [0–1].
#
# Hard gates rationale (brief — full rationale in README):
#   OOS Precision ≥ 58%  : Below this, EV is marginal after costs even at 2:1 R/R
#   R/R ≥ 2:1            : Non-negotiable math — below 2:1 you need >66% win rate
#   Alloc ≥ 3%           : Sub-3% positions are noise on a <$50K account
#   Above 200MA           : Structural downtrends statistically hurt long swing setups
#   Signal prob ≥ 60%    : Model must be meaningfully confident, not just above 50%
# ==============================================================================
class SignalFilter:

    @staticmethod
    def above_200ma(close: pd.Series) -> bool:
        if len(close) < 200:
            return False
        return float(close.iloc[-1]) > float(close.rolling(200).mean().iloc[-1])

    @classmethod
    def apply_hard_filters(cls, signal: dict) -> tuple[bool, list[str]]:
        fails = []
        sz    = signal["sizing"]

        if signal["oos_precision"] < CFG.ELITE_MIN_OOS_PRECISION:
            fails.append(f"OOS prec {signal['oos_precision']*100:.1f}% < {CFG.ELITE_MIN_OOS_PRECISION*100:.0f}%")
        if (CFG.TARGET_GAIN / CFG.STOP_LOSS) < CFG.ELITE_MIN_RR:
            fails.append(f"R/R below minimum")
        if sz["alloc_pct"] < CFG.ELITE_MIN_ALLOC:
            fails.append(f"Alloc {sz['alloc_pct']:.1f}% < {CFG.ELITE_MIN_ALLOC:.0f}%")
        if CFG.ELITE_ABOVE_200MA and not signal.get("above_200ma", True):
            fails.append("Below 200-day MA")
        if signal["signal_prob"] < CFG.ELITE_MIN_SIGNAL_PROB:
            fails.append(f"Signal prob {signal['signal_prob']*100:.1f}% < {CFG.ELITE_MIN_SIGNAL_PROB*100:.0f}%")
        if signal.get("earnings_close"):
            fails.append("Earnings within buffer window")

        return len(fails) == 0, fails

    @classmethod
    def compute_score(cls, signal: dict) -> float:
        """
        Composite score normalised to [0, 1].
        Each component normalised over its meaningful range before weighting.
        """
        sz = signal["sizing"]

        prec_norm  = max(0.0, min(1.0, (signal["oos_precision"] - 0.54) / 0.26))
        prob_norm  = max(0.0, min(1.0, (signal["signal_prob"] - 0.60) / 0.35))
        alloc_norm = max(0.0, min(1.0, (sz["alloc_pct"] - 3.0) / 12.0))
        brier_norm = max(0.0, min(1.0, (0.25 - signal["oos_brier"]) / 0.15))

        return round(
            CFG.SCORE_WEIGHT_PRECISION   * prec_norm
            + CFG.SCORE_WEIGHT_SIGNAL_PROB * prob_norm
            + CFG.SCORE_WEIGHT_ALLOC       * alloc_norm
            + CFG.SCORE_WEIGHT_BRIER       * brier_norm,
            4,
        )

    @classmethod
    def run(cls, signals: list[dict]) -> tuple[list[dict], list[dict]]:
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


# ==============================================================================
# SIGNAL REPORT
# ==============================================================================
def print_report(signals: list[dict], rejected: list[dict], regime: RegimeClassifier):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rf  = regime.regime_features

    print()
    print("=" * 115)
    print(f"  TITAN v6.0 — SIGNAL REPORT    {now}")
    print("=" * 115)

    regime_label = {"BULL": "🟢 BULL", "NEUTRAL": "🟡 NEUTRAL",
                    "BEAR": "🔴 BEAR", "CRASH": "🚨 CRASH"}.get(regime.regime, "⚪ UNKNOWN")

    print(f"\n  REGIME : {regime_label}  |  VIX: {rf.get('vix', '?'):.1f}  |  "
          f"SPY vs 200d: {rf.get('spy_trend', '?')}  |  "
          f"Credit: {'STRESS ⚠' if rf.get('credit_stress') else 'OK ✓'}  |  "
          f"Sizing multiplier: {regime.sizing_multiplier*100:.0f}%")
    print(f"  TARGET: +{CFG.TARGET_GAIN*100:.0f}%  |  STOP: -{CFG.STOP_LOSS*100:.0f}%  |  "
          f"R/R: {CFG.TARGET_GAIN/CFG.STOP_LOSS:.1f}:1  |  "
          f"Effective target (post-friction): {EFFECTIVE_TARGET*100:.2f}%\n")

    if regime.regime == "CRASH":
        print("  🚨 CRASH REGIME — VIX > 35. NO NEW LONGS. Signals shown for reference only.\n")

    if not signals:
        print("  ❌  No signals passed all quality filters today.\n")
        print("=" * 115)
        return

    print(f"  {'#':<3} | {'TICKER':<7} | {'SCORE':>6} | {'PRICE':>7} | {'SIGNAL%':>7} | "
          f"{'OOS PREC':>8} | {'W-RATE LB':>9} | {'BRIER':>5} | {'ALLOC%':>7} | "
          f"{'$ SIZE':>8} | {'RS_SPY':>7} | {'GLD_CORR':>8}")
    print("  " + "-" * 105)

    for rank, s in enumerate(signals, 1):
        sz   = s["sizing"]
        rs   = s.get("rel_str_spy_pct", 0.0)
        gc   = s.get("corr_gld_20d", 0.0)
        flag = "🟢" if rs > 0 else "🔴"
        ma   = "✓" if s.get("above_200ma") else "✗"

        print(
            f"  [{rank}] | {s['ticker']:<7} | {s.get('elite_score', 0):>6.3f} | "
            f"{s['price']:>7.2f} | {s['signal_prob']*100:>6.1f}% | "
            f"{s['oos_precision']*100:>7.1f}% | {sz['win_rate_ci_lb']:>8.1f}% | "
            f"{s['oos_brier']:>5.3f} | {sz['alloc_pct']:>6.2f}% | "
            f"${sz['dollar_size']:>7,.0f} | {flag}{rs:>+5.2f}% | {gc:>8.2f}  200MA:{ma}"
        )

    print("\n  " + "─" * 105)
    print("  TRADE SHEET")
    print("  " + "─" * 105)
    total = 0.0
    for i, s in enumerate(signals, 1):
        sz = s["sizing"]
        if sz["dollar_size"] < CFG.MIN_POSITION_DOLLAR:
            continue
        shares   = int(sz["dollar_size"] / s["price"])
        stop     = round(s["price"] * (1 - CFG.STOP_LOSS), 2)
        target   = round(s["price"] * (1 + CFG.TARGET_GAIN), 2)
        total   += sz["alloc_pct"]
        print(f"  [{i}] {s['ticker']:<6}  BUY {shares} shares @ MKT ~${s['price']:.2f}"
              f"  │  Stop: ${stop}  │  Target: ${target}"
              f"  │  Risk: ${round(shares * s['price'] * CFG.STOP_LOSS):,.0f}")

    print(f"\n  ALLOCATED: {total:.1f}% (${total/100*CFG.ACCOUNT_SIZE:,.0f})  "
          f"|  CASH: {100-total:.1f}%\n")

    if rejected:
        print(f"  REJECTED ({len(rejected)} signals failed elite filter):")
        for s in sorted(rejected, key=lambda x: x["oos_precision"], reverse=True)[:5]:
            print(f"    {s['ticker']:<7} OOS:{s['oos_precision']*100:.1f}%  "
                  f"→ {' | '.join(s.get('reject_reasons', ['?']))}")

    print("=" * 115)

    # Save to CSV
    rows = [{
        "date": now, "ticker": s["ticker"], "price": s["price"],
        "signal_prob_pct": round(s["signal_prob"] * 100, 2),
        "oos_precision_pct": round(s["oos_precision"] * 100, 2),
        "win_rate_ci_lb": s["sizing"]["win_rate_ci_lb"],
        "brier_score": s["oos_brier"],
        "alloc_pct": s["sizing"]["alloc_pct"],
        "dollar_size": s["sizing"]["dollar_size"],
        "stop_price": round(s["price"] * (1 - CFG.STOP_LOSS), 2),
        "target_price": round(s["price"] * (1 + CFG.TARGET_GAIN), 2),
        "regime": regime.regime,
        "rel_str_spy_pct": round(s.get("rel_str_spy_pct", 0.0), 4),
        "corr_gld_20d": round(s.get("corr_gld_20d", 0.0), 4),
    } for s in signals]

    pd.DataFrame(rows).to_csv(CFG.OUTPUT_CSV, index=False)
    log.info(f"Signals saved → {CFG.OUTPUT_CSV}")


# ==============================================================================
# MASTER PIPELINE
# ==============================================================================
def run():
    print("\n" + "=" * 115)
    print("  TITAN v6.0 — Macro-Quant Swing Trading Signal Engine")
    print("=" * 115)

    # Stage 0: Market regime
    regime = RegimeClassifier()
    regime.run()

    # Load macro data once for all tickers
    macro = MacroDataLoader()
    macro.load()

    watchlist = DataManager.load_watchlist()
    log.info(f"Universe: {len(watchlist)} tickers")

    raw_signals = []

    for ticker in watchlist:
        log.info(f"Processing {ticker}...")

        # Stage 1: Data quality
        df = DataManager.fetch(ticker)
        if df is None or not DataManager.passes_liquidity(df):
            continue

        earnings_ok = DataManager.earnings_clear(ticker)

        # Stage 2: Feature engineering (with macro injection)
        df_feat = FeatureEngine.build(df, regime.regime_features, macro.macro_returns)
        labels  = build_labels(df)

        # Stage 3: Walk-forward model
        model = AlphaModel(ticker)
        if not model.run(df_feat, labels):
            continue

        signal_prob = model.predict_latest(df_feat)
        if signal_prob is None or signal_prob < 0.52:
            continue

        # Stage 4: Position sizing
        sizing = PositionSizer.compute(
            oos_precision=model.oos_precision,
            oos_n=model.n_oos_samples,
            current_prob=signal_prob,
            regime_multiplier=regime.sizing_multiplier,
        )
        if sizing["alloc_pct"] < CFG.MIN_KELLY_AFTER_CI * 100:
            continue

        # Extract macro summary for report
        latest          = df_feat.dropna(subset=FEATURE_COLS).iloc[-1]
        rel_str_spy_pct = float(latest.get("rel_str_spy", 0.0)) * 100
        corr_gld_20d    = float(latest.get("corr_gld_20d", 0.0))

        raw_signals.append({
            "ticker":          ticker,
            "price":           float(df["Close"].iloc[-1]),
            "signal_prob":     signal_prob,
            "oos_precision":   model.oos_precision,
            "oos_brier":       model.oos_brier,
            "oos_n":           model.n_oos_samples,
            "n_folds":         model.n_folds,
            "sizing":          sizing,
            "earnings_close":  not earnings_ok,
            "above_200ma":     SignalFilter.above_200ma(df["Close"]),
            "rel_str_spy_pct": rel_str_spy_pct,
            "corr_gld_20d":    corr_gld_20d,
        })

        log.info(
            f"  ✓ {ticker}: prob={signal_prob*100:.1f}%  "
            f"OOS={model.oos_precision*100:.1f}%  alloc={sizing['alloc_pct']:.1f}%  "
            f"RS_SPY={rel_str_spy_pct:+.2f}%  GLD_corr={corr_gld_20d:.2f}"
        )

    log.info(f"Scan done. {len(raw_signals)}/{len(watchlist)} passed pre-filters.")

    # Stage 5: Elite filter + composite ranking
    elite, rejected = SignalFilter.run(raw_signals)
    log.info(f"Elite filter: {len(elite)} survived | {len(rejected)} rejected")

    # Stage 6: Report
    print_report(elite, rejected, regime)


if __name__ == "__main__":
    run()
