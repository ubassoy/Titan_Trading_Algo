"""
config.py — Central configuration for TITAN v7.1
All tuneable parameters live here. Edit this file to change behaviour.
"""

import logging
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TITAN")


class Config:
    # ── Universe ───────────────────────────────────────────────────────────────
    WATCHLIST: list[str] = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD",
        "JPM", "BAC", "GS", "V", "MA", "XOM", "CVX", "LLY", "UNH",
        "HD", "COST", "PLTR", "COIN", "MSTR", "SMCI", "MU", "CRM",
    ]
    WATCHLIST_CSV: str = "winning_stocks.csv"

    # ── Regime detection assets ────────────────────────────────────────────────
    REGIME_ASSETS: list[str] = [
        "SPY", "QQQ", "IWM", "TLT", "GLD", "UUP", "^VIX", "HYG",
    ]

    # ── Cross-asset macro tickers ──────────────────────────────────────────────
    MACRO_TICKERS: dict[str, str] = {
        "EU": "^STOXX50E", "SPY": "SPY", "GLD": "GLD",
    }
    MACRO_COLS: dict[str, str] = {
        "^STOXX50E": "EU_Return", "SPY": "SPY_Return", "GLD": "GLD_Return",
    }

    # ── Trade parameters ───────────────────────────────────────────────────────
    HOLD_DAYS_MIN:  int   = 5
    HOLD_DAYS_MAX:  int   = 20
    TARGET_GAIN:    float = 0.08    # [PARAM] — calibrate to your R/R preference
    STOP_LOSS:      float = 0.04    # [PARAM] — calibrate to your risk tolerance
    COMMISSION_BPS: float = 0.0
    SLIPPAGE_BPS:   float = 10.0
    TOTAL_FRICTION: float = (0.0 + 10.0) / 10_000

    # ── Account ────────────────────────────────────────────────────────────────
    ACCOUNT_SIZE:        float = 40_000
    MAX_POSITION_PCT:    float = 0.15
    MAX_POSITIONS:       int   = 6
    MIN_POSITION_DOLLAR: float = 500

    # ── Data quality pre-filters ───────────────────────────────────────────────
    MIN_VOLUME_30D:       int   = 1_500_000
    MIN_OOS_PRECISION:    float = 0.54   # [PARAM] — minimum OOS precision to proceed
    MIN_KELLY_AFTER_CI:   float = 0.01
    EARNINGS_BUFFER_DAYS: int   = 14

    # ── Elite filter hard gates ────────────────────────────────────────────────
    # All five gates must pass for a signal to appear in the output.
    ELITE_MIN_OOS_PRECISION: float = 0.58   # [PARAM]
    ELITE_MIN_RR:            float = 2.0
    ELITE_MIN_ALLOC:         float = 3.0
    ELITE_ABOVE_200MA:       bool  = True
    ELITE_MIN_SIGNAL_PROB:   float = 0.60   # [PARAM]
    ELITE_MIN_HYBRID_SCORE:  int   = 30

    # ── Composite score weights — must sum to 1.0 ──────────────────────────────
    SCORE_WEIGHT_PRECISION:   float = 0.30
    SCORE_WEIGHT_SIGNAL_PROB: float = 0.20
    SCORE_WEIGHT_ALLOC:       float = 0.15
    SCORE_WEIGHT_BRIER:       float = 0.15
    SCORE_WEIGHT_HYBRID:      float = 0.20

    MAX_SIGNALS: int = 5

    # ── Walk-forward model ─────────────────────────────────────────────────────
    DATA_PERIOD:         str = "5y"
    TRAIN_WINDOW_DAYS:   int = 504   # ~2 trading years initial training window
    OOS_WINDOW_DAYS:     int = 63    # ~1 quarter OOS test window
    MIN_TRAIN_SAMPLES:   int = 150
    MIN_POSITIVE_LABELS: int = 20

    # ── VIX regime tiers ───────────────────────────────────────────────────────
    VIX_CALM:   float = 15.0   # Below → low vol bonus in HybridScorer
    VIX_NORMAL: float = 20.0
    VIX_STRESS: float = 28.0
    VIX_CRASH:  float = 35.0   # Above → sizing = 0, no new longs

    OUTPUT_CSV: str = f"signals_{datetime.today().strftime('%Y%m%d')}.csv"

    # ── Backtester ─────────────────────────────────────────────────────────────
    BT_INITIAL_CAPITAL:   float = 100_000
    BT_COMMISSION:        float = 0.001     # 0.1% per side
    BT_ENTRY_SCORE_MIN:   int   = 50        # HybridScorer threshold to enter
    BT_EXIT_SCORE_MIN:    int   = 20        # Exit if score drops below this
    BT_POSITION_SIZE_PCT: float = 0.15      # 15% of capital per position
    BT_ATR_STOP_MULT:     float = 2.0       # Stop = entry - 2×ATR
    BT_PROFIT_TARGET:     float = 0.15      # 15% take-profit
    BT_MAX_HOLD_DAYS:     int   = 60        # Time stop
    BT_REGIME_LOOKBACK:   int   = 252       # Bars needed to warm up regime engine
    BT_OUTPUT_CSV:        str   = f"backtest_{datetime.today().strftime('%Y%m%d')}.csv"

    @staticmethod
    def estimate_slippage(avg_volume: float, avg_price: float) -> float:
        """
        Dynamic slippage model based on dollar volume.
        More realistic than a fixed bps assumption across all stocks.
          > $500M/day  → 2bps   (very liquid large-cap, e.g. AAPL)
          > $100M/day  → 5bps   (liquid mid-cap)
          > $20M/day   → 10bps  (moderate liquidity)
          <= $20M/day  → 20bps  (thin — avoid where possible)
        """
        dollar_volume = avg_volume * avg_price
        if dollar_volume > 500_000_000:   return 2.0
        elif dollar_volume > 100_000_000: return 5.0
        elif dollar_volume > 20_000_000:  return 10.0
        else:                             return 20.0


CFG = Config()
EFFECTIVE_TARGET = ((1 + CFG.TARGET_GAIN) * (1 + CFG.TOTAL_FRICTION) ** 2) - 1
