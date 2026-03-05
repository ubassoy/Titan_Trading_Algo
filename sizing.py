"""
sizing.py — Position Sizing via Wilson CI + Half-Kelly

THE PROBLEM WITH NAIVE KELLY
─────────────────────────────
The Kelly criterion says: bet fraction f = (p×b − (1−p)) / b
where p = win probability, b = win/loss ratio.

If you use raw OOS precision as p (e.g. 62%), you will over-bet.
Why? Because 62% is a point estimate from ~200-400 OOS observations.
With that sample size, the true win rate could easily be 54-70%.
If the true rate is 54% and you bet as if it's 62%, you are taking
excessive risk on a noisy estimate. This causes blow-up on small accounts.

THE SOLUTION: Wilson Score Confidence Interval
───────────────────────────────────────────────
The Wilson score CI gives a lower bound on the true win rate that
we are 90% confident the true precision exceeds.

Example:
  OOS precision = 62%, n = 250 samples
  Wilson 90% lower bound ≈ 57%
  Half-Kelly based on 57% (not 62%) → conservative, robust position size

This is standard practice in institutional quant risk management.
The half-Kelly further divides by 2 as an additional safety margin.

SIZING PIPELINE
───────────────
1. Wilson CI lower bound on OOS precision (conservative win rate estimate)
2. Half-Kelly on that lower bound
3. Multiply by regime sizing multiplier (0.0 – 1.0)
4. Multiply by conviction scalar (how much the model probability exceeds 50%)
5. Cap at MAX_POSITION_PCT (15%)
"""

import numpy as np
from scipy import stats
from config import CFG


class PositionSizer:

    @staticmethod
    def wilson_lower_bound(precision: float, n: int, confidence: float = 0.90) -> float:
        """
        Wilson score confidence interval lower bound.
        Returns the win rate we are `confidence`% sure the true precision exceeds.

        Formula:
          centre = (p + z²/2n) / (1 + z²/n)
          margin = z × sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)
          lower bound = centre - margin

        Where z = z-score for the confidence level (1.645 for 90%).
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
        Half-Kelly fraction.
        b = TARGET_GAIN / STOP_LOSS (the win/loss ratio)
        Kelly = (p×b − (1−p)) / b
        Half-Kelly = Kelly / 2
        """
        b     = CFG.TARGET_GAIN / CFG.STOP_LOSS
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
        """
        Full sizing pipeline:
          win_rate_lb  = Wilson CI lower bound (conservative floor)
          kelly_base   = Half-Kelly on that floor
          kelly_regime = Scaled by regime multiplier (0x in crash, 1x in bull)
          conviction   = (signal_prob - 0.5) × 2  → maps [0.5, 1.0] to [0.0, 1.0]
          kelly_final  = min(kelly_regime × conviction, MAX_POSITION_PCT)
        """
        win_rate_lb  = cls.wilson_lower_bound(oos_precision, oos_n)
        kelly_base   = cls.half_kelly(win_rate_lb)
        kelly_regime = kelly_base * regime_multiplier
        conviction   = max(0.0, (current_prob - 0.5) * 2)
        kelly_final  = min(kelly_regime * conviction, CFG.MAX_POSITION_PCT)

        return {
            "win_rate_ci_lb": round(win_rate_lb * 100, 1),
            "kelly_base_pct": round(kelly_base * 100, 1),
            "alloc_pct":      round(kelly_final * 100, 2),
            "dollar_size":    round(kelly_final * CFG.ACCOUNT_SIZE, 0),
            "conviction":     round(conviction, 3),
        }
