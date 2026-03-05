"""
scoring.py — HybridScorer: Rule-Based Signal Validation Layer

PURPOSE
───────
The HybridScorer runs AFTER the ML model. It does not replace the
walk-forward XGBoost — it validates it with interpretable human logic.

A signal needs BOTH the ML model AND the rules to agree for maximum
conviction. When they diverge (high ML, low hybrid), it is a flag to
investigate rather than an automatic entry.

SCORE NORMALISATION
───────────────────
Raw score range: -160 (worst case) to +130 (best case) = 290 span.
Normalised to 0–100 using: norm = clip((raw + 160) / 290 × 100, 0, 100)

The -160 floor accounts for the maximum possible penalty stack:
  Below 200MA:         -40
  Overbought RSI:      -15
  Deep below 52w high: -20
  MTF strongly bearish:-40
  Distribution volume: -15
  Bear regime:         -30 (bear -20 + high RSI -10)
  STRESS VIX:          -12
  Credit stress:       -8
  Very volatile stock: -15
  High & rising rates: -30
  Total worst case:   -225 (but many penalties are mutually exclusive in practice)

INTEREST RATE TIERS (mutually exclusive elif — max one fires per signal)
─────────────────────────────────────────────────────────────────────────
  Tier 1: rates_high AND rates_rising → -30  (worst: high + still going up)
  Tier 2: rate_3m_change > 0.5%       → -20  (fast rising, not yet high)
  Tier 3: rates_rising (slowly)       → -10  (mild headwind)
  Tier 4: rate_3m_change < -0.5%      → +15  (falling = easing cycle tailwind)
  Tier 5: current_rate < 2.0%         → +10  (near-ZIRP = maximum multiple expansion)
"""

import numpy as np
import pandas as pd


class HybridScorer:

    @staticmethod
    def score(latest: pd.Series, regime_features: dict) -> tuple[int, list[str]]:
        """
        Returns (raw_score: int, reasons: list[str]).
        Raw score is then normalised to 0-100 in compute().
        """
        score, reasons = 0, []

        # ── Trend (max +40 / min -40) ──────────────────────────────────────────
        dist_200 = float(latest.get("dist_200", 0.0))
        if dist_200 > 0:
            score += 30; reasons.append("Above 200MA")
            if dist_200 > 0.10:
                score += 10; reasons.append("Strong uptrend (>10% above 200MA)")
        else:
            score -= 40; reasons.append("⚠️ Below 200MA")

        # ── RSI (max +20 / min -15) ────────────────────────────────────────────
        rsi = float(latest.get("rsi", 50.0))
        if 45 <= rsi <= 65:
            score += 20; reasons.append(f"Healthy RSI ({rsi:.0f})")
        elif rsi > 75:
            score -= 15; reasons.append(f"⚠️ Overbought ({rsi:.0f})")
        elif rsi < 30:
            score += 5;  reasons.append(f"Oversold ({rsi:.0f}) — bounce candidate")

        # ── MACD (max +15) ─────────────────────────────────────────────────────
        if float(latest.get("macd_hist", 0.0)) > 0:
            score += 10; reasons.append("MACD bullish")
            if float(latest.get("macd_cross", 0)) == 1:
                score += 5; reasons.append("🔥 Fresh MACD crossover")

        # ── 52-week high proximity (max +20 / min -20) ────────────────────────
        # George & Hwang (2004): proximity to 52w high predicts momentum.
        # 85–97% zone is the sweet spot — trending without being at resistance.
        p52 = float(latest.get("price_to_52w_high", 1.0))
        if 0.85 <= p52 < 0.97:
            score += 20; reasons.append(f"🔥 52w-high sweet spot ({p52*100:.0f}%)")
        elif p52 >= 0.97:
            score -= 10; reasons.append(f"⚠️ At 52w resistance ({p52*100:.0f}%)")
        elif 0.70 <= p52 < 0.85:
            score += 5;  reasons.append(f"Recovering from lows ({p52*100:.0f}%)")
        elif p52 < 0.70:
            score -= 20; reasons.append(f"⚠️ Deep below 52w high ({p52*100:.0f}%)")

        # ── Multi-timeframe alignment (max +25 / min -40) ─────────────────────
        mtf = float(latest.get("mtf_alignment", 0.0))
        if mtf >= 5:
            score += 25; reasons.append("🔥 MTF aligned bullish (5-6/6)")
        elif mtf >= 3:
            score += 12; reasons.append(f"MTF moderate bullish ({int(mtf)}/6)")
        elif mtf <= -5:
            score -= 40; reasons.append("🚨 MTF strongly bearish")
        elif mtf <= -3:
            score -= 25; reasons.append(f"⚠️ MTF bearish ({int(mtf)}/6)")

        # ── Volume conviction (max +20 / min -15) ─────────────────────────────
        vol_rel   = float(latest.get("vol_rel_20", 1.0))
        vol_agree = float(latest.get("vol_price_agree", 0.0))
        if vol_rel > 1.2:
            if vol_agree > 1.2:
                score += 20; reasons.append("🔥 High volume confirming upside")
            elif vol_agree < -1.2:
                score -= 15; reasons.append("⚠️ Distribution (high vol + down day)")
        elif vol_rel > 1.0:
            score += 5; reasons.append("Above-average volume")

        # ── Regime adjustment (max +10 / min -30) ─────────────────────────────
        vix_tier  = regime_features.get("vix_tier", "NORMAL")
        bull_sig  = regime_features.get("bullish_signals", 2)
        credit_ok = not regime_features.get("credit_stress", False)

        if bull_sig >= 4:
            score += 10; reasons.append(f"✅ Bull regime ({bull_sig}/5)")
        elif bull_sig <= 1:
            score -= 20; reasons.append(f"⚠️ Bear regime ({bull_sig}/5)")
            if rsi > 55:
                score -= 10; reasons.append("⚠️ High RSI in bear — don't chase")

        if vix_tier == "CALM":
            score += 8;  reasons.append("✅ Low vol (VIX<15)")
        elif vix_tier == "STRESS":
            score -= 12; reasons.append(f"⚠️ Elevated VIX ({regime_features.get('vix',0):.1f})")

        if not credit_ok:
            score -= 8; reasons.append("⚠️ Credit stress")

        # ── Stock volatility guard (penalty only) ──────────────────────────────
        atr_pct = float(latest.get("atr_pct", 0.02))
        if atr_pct > 0.05:
            score -= 15; reasons.append(f"⚠️ Very volatile stock (ATR {atr_pct*100:.1f}%/d)")
        elif atr_pct > 0.035:
            score -= 5;  reasons.append(f"Moderate volatility (ATR {atr_pct*100:.1f}%/d)")

        # ── Interest rate adjustment (max +15 / min -30, mutually exclusive) ──
        rates_rising   = bool(latest.get("rates_rising", 0))
        rates_high     = bool(latest.get("rates_high", 0))
        rate_3m_change = float(latest.get("rate_3m_change", 0.0))
        current_rate   = float(latest.get("rate_10y", 3.5))

        if rates_high and rates_rising:
            score -= 30
            reasons.append(
                f"🚨 High & rising rates (10Y: {current_rate:.1f}%, Δ3m: +{rate_3m_change:.2f}%)"
            )
        elif rate_3m_change > 0.5:
            score -= 20
            reasons.append(f"⚠️ Rapidly rising rates (Δ3m: +{rate_3m_change:.2f}%)")
        elif rates_rising:
            score -= 10
            reasons.append(f"⚠️ Rising rate environment (Δ3m: +{rate_3m_change:.2f}%)")
        elif rate_3m_change < -0.5:
            score += 15
            reasons.append(f"✅ Falling rates (Δ3m: {rate_3m_change:.2f}%) — easing cycle")
        elif current_rate < 2.0:
            score += 10
            reasons.append(f"✅ Very low rates ({current_rate:.1f}%) — accommodative")

        return score, reasons

    @classmethod
    def compute(cls, latest: pd.Series, regime_features: dict) -> dict:
        """
        Returns hybrid score normalised 0-100, raw score, and reasons list.
        Normalisation: floor=-160, ceiling=+130, span=290.
        """
        raw, reasons = cls.score(latest, regime_features)
        norm = int(np.clip((raw + 160) / 290 * 100, 0, 100))
        return {"hybrid_raw": raw, "hybrid_norm": norm, "reasons": reasons}
