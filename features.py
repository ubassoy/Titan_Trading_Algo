"""
features.py — Feature Engineering + Label Construction

46 features across 10 groups fed to the XGBoost model.

LABEL CONSTRUCTION NOTE:
  Labels use Close-to-Close only — NOT intraday High.
  Using High would introduce look-ahead bias: you cannot guarantee
  realising an intraday high at close price. Labels are therefore
  conservative but honest.

  Label = 1 if max(Close[t+1 : t+HOLD_MAX]) >= Close[t] × (1 + effective_target)
  Label = 0 otherwise
  Label = -1 for last HOLD_MAX rows (excluded from training — no valid label)
"""

from typing import Optional
import numpy as np
import pandas as pd

from config import CFG, EFFECTIVE_TARGET, log


class FeatureEngine:

    @staticmethod
    def build(
        df:              pd.DataFrame,
        regime_features: dict,
        macro_returns:   Optional[pd.DataFrame] = None,
        rate_data:       Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:

        d = df.copy()

        # ── Group 1: Price momentum ────────────────────────────────────────────
        # Raw return signals at multiple horizons. The model learns which
        # lookback matters most for each stock's behaviour.
        d["log_ret"]  = np.log(d["Close"] / d["Close"].shift(1))
        d["ret_5d"]   = d["Close"].pct_change(5)
        d["ret_10d"]  = d["Close"].pct_change(10)
        d["ret_20d"]  = d["Close"].pct_change(20)
        d["ret_60d"]  = d["Close"].pct_change(60)

        # ── Group 2: Multi-timeframe momentum (Moskowitz et al. 2012) ─────────
        # Log-spaced lookbacks (1,2,4,8,16,32) sample trend persistence at
        # multiple scales simultaneously. mtf_alignment = count of positive
        # windows − count of negative windows: ranges from -6 to +6.
        # All 6 positive = trend clean across all timeframes.
        for days in [1, 2, 4, 8, 16, 32]:
            d[f"mtf_ret_{days}d"] = d["Close"].pct_change(days)
        d["mtf_alignment"] = sum(
            np.sign(d[f"mtf_ret_{days}d"]) for days in [1, 2, 4, 8, 16, 32]
        )

        # ── Group 3: Volatility ────────────────────────────────────────────────
        # vol_ratio > 1 = volatility expanding = potential regime shift signal.
        d["vol_10d"]   = d["log_ret"].rolling(10).std() * np.sqrt(252)
        d["vol_20d"]   = d["log_ret"].rolling(20).std() * np.sqrt(252)
        d["vol_ratio"] = d["vol_10d"] / d["vol_20d"].replace(0, np.nan)

        # ── Group 4: Trend / mean-reversion ───────────────────────────────────
        # Distance from moving averages captures both trend strength and
        # mean-reversion risk. trend_50_200 = golden/death cross magnitude.
        d["sma_20"]       = d["Close"].rolling(20).mean()
        d["sma_50"]       = d["Close"].rolling(50).mean()
        d["sma_200"]      = d["Close"].rolling(200).mean()
        d["dist_20"]      = (d["Close"] - d["sma_20"])  / d["sma_20"]
        d["dist_50"]      = (d["Close"] - d["sma_50"])  / d["sma_50"]
        d["dist_200"]     = (d["Close"] - d["sma_200"]) / d["sma_200"]
        d["trend_50_200"] = (d["sma_50"] - d["sma_200"]) / d["sma_200"]

        # ── Group 5: Momentum indicators ──────────────────────────────────────
        # RSI uses Wilder's EWM smoothing (alpha=1/14) — the correct
        # implementation. Standard rolling-mean RSI is technically wrong
        # and gives different values, especially in volatile periods.
        delta    = d["Close"].diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        d["rsi"]            = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))
        d["rsi_oversold"]   = (d["rsi"] < 35).astype(int)
        d["rsi_overbought"] = (d["rsi"] > 70).astype(int)

        ema12           = d["Close"].ewm(span=12, adjust=False).mean()
        ema26           = d["Close"].ewm(span=26, adjust=False).mean()
        d["macd"]       = ema12 - ema26
        d["macd_sig"]   = d["macd"].ewm(span=9, adjust=False).mean()
        d["macd_hist"]  = d["macd"] - d["macd_sig"]
        d["macd_cross"] = (
            (d["macd"] > d["macd_sig"]) & (d["macd"].shift(1) <= d["macd_sig"].shift(1))
        ).astype(int)

        # Bollinger band position: normalised to [-1, +1]
        bb_mid      = d["Close"].rolling(20).mean()
        bb_std      = d["Close"].rolling(20).std()
        d["bb_pos"] = (d["Close"] - bb_mid) / (2 * bb_std.replace(0, np.nan))

        # ── Group 6: Volume / microstructure ──────────────────────────────────
        # vol_price_agree: positive = volume confirms price direction (bullish).
        # trend_confirm: composite flag requiring price trend + vol expansion
        # + above-average volume simultaneously — all three must agree.
        # choppiness_5d: counts direction changes in last 5 bars (0=trend, 4=choppy).
        d["vol_rel_20"]      = d["Volume"] / d["Volume"].rolling(20).mean().replace(0, np.nan)
        d["vol_rel_5"]       = d["Volume"] / d["Volume"].rolling(5).mean().replace(0, np.nan)
        d["vol_price_agree"] = d["vol_rel_20"] * np.sign(d["log_ret"])

        hl  = d["High"] - d["Low"]
        hc  = (d["High"] - d["Close"].shift(1)).abs()
        lc  = (d["Low"]  - d["Close"].shift(1)).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        d["atr_14"]  = tr.ewm(span=14, adjust=False).mean()
        d["atr_pct"] = d["atr_14"] / d["Close"]

        mom_ok  = (d["Close"] > d["sma_20"]).astype(int)
        atr_exp = (d["atr_14"] > d["atr_14"].rolling(14).mean()).astype(int)
        vol_ok  = (d["Volume"] > d["Volume"].rolling(20).mean()).astype(int)
        d["trend_confirm"] = mom_ok * atr_exp * vol_ok

        daily_dir = np.sign(d["Close"].diff())
        d["choppiness_5d"] = sum(
            (daily_dir.shift(i) != daily_dir.shift(i + 1)).astype(int) for i in range(4)
        )

        # ── Group 7: Market regime context ────────────────────────────────────
        # Scalar regime values injected from RegimeClassifier.
        # Same value across all rows on a given scan date.
        # This lets the model learn that signals in bear regimes have
        # lower historical success rates.
        d["regime_vix"]       = regime_features.get("vix", 20.0)
        d["regime_spy_trend"] = 1 if regime_features.get("spy_trend") == "ABOVE" else 0
        d["regime_credit_ok"] = 0 if regime_features.get("credit_stress") else 1
        d["regime_risk_app"]  = regime_features.get("risk_appetite_5d", 0.0)

        # ── Group 8: Cross-asset macro ─────────────────────────────────────────
        # eu_return    : EU opens before US — lead-lag signal for global risk sentiment
        # rel_str_spy  : stock return - SPY return = idiosyncratic alpha
        # corr_spy_20d : rolling correlation to SPY (low = independent signal)
        # corr_gld_20d : rolling correlation to GLD (high = risk-off/safe-haven)
        # Forward-fill only (no backward-fill) to prevent look-ahead bias.
        if macro_returns is not None:
            aligned = macro_returns.reindex(d.index, method="ffill").fillna(0.0)
            spy_ret = aligned.get("SPY_Return", pd.Series(0.0, index=d.index))
            gld_ret = aligned.get("GLD_Return", pd.Series(0.0, index=d.index))
            eu_ret  = aligned.get("EU_Return",  pd.Series(0.0, index=d.index))
            d["eu_return"]    = eu_ret
            d["rel_str_spy"]  = d["log_ret"] - spy_ret
            d["corr_spy_20d"] = d["log_ret"].rolling(20).corr(spy_ret)
            d["corr_gld_20d"] = d["log_ret"].rolling(20).corr(gld_ret)
        else:
            for col in ["eu_return", "rel_str_spy", "corr_spy_20d", "corr_gld_20d"]:
                d[col] = 0.0

        # ── Group 9: 52-week high proximity (George & Hwang 2004) ─────────────
        # Stocks near their 52w high (85-97% zone) are in confirmed uptrends
        # without being at resistance. One of the most cited momentum factors
        # in academic finance. At or above 98% = resistance zone, R/R worsens.
        d["price_to_52w_high"] = d["Close"] / d["High"].rolling(252).max().replace(0, np.nan)

        # ── Group 10: Interest rate context ───────────────────────────────────
        # Rising 10Y Treasury = higher discount rate = headwind for growth stocks.
        # Forward-fill only — zero-fill leading NaN rows to avoid look-ahead
        # (important when rate history is shorter than stock price history).
        if rate_data is not None:
            aligned = rate_data.reindex(d.index, method="ffill").fillna(0.0)
            d["rate_10y"]       = aligned["treasury_10y"]
            d["rate_1m_change"] = aligned["rate_1m_change"]
            d["rate_3m_change"] = aligned["rate_3m_change"]
            d["rate_6m_change"] = aligned["rate_6m_change"]
            d["rates_rising"]   = aligned["rates_rising"]
            d["rates_high"]     = aligned["rates_high"]
        else:
            for col in ["rate_10y", "rate_1m_change", "rate_3m_change",
                        "rate_6m_change", "rates_rising", "rates_high"]:
                d[col] = 0.0

        return d.replace([np.inf, -np.inf], np.nan)


# Complete feature list fed to XGBoost — 46 features total
FEATURE_COLS = [
    # 1. Price momentum (5)
    "log_ret", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    # 2. Multi-timeframe momentum (7)
    "mtf_ret_1d", "mtf_ret_2d", "mtf_ret_4d",
    "mtf_ret_8d", "mtf_ret_16d", "mtf_ret_32d", "mtf_alignment",
    # 3. Volatility (3)
    "vol_10d", "vol_20d", "vol_ratio",
    # 4. Trend / mean-reversion (4)
    "dist_20", "dist_50", "dist_200", "trend_50_200",
    # 5. Momentum indicators (6)
    "rsi", "rsi_oversold", "rsi_overbought", "macd_hist", "macd_cross", "bb_pos",
    # 6. Volume / microstructure (6)
    "vol_rel_20", "vol_rel_5", "vol_price_agree",
    "atr_pct", "trend_confirm", "choppiness_5d",
    # 7. Market regime context (4)
    "regime_vix", "regime_spy_trend", "regime_credit_ok", "regime_risk_app",
    # 8. Cross-asset macro (4)
    "eu_return", "rel_str_spy", "corr_spy_20d", "corr_gld_20d",
    # 9. 52-week high proximity (1)
    "price_to_52w_high",
    # 10. Interest rate context (6)
    "rate_10y", "rate_1m_change", "rate_3m_change", "rate_6m_change",
    "rates_rising", "rates_high",
]


def build_labels(df: pd.DataFrame) -> pd.Series:
    """
    Construct binary labels for the ML model.

    Label = 1 if the stock achieves the effective target gain within
            HOLD_DAYS_MAX trading days, using Close prices only.
    Label = 0 if it does not.
    Label = -1 for the last HOLD_DAYS_MAX rows (no valid forward window).

    effective_target adjusts the raw TARGET_GAIN upward for round-trip
    slippage so the model only fires on trades that are profitable after costs.
    """
    close  = df["Close"].values
    n      = len(close)
    H      = CFG.HOLD_DAYS_MAX
    labels = np.zeros(n, dtype=int)
    for i in range(n - H - 1):
        if close[i + 1 : i + H + 1].max() >= close[i] * (1 + EFFECTIVE_TARGET):
            labels[i] = 1
    labels[-(H + 1):] = -1
    return pd.Series(labels, index=df.index, name="label")
