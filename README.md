# TITAN v6.0 — Macro-Quant Swing Trading Signal Engine

> **Portfolio project** demonstrating institutional-grade ML pipeline design for systematic equity trading.  
> Built for swing trading (5–20 day holds) on accounts under $50K with manual execution.

---

## What This Is

TITAN is a **quantitative signal generation engine** that scans a universe of equities and outputs a ranked shortlist of high-conviction swing trade candidates each day. It combines cross-asset macro feature engineering with a walk-forward validated XGBoost classifier, calibrated probability outputs, and Kelly-criterion position sizing with statistical confidence intervals.

The goal was not just to build something that *works*, but to build something architecturally sound — the kind of pipeline where every design decision has a documented rationale and every methodological trap (look-ahead bias, data leakage, overconfident probabilities) has been explicitly addressed.

---

## Pipeline Architecture

The engine runs in 5 sequential stages, each with a clear responsibility:

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 0 │ Regime Classifier                                     │
│          │ VIX + SPY trend + credit spread + breadth → BULL /   │
│          │ NEUTRAL / BEAR / CRASH  →  sizing multiplier (0–1x)  │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 1 │ Data Quality Gate                                     │
│          │ Liquidity filter · Earnings proximity blackout ·      │
│          │ Minimum data history check                            │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 2 │ Feature Engineering  (39 features across 6 groups)    │
│          │ Price momentum · Volatility regime · Trend/MR ·       │
│          │ Volume microstructure · Cross-asset macro ·           │
│          │ Market regime context                                  │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 3 │ Walk-Forward Alpha Model                              │
│          │ Expanding-window OOS · XGBoost · Platt calibration ·  │
│          │ Brier score validation                                 │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 4 │ Position Sizing                                       │
│          │ Wilson CI lower bound · Half-Kelly · Regime scaling · │
│          │ Conviction scalar · Hard 15% cap                      │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 5 │ Elite Filter + Composite Scorer                       │
│          │ Hard gates (precision, R/R, allocation, 200MA,        │
│          │ signal prob) → weighted composite rank → Top 5        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

This section documents the *why* behind the most important architectural choices. These are the decisions that separate a robust quantitative system from a curve-fitted backtest.

---

### 1. Walk-Forward OOS Validation (not k-fold cross-validation)

**Problem:** Standard k-fold cross-validation on time series data leaks future information into training folds. A model trained on data from 2023 should never be evaluated on 2022 data — but k-fold will happily do this. The result is inflated backtest metrics that collapse in live trading.

**Solution:** Expanding-window walk-forward validation. The model is trained on all data up to time `t`, evaluated on the next `OOS_WINDOW` days, then the window expands and repeats. OOS predictions are accumulated across all folds and metrics are computed only on this fully out-of-sample prediction set.

```python
# Simplified illustration of the walk-forward loop
train_end = INITIAL_TRAIN_WINDOW

while train_end + OOS_WINDOW <= len(data):
    X_train = X_all[:train_end]          # Past only
    X_test  = X_all[train_end:train_end + OOS_WINDOW]   # Future, unseen

    model.fit(X_train, y_train)
    oos_predictions.extend(model.predict(X_test))

    train_end += OOS_WINDOW              # Expand, never shuffle
```

**Why it matters:** A model with 58% OOS precision in walk-forward is genuinely predictive. A model with 58% precision in k-fold on financial data is probably overfit and will underperform in production.

---

### 2. Probability Calibration (Platt Scaling)

**Problem:** XGBoost outputs raw scores, not true probabilities. A raw XGB output of 0.72 does not mean "72% chance of success." In practice, tree ensemble models are systematically overconfident — their probabilities cluster near 0 and 1 more than they should. Using uncalibrated scores for position sizing is dangerous.

**Solution:** Platt scaling (logistic regression on top of XGBoost outputs) applied on the OOS test fold — not the training fold. This ensures the calibrator itself is not trained on data the base model has seen.

```python
# Train base model on training fold
base_model.fit(X_train, y_train)

# Calibrate on OOS fold (never on training data)
calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
calibrated.fit(X_oos, y_oos)

# Now predict_proba() outputs are true probabilities
true_prob = calibrated.predict_proba(X_new)[0, 1]
```

**Validation:** Brier score is computed on OOS predictions. A score of 0.25 is equivalent to random guessing on a binary outcome. The elite filter targets Brier < 0.20, meaning the model's confidence is meaningfully informative.

---

### 3. Wilson Score CI on Kelly Criterion

**Problem:** The standard Kelly formula takes a win rate as input and outputs a position size. But OOS precision is estimated from a finite sample — 200–400 observations. A measured precision of 60% could easily be 54% or 66% at the true underlying rate. Plugging the noisy point estimate into Kelly directly produces position sizes that are far too aggressive and will ruin a small account.

**Solution:** Compute the Wilson score confidence interval lower bound on OOS precision at 90% confidence. Use this conservative estimate as the Kelly input instead of the raw precision.

```python
def wilson_lower_bound(precision: float, n: int, confidence: float = 0.90) -> float:
    """
    Returns the lower bound of the Wilson score interval.
    This is the value we're 90% confident the true precision is at least.
    On n=300 samples at 60% measured precision, this might return ~55%.
    We bet on 55%, not 60% — conservative by design.
    """
    z      = stats.norm.ppf((1 + confidence) / 2)
    denom  = 1 + z**2 / n
    centre = (precision + z**2 / (2*n)) / denom
    margin = (z * sqrt(precision*(1-precision)/n + z**2/(4*n**2))) / denom
    return max(0.0, centre - margin)

# Then: half-Kelly on the conservative win rate, not the measured one
conservative_win_rate = wilson_lower_bound(oos_precision, n_oos_samples)
position_size = half_kelly(conservative_win_rate) * account_size
```

**Additionally:** Final Kelly is scaled by (a) the regime multiplier (0–1x based on market environment) and (b) a conviction scalar derived from the calibrated signal probability. Hard cap at 15% per position regardless.

---

### 4. Cross-Asset Macro Feature Engineering

**Innovation from v3.3:** Four cross-asset features derived from EU equity index, SPY, and Gold are computed and injected into every stock's feature matrix. These capture macro context that pure price-action models miss entirely.

| Feature | Source | Economic Logic |
|---|---|---|
| `EU_Return` | Euro Stoxx 50 log return | EU opens before US — a leading signal for multinational stocks and global risk appetite |
| `Rel_Strength_SPY` | Stock return − SPY return | Measures idiosyncratic alpha vs broad market; persistent outperformers tend to continue (momentum factor, Jegadeesh & Titman 1993) |
| `Corr_SPY_20d` | Rolling 20d correlation to SPY | Low correlation = stock moving independently = stronger signal, less macro noise |
| `Corr_GLD_20d` | Rolling 20d correlation to Gold | High positive correlation = risk-off / inflation-hedge behaviour in the name |

These are downloaded **once** at pipeline start and aligned to each ticker's date index via forward-fill — not re-downloaded per ticker (efficient and consistent).

```python
# Macro data loaded once, injected per ticker
aligned = macro_returns.reindex(stock_df.index, method="ffill").fillna(0.0)

features["rel_str_spy"]  = stock_log_ret - aligned["SPY_Return"]
features["corr_spy_20d"] = stock_log_ret.rolling(20).corr(aligned["SPY_Return"])
features["corr_gld_20d"] = stock_log_ret.rolling(20).corr(aligned["GLD_Return"])
```

---

### 5. Rules-Based Regime Classifier (deliberately not ML)

**Design choice that might seem counterintuitive:** The regime classifier — which gates all position sizing — is a deterministic rules-based system, not a machine-learned model.

**Reason:** A regime classifier trained on historical market states has access to at most 30–40 regime-shift events in 20 years of data. This is far too few samples to train any ML model without severe overfitting. Rules derived from economic principles (VIX level, credit spreads, trend confirmation, breadth) are transparent, auditable, and don't overfit to the specific shape of 2008 or 2020.

```
Regime score = sum of 5 binary signals:
  +2 if VIX < 20       (calm environment)
  +1 if VIX < 28       (moderate stress)
  +1 if SPY > 200-day MA
  +1 if SPY 1-month return > 0
  +1 if HYG above 20-day MA (no credit stress)

Score ≥ 4  →  BULL    (100% Kelly sizing)
Score ≥ 2  →  NEUTRAL (60% Kelly sizing)
Score < 2  →  BEAR    (25% Kelly sizing)
VIX ≥ 35   →  CRASH   (0% sizing, signals informational only)
```

---

### 6. Elite Signal Filter

Even after OOS validation, many signals survive that shouldn't be traded. The elite filter applies hard binary gates before ranking:

| Filter | Threshold | Rationale |
|---|---|---|
| OOS Precision | ≥ 58% | At 54–57% with a 2:1 R/R you're barely above breakeven after slippage |
| Reward/Risk | ≥ 2:1 | Below 2:1, you need >66% win rate to break even — unrealistic in swing trading |
| Kelly Allocation | ≥ 3% | Below 3%, the dollar position on a <$50K account is not worth the cognitive overhead |
| Price vs 200MA | Must be above | Structural downtrends dramatically reduce win rates on long swing setups |
| Signal Probability | ≥ 60% | Calibrated model must be meaningfully confident, not just above 50% |

Signals that pass all hard gates are ranked by a weighted composite score:

```
Score = 0.35 × OOS_precision_norm
      + 0.25 × signal_probability_norm
      + 0.20 × kelly_allocation_norm
      + 0.20 × brier_score_norm        ← rewards well-calibrated models
```

---

## Feature Set Summary

39 features across 6 groups, all computed using only data available at time `t`:

| Group | Features | Count |
|---|---|---|
| Price Momentum | Log return, 5/10/20/60d returns, MTF momentum (1–32d log-spaced), alignment score | 12 |
| Volatility | 10d/20d annualised vol, vol ratio (regime detection) | 3 |
| Trend / Mean-Reversion | Distance from 20/50/200-day SMA, SMA50/200 cross, Bollinger band position | 5 |
| Momentum Indicators | Wilder RSI, RSI regime buckets, MACD histogram, MACD cross signal | 5 |
| Volume / Microstructure | Relative volume (5d, 20d), volume-price agreement, ATR%, trend confirmation composite, choppiness | 6 |
| Market Regime Context | VIX level, SPY trend flag, credit stress flag, risk appetite | 4 |
| **Cross-Asset Macro** (v6.0) | EU return, Relative strength vs SPY, SPY correlation, Gold correlation | **4** |

---

## Label Construction

The binary target label is constructed carefully to avoid look-ahead bias:

- **Label = 1** if `max(Close[t+1 : t+HOLD_MAX])` ≥ `Close[t] × (1 + effective_target)`
- Uses **Close prices only** (not High). You can only realise a gain on a closing price — using High inflates hit rates and creates subtle look-ahead via intraday data.
- Effective target = raw target adjusted upward for round-trip slippage (10bps assumed).
- Last `HOLD_MAX + 1` rows receive label = -1 and are excluded from all training and evaluation.

---

## Tech Stack

- **Python 3.10+**
- `yfinance` — market data
- `xgboost` — gradient boosted classifier
- `scikit-learn` — calibration, metrics, preprocessing
- `scipy` — Wilson score confidence intervals
- `pandas / numpy` — data pipeline

---

## What This Demonstrates

For recruiters and hiring managers, this project is intended to show:

- Awareness of **common ML pitfalls in finance** (look-ahead bias, data leakage, miscalibrated probabilities) and how to fix them
- Ability to implement **institutional-grade methodologies** (walk-forward OOS, Platt scaling, Kelly with CI)
- Understanding of **cross-asset relationships** and how to engineer features from macro data
- **Systems thinking** — each component has a single responsibility, documented rationale, and graceful failure modes
- Clean, readable code that a team could maintain and extend

---

## Disclaimer

This project is for educational and portfolio demonstration purposes only. It does not constitute financial advice. Past signal performance does not guarantee future results. Always do your own research before making any investment decision.

---

*Built iteratively across TITAN v3.3 (Macro-Quant) → v5.0 (Institutional Architecture) → v6.0 (Synthesis)*
