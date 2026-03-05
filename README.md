# TITAN v7.1 — Macro-Quant Hybrid Swing Trading Engine

A quantitative equity signal engine combining institutional-grade walk-forward machine learning, cross-asset macro features, and interpretable rule-based validation. Built as a personal research project to explore systematic approaches to swing trading.

> **Disclaimer:** This is a personal research and learning project. It is not financial advice, not production-ready, and past backtest performance does not guarantee future results. Use at your own risk.

---

## What It Does

TITAN scans a watchlist of equities daily and outputs a ranked shortlist of swing trade candidates. Each signal passes through seven validation stages before appearing in the output, and every decision is explained in plain English via the HybridScorer breakdown.

**Sample output:**
```
REGIME : 🟢 BULL  |  VIX: 13.4 (CALM)  |  SPY: ABOVE  |  Credit: OK ✓  |  Sizing: 100%
RATES  : 10Y Treasury: 4.21%  (📉 FALLING, Δ3m: -0.42%)

[1] NVDA    SCORE: 0.721  PRICE:  134.50  ML: 94.2%  OOS: 71.3%  HYB: 78%  52WH: 91%
    BUY 44 @ ~$134.50  │  Stop: $129.12  │  Target: $145.26  │  Risk: $236
    [78%] : Above 200MA · Strong uptrend · Healthy RSI (54) · 🔥 52w-high sweet spot · ✅ Bull regime
```

---

## Architecture

```
Stage 0 │ Regime Classifier      │ 8-asset market regime (BULL/NEUTRAL/BEAR/CRASH)
        │ Interest Rate Loader   │ 10Y Treasury yield + 3 change windows
        │ Macro Data Loader      │ EU Stoxx, SPY, GLD cross-asset returns
────────┼────────────────────────┼──────────────────────────────────────────────────
Stage 1 │ Data Manager           │ Liquidity filter + earnings calendar check
Stage 2 │ Feature Engine         │ 46 features across 10 groups
Stage 3 │ Alpha Model            │ Walk-forward XGBoost + Platt calibration
Stage 4 │ Position Sizer         │ Wilson CI lower bound + Half-Kelly
Stage 5 │ Hybrid Scorer          │ Rule-based validation layer
Stage 6 │ Signal Filter          │ 7 hard gates + composite score ranking
Stage 7 │ Report                 │ Terminal output + CSV export
```

---

## Key Design Decisions

### 1. Walk-Forward Validation — Not k-Fold

Standard k-fold cross-validation cannot be used on time series. Randomly shuffling the data causes future bars to appear in the training set — the model appears to work because it has seen the future.

Walk-forward validation respects temporal order:
- Fold 1: Train [0:504], Test [504:567]
- Fold 2: Train [0:567], Test [567:630]
- Each test window contains only bars the model has never seen.

The OOS precision metric in the output reflects genuine out-of-sample performance.

### 2. Platt Scaling — Calibrated Probabilities

Raw XGBoost outputs are not probabilities. A model might output 0.9 for events that actually occur only 60% of the time. This overconfidence causes over-betting and eventual ruin.

Platt scaling fits a logistic regression on OOS fold outputs, mapping raw scores to calibrated probabilities. The Brier score in the output measures calibration quality (lower is better; 0.25 is random).

### 3. Wilson CI Kelly Sizing — Conservative Position Sizing

The Kelly criterion requires an accurate win rate estimate. A single OOS precision of 62% from 250 samples has a wide confidence interval — the true rate could be anywhere from 54% to 70%.

The pipeline:
1. Compute Wilson score 90% lower bound on OOS precision → conservative win rate floor
2. Apply Half-Kelly to that floor (divide by 2 for additional safety margin)
3. Scale by regime multiplier (0% in CRASH, 25% in BEAR, 100% in BULL)
4. Scale by ML conviction ((signal_prob − 0.5) × 2)
5. Cap at 15% max position

### 4. Rule-Based Regime — Not Machine-Learned

There are roughly 30-40 genuine regime shifts in 20 years of market data. That is far too few observations to train an ML model on regime. A model fit on ~35 events cannot generalise.

Rules derived from economics are transparent, auditable, and robust to the small-sample problem. VIX, credit spreads (HYG), SPY trend, and risk appetite are well-understood regime signals with decades of academic backing.

### 5. Historical Regime Reconstruction in Backtester

A common backtesting error: fetching today's live VIX and applying it to all historical bars. A 2021–2024 backtest using March 2026 regime data is fiction — the strategy knows the future.

The `BacktestRegimeEngine` pre-downloads all regime asset history and reconstructs the regime at each bar using only data available up to that point. This is the correct approach.

### 6. 52-Week High Proximity (George & Hwang, 2004)

One of the most cited momentum factors in academic finance. Stocks approaching but not yet at their 52-week high tend to continue outperforming. The mechanism: investors use the 52-week high as a reference point (anchoring bias), causing gradual price discovery rather than a single jump.

Sweet spot: 85–97% of 52-week high. At or above 97%: resistance zone, R/R worsens.

---

## Feature Groups (46 total)

| Group | Description | Count |
|-------|-------------|-------|
| 1 | Price momentum (5d, 10d, 20d, 60d log returns) | 5 |
| 2 | Multi-timeframe momentum (1/2/4/8/16/32d, alignment score) | 7 |
| 3 | Volatility (10d, 20d annualised, vol ratio) | 3 |
| 4 | Trend / mean-reversion (distance from 20/50/200MA, golden cross) | 4 |
| 5 | Momentum indicators (RSI Wilder, MACD histogram/cross, BB position) | 6 |
| 6 | Volume / microstructure (relative volume, ATR, trend confirmation, choppiness) | 6 |
| 7 | Market regime context (VIX, SPY trend, credit status, risk appetite) | 4 |
| 8 | Cross-asset macro (EU return, relative strength vs SPY, GLD correlation) | 4 |
| 9 | 52-week high proximity | 1 |
| 10 | Interest rate context (10Y yield, 1m/3m/6m changes, rising/high flags) | 6 |

---

## File Structure

```
TITAN/
├── README.md              ← This file
├── config.py              ← All parameters in one place
├── regime.py              ← RegimeClassifier, InterestRateLoader, MacroDataLoader
├── features.py            ← FeatureEngine (46 features), build_labels()
├── model.py               ← AlphaModel: walk-forward XGBoost + Platt calibration
├── sizing.py              ← PositionSizer: Wilson CI + Half-Kelly
├── scoring.py             ← HybridScorer: rule-based validation layer
├── filters.py             ← DataManager, SignalFilter, composite scorer
├── backtest.py            ← Backtester, BacktestRegimeEngine, run_backtest()
├── main.py                ← run_live(), print_report(), entry point
└── winning_stocks.csv     ← Watchlist (edit with your own tickers)
```

---

## Installation

```bash
pip install yfinance xgboost scikit-learn scipy numpy pandas
```

Python 3.10+ required.

---

## Usage

**Live signal scan:**
```bash
python main.py
```

**Backtest on watchlist:**
```bash
python main.py --backtest --start 2021-01-01 --end 2024-01-01
```

**Google Colab:**
Edit the `RUN_MODE` variable at the bottom of `main.py` and run the cell.

**Custom watchlist:**
Edit `winning_stocks.csv` with your tickers (one per row, column header: `ticker`).

---

## Output Columns Explained

| Column | Meaning |
|--------|---------|
| SCORE | Weighted composite rank score (higher = better) |
| ML% | Model's calibrated probability of hitting target within hold window |
| OOS | Out-of-sample precision across all walk-forward folds |
| W-LB | Wilson CI 90% lower bound on win rate (conservative estimate) |
| BRIER | Calibration quality score (lower = better; 0.25 = random) |
| ALLOC | Kelly-based position size as % of account |
| HYB | HybridScorer rule-based validation score (0–100) |
| 52WH | Current price as % of 52-week high |
| RS_SPY | 20-day return relative to SPY (green = outperforming) |

---

## Limitations

- Data source is Yahoo Finance via yfinance — data quality varies
- Walk-forward model is trained per-ticker independently (no cross-sectional learning)
- Backtester uses rule-based HybridScorer for speed, not the full ML pipeline
- No portfolio-level position correlation or drawdown management
- Transaction costs are estimated; real costs depend on broker and order size

---

## References

- George, T. & Hwang, C. (2004). *The 52-Week High and Momentum Investing*. Journal of Finance.
- Kelly, J.L. (1956). *A New Interpretation of Information Rate*. Bell System Technical Journal.
- Wilson, E.B. (1927). *Probable Inference, the Law of Succession, and Statistical Inference*. JASA.
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines*. MIT Press.
- Moskowitz, T., Ooi, Y.H., & Pedersen, L.H. (2012). *Time Series Momentum*. Journal of Financial Economics.
