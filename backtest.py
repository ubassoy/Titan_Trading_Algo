"""
backtest.py — Historical Backtester with Per-Bar Regime Reconstruction

THE CRITICAL DESIGN DECISION: No Look-Ahead on Regime
──────────────────────────────────────────────────────
The original Module 5 (HybridBacktester from Colab) had a fatal flaw:
  regime = get_market_regime()  # called ONCE, returns TODAY's live data
  for i in range(200, len(df)):
      score = calculate_hybrid_score(..., regime)  # same regime for ALL bars

This means a 2021-2024 backtest was using March 2026 VIX and SPY data to
make 2021 entry decisions. The strategy appeared to work because it knew
the future regime. Results were meaningless.

THE FIX: BacktestRegimeEngine
  1. Downloads full historical data for all regime assets (SPY, VIX, HYG, QQQ, IWM)
     starting 1 year before the backtest period (to warm up rolling windows)
  2. At each bar, slices data to [start : current_bar] — strictly past data
  3. Calls compute_regime_from_slice() on that historical slice
  4. Caches results per date to avoid recomputing (major speed improvement)

This is the only architecturally honest approach to backtesting a strategy
that uses real-time regime data.

NOTE ON ML MODEL IN BACKTEST
─────────────────────────────
The backtester uses HybridScorer (rule-based) for entry/exit decisions,
NOT the walk-forward ML model. Running the full walk-forward XGBoost per
bar across a multi-year backtest would take hours per ticker.
The ML model's walk-forward OOS metrics from model.py are the rigorous
validation. The backtester tests the rule-based logic independently.
"""

from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf

from config import CFG, log
from regime import compute_regime_from_slice
from features import FeatureEngine
from scoring import HybridScorer
from filters import DataManager


# ==============================================================================
# BACKTEST REGIME ENGINE
# ==============================================================================
class BacktestRegimeEngine:
    """
    Pre-downloads regime asset history.
    Reconstructs regime at each bar using only past data.
    """

    def __init__(self, start: str, end: str):
        self.start = start
        self.end   = end
        self._cache: dict[pd.Timestamp, dict] = {}
        self._regime_data: Optional[pd.DataFrame] = None

    def load(self) -> bool:
        log.info("Backtest ▶ Downloading historical regime data...")
        try:
            # Fetch 1 extra year before start to warm up 200-day rolling windows
            extended_start = (
                pd.Timestamp(self.start) - pd.DateOffset(years=1)
            ).strftime("%Y-%m-%d")

            raw = yf.download(
                ["SPY", "^VIX", "HYG", "QQQ", "IWM"],
                start=extended_start, end=self.end,
                progress=False, auto_adjust=True,
            )
            close = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            close.dropna(how="all", inplace=True)
            self._regime_data = close
            log.info(f"  Regime history: {len(close)} bars ({extended_start} → {self.end})")
            return True
        except Exception as e:
            log.error(f"Backtest regime load failed: {e}")
            return False

    def get_regime_at(self, date: pd.Timestamp) -> dict:
        """
        Return regime reconstructed using only data up to `date`.
        Cached after first computation for each date.
        """
        if date in self._cache:
            return self._cache[date]

        _fallback = {
            "vix": 20.0, "vix_tier": "NORMAL", "spy_trend": "ABOVE",
            "credit_stress": False, "risk_appetite_5d": 0.0,
            "bullish_signals": 2, "regime": "NEUTRAL", "sizing_mult": 0.6,
        }

        if self._regime_data is None:
            return _fallback

        slice_ = self._regime_data[self._regime_data.index <= date]
        if len(slice_) < 210:
            return _fallback

        result = compute_regime_from_slice(
            spy_slice=slice_["SPY"].dropna(),
            vix_slice=slice_["^VIX"].dropna(),
            hyg_slice=slice_["HYG"].dropna(),
            qqq_slice=slice_["QQQ"].dropna(),
            iwm_slice=slice_["IWM"].dropna(),
        )
        self._cache[date] = result
        return result


# ==============================================================================
# BACKTESTER
# ==============================================================================
class Backtester:

    def __init__(
        self,
        regime_engine: BacktestRegimeEngine,
        rate_data: Optional[pd.DataFrame] = None,
    ):
        self.regime_engine = regime_engine
        self.rate_data     = rate_data

    def _compute_metrics(
        self,
        ticker:     str,
        trades:     list[dict],
        equity:     list[float],
        spy_return: float,
    ) -> Optional[dict]:
        """
        Compute full performance metrics from the trade log and equity curve.

        Metrics:
          total_return : Strategy total return over the period
          spy_return   : SPY buy-and-hold return (benchmark)
          alpha        : Strategy return minus SPY (the only metric that matters)
          win_rate     : % of closed trades that were profitable
          avg_win_pct  : Average winner return
          avg_loss_pct : Average loser return
          profit_factor: Gross wins / gross losses (>1.0 = net positive)
          expectancy   : Average outcome per trade as % — E[return]
          sharpe       : Annualised Sharpe ratio from daily equity curve
          max_dd_pct   : Maximum peak-to-trough drawdown on equity curve
          avg_hold_days: Average trade duration
        """
        sell_trades = [t for t in trades if t["type"] == "SELL"]
        n_trades    = len(sell_trades)
        if n_trades == 0:
            return None

        returns    = [t["pnl"] for t in sell_trades]
        wins       = [r for r in returns if r > 0]
        losses     = [r for r in returns if r <= 0]
        win_rate   = len(wins) / n_trades

        eq_series  = pd.Series(equity)
        daily_ret  = eq_series.pct_change().dropna()
        sharpe     = (
            daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        ) if daily_ret.std() > 0 else 0.0

        roll_max   = eq_series.cummax()
        max_dd     = float(((eq_series - roll_max) / roll_max).min())

        gross_win  = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-9

        total_return = (eq_series.iloc[-1] / eq_series.iloc[0]) - 1
        avg_hold     = np.mean([t.get("days_held", 0) for t in sell_trades])

        return {
            "ticker":         ticker,
            "total_return":   round(total_return * 100, 2),
            "spy_return":     round(spy_return * 100, 2),
            "alpha":          round((total_return - spy_return) * 100, 2),
            "n_trades":       n_trades,
            "win_rate":       round(win_rate * 100, 1),
            "avg_win_pct":    round(np.mean(wins) * 100, 2) if wins else 0.0,
            "avg_loss_pct":   round(np.mean(losses) * 100, 2) if losses else 0.0,
            "profit_factor":  round(gross_win / gross_loss, 2),
            "expectancy_pct": round(np.mean(returns) * 100, 2),
            "sharpe":         round(sharpe, 2),
            "max_dd_pct":     round(max_dd * 100, 2),
            "avg_hold_days":  round(avg_hold, 1),
        }

    def run(self, ticker: str, start: str, end: str) -> Optional[dict]:
        log.info(f"  Backtesting {ticker}...")

        df = DataManager.fetch(ticker, start=start, end=end)
        if df is None:
            return None

        # SPY buy-and-hold benchmark for the same period
        try:
            spy_raw = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(spy_raw.columns, pd.MultiIndex):
                spy_raw.columns = spy_raw.columns.get_level_values(0)
            spy_return = float(spy_raw["Close"].iloc[-1] / spy_raw["Close"].iloc[0] - 1)
        except Exception:
            spy_return = 0.0

        # Build features with neutral regime placeholder
        # (real regime is injected bar-by-bar in the loop below)
        neutral_regime = {
            "vix": 20.0, "vix_tier": "NORMAL", "spy_trend": "ABOVE",
            "credit_stress": False, "risk_appetite_5d": 0.0, "bullish_signals": 2,
        }
        df_feat = FeatureEngine.build(
            df, neutral_regime, macro_returns=None, rate_data=self.rate_data
        )
        df_feat = df_feat.dropna(subset=["sma_200", "rsi", "atr_14"])

        if len(df_feat) < 200:
            return None

        # Dynamic slippage based on this stock's liquidity
        avg_volume = df["Volume"].rolling(30).mean().iloc[-1]
        avg_price  = df["Close"].rolling(30).mean().iloc[-1]
        slippage_bps = CFG.estimate_slippage(avg_volume, avg_price)

        # ── Simulation state ──────────────────────────────────────────────────
        capital       = CFG.BT_INITIAL_CAPITAL
        position      = 0
        entry_price   = 0.0
        entry_atr     = 0.0
        days_in_trade = 0
        trades: list[dict]  = []
        equity: list[float] = [capital]

        start_idx = max(200, CFG.BT_REGIME_LOOKBACK)

        for i in range(start_idx, len(df_feat)):
            row  = df_feat.iloc[i]
            date = df_feat.index[i]

            # ── Historical regime reconstruction — KEY anti-look-ahead step ──
            regime = self.regime_engine.get_regime_at(date)

            # Inject per-bar regime scalars into the row for HybridScorer
            row_with_regime = row.copy()
            row_with_regime["regime_vix"]       = regime.get("vix", 20.0)
            row_with_regime["regime_spy_trend"]  = 1 if regime.get("spy_trend") == "ABOVE" else 0
            row_with_regime["regime_credit_ok"]  = 0 if regime.get("credit_stress") else 1
            row_with_regime["regime_risk_app"]   = regime.get("risk_appetite_5d", 0.0)

            hybrid_score, _ = HybridScorer.score(row_with_regime, regime)

            if position > 0:
                days_in_trade += 1

            close = float(row["Close"])
            atr   = float(row["atr_14"]) if not pd.isna(row["atr_14"]) else close * 0.02

            # ── EXIT LOGIC ────────────────────────────────────────────────────
            if position > 0:
                pnl_pct     = (close / entry_price) - 1
                atr_stop    = close < (entry_price - CFG.BT_ATR_STOP_MULT * entry_atr)
                fixed_stop  = pnl_pct <= -CFG.STOP_LOSS
                take_profit = pnl_pct >= CFG.BT_PROFIT_TARGET
                score_exit  = hybrid_score < CFG.BT_EXIT_SCORE_MIN
                time_stop   = days_in_trade >= CFG.BT_MAX_HOLD_DAYS
                crash       = regime.get("regime") == "CRASH"

                if any([atr_stop, fixed_stop, take_profit, score_exit, time_stop, crash]):
                    exit_price = close * (1 - slippage_bps / 10_000)
                    proceeds   = position * exit_price * (1 - CFG.BT_COMMISSION)
                    capital   += proceeds
                    pnl_final  = (exit_price / entry_price) - 1
                    reason = (
                        "ATR-stop"    if atr_stop    else
                        "Fixed-stop"  if fixed_stop  else
                        "Take-profit" if take_profit  else
                        "Score-exit"  if score_exit   else
                        "CRASH"       if crash         else
                        "Time-stop"
                    )
                    trades.append({
                        "type": "SELL", "date": date, "price": exit_price,
                        "pnl": pnl_final, "reason": reason,
                        "days_held": days_in_trade, "regime": regime.get("regime", "?"),
                    })
                    position, days_in_trade = 0, 0

            # ── ENTRY LOGIC ───────────────────────────────────────────────────
            elif position == 0:
                in_regime    = regime.get("regime") not in ("CRASH", "BEAR")
                score_ok     = hybrid_score >= CFG.BT_ENTRY_SCORE_MIN
                above_200    = float(row.get("dist_200", -1)) > 0
                not_high_vol = float(row.get("atr_pct", 0)) < 0.05

                if in_regime and score_ok and above_200 and not_high_vol:
                    entry_fill = close * (1 + slippage_bps / 10_000)
                    shares     = int((capital * CFG.BT_POSITION_SIZE_PCT) / entry_fill)
                    if shares > 0:
                        cost = shares * entry_fill * (1 + CFG.BT_COMMISSION)
                        if cost <= capital:
                            position, entry_price, entry_atr = shares, entry_fill, atr
                            capital      -= cost
                            days_in_trade = 0
                            trades.append({
                                "type": "BUY", "date": date, "price": entry_fill,
                                "score": hybrid_score, "regime": regime.get("regime", "?"),
                            })

            # Mark-to-market equity at end of each bar
            equity.append(capital + (position * close if position > 0 else 0))

        # Force-close any open position at period end
        if position > 0:
            final_price = float(df_feat["Close"].iloc[-1]) * (1 - slippage_bps / 10_000)
            capital    += position * final_price * (1 - CFG.BT_COMMISSION)
            trades.append({
                "type": "SELL", "date": df_feat.index[-1], "price": final_price,
                "pnl": (final_price / entry_price) - 1,
                "reason": "Period-end", "days_held": days_in_trade, "regime": "?",
            })
            equity.append(capital)

        return self._compute_metrics(ticker, trades, equity, spy_return)


# ==============================================================================
# BACKTEST RUNNER
# ==============================================================================
def run_backtest(start: str, end: str):
    """Entry point for backtest mode. Called from main.py."""
    from regime import InterestRateLoader

    print()
    print("=" * 110)
    print(f"  TITAN v7.1 — BACKTESTER    {start} → {end}")
    print(
        f"  Capital: ${CFG.BT_INITIAL_CAPITAL:,.0f}  |  "
        f"Entry score ≥ {CFG.BT_ENTRY_SCORE_MIN}  |  "
        f"Exit score < {CFG.BT_EXIT_SCORE_MIN}  |  "
        f"Stop: -{CFG.STOP_LOSS*100:.0f}% or {CFG.BT_ATR_STOP_MULT}×ATR  |  "
        f"Target: +{CFG.BT_PROFIT_TARGET*100:.0f}%  |  "
        f"Time stop: {CFG.BT_MAX_HOLD_DAYS}d"
    )
    print("=" * 110)

    regime_engine = BacktestRegimeEngine(start, end)
    if not regime_engine.load():
        print("❌ Could not load historical regime data. Aborting.")
        return

    rate_loader = InterestRateLoader()
    rate_loader.load(period="max")

    backtester = Backtester(regime_engine, rate_data=rate_loader.rate_data)
    watchlist  = DataManager.load_watchlist()
    results    = []

    for ticker in watchlist:
        res = backtester.run(ticker, start, end)
        if res:
            results.append(res)

    if not results:
        print("❌ No results produced.")
        return

    results.sort(key=lambda x: x["sharpe"], reverse=True)

    # ── Results table ─────────────────────────────────────────────────────────
    print()
    print(
        f"  {'TICKER':<7} | {'RETURN':>7} | {'SPY':>6} | {'ALPHA':>7} | "
        f"{'TRADES':>6} | {'WIN%':>5} | {'AVG_W':>6} | {'AVG_L':>6} | "
        f"{'PF':>5} | {'EXPCT':>6} | {'SHARPE':>6} | {'MAX_DD':>7} | {'HOLD':>5}"
    )
    print("  " + "-" * 108)

    for r in results:
        flag = (
            "🏆" if r["sharpe"] > 1.0 and r["alpha"] > 5 else
            "✅" if r["alpha"] > 0 else "⚠️"
        )
        print(
            f"  {r['ticker']:<7} | {r['total_return']:>+6.1f}% | {r['spy_return']:>+5.1f}% | "
            f"{r['alpha']:>+6.1f}% | {r['n_trades']:>6} | {r['win_rate']:>4.0f}% | "
            f"{r['avg_win_pct']:>+5.1f}% | {r['avg_loss_pct']:>+5.1f}% | "
            f"{r['profit_factor']:>5.2f} | {r['expectancy_pct']:>+5.2f}% | "
            f"{r['sharpe']:>+6.2f} | {r['max_dd_pct']:>+6.1f}% | "
            f"{r['avg_hold_days']:>4.0f}d  {flag}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n  " + "─" * 108)
    print("  PORTFOLIO SUMMARY")
    print("  " + "─" * 108)
    n = len(results)
    print(
        f"  Avg return:    {_mean([r['total_return'] for r in results]):+.1f}%  |  "
        f"Avg alpha:    {_mean([r['alpha'] for r in results]):+.1f}%"
    )
    print(
        f"  Avg Sharpe:    {_mean([r['sharpe'] for r in results]):+.2f}  |  "
        f"Avg max DD:   {_mean([r['max_dd_pct'] for r in results]):+.1f}%"
    )
    print(
        f"  Avg win rate:  {_mean([r['win_rate'] for r in results]):.0f}%  |  "
        f"Avg PF:       {_mean([r['profit_factor'] for r in results]):.2f}"
    )
    print(f"  Positive alpha: {sum(1 for r in results if r['alpha'] > 0)}/{n} tickers")
    print()
    print("  COLUMN LEGEND:")
    print("    ALPHA  : Strategy return minus SPY buy-and-hold (the key metric)")
    print("    PF     : Profit factor — gross wins / gross losses. >1.0 = net positive")
    print("    EXPCT  : Expectancy — average % return per trade")
    print("    SHARPE : Annualised Sharpe ratio from daily equity curve")
    print("    MAX_DD : Maximum peak-to-trough drawdown on equity curve")
    print()
    print("  ⚠️  NOTE: Backtester uses HybridScorer (rule-based) for speed.")
    print("      Walk-forward OOS precision from model.py is the ML validation.")
    print("      Past performance does not guarantee future results.")
    print("=" * 110)

    pd.DataFrame(results).to_csv(CFG.BT_OUTPUT_CSV, index=False)
    log.info(f"Backtest results saved → {CFG.BT_OUTPUT_CSV}")


def _mean(lst: list) -> float:
    """Simple mean helper — avoids a top-level numpy import just for this."""
    return sum(lst) / len(lst) if lst else 0.0
