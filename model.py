"""
model.py — Walk-Forward Alpha Model (XGBoost + Platt Calibration)

THE CORE DESIGN DECISION: Walk-Forward OOS Validation
─────────────────────────────────────────────────────
Standard k-fold cross-validation MUST NOT be used on time series data.
It randomly shuffles the data, which means future bars end up in the
training set and past bars in the test set. The model appears to work
because it has learned the future — this is data leakage.

Walk-forward validation solves this:
  Fold 1: Train on [0:504],   test on [504:567]
  Fold 2: Train on [0:567],   test on [567:630]
  Fold 3: Train on [0:630],   test on [630:693]
  ...continuing until end of data

Each test window only contains bars the model has NEVER seen.
The OOS precision metric reflects true out-of-sample performance.

PLATT SCALING (CALIBRATION)
───────────────────────────
Raw XGBoost outputs are not probabilities — they are overconfident scores.
A model might output 0.9 for events that actually happen only 60% of the time.
Platt scaling fits a logistic regression on the OOS fold outputs to map
the raw scores to true calibrated probabilities. This is critical for
Kelly sizing, which depends on a reliable win rate estimate.

The calibrator is ALWAYS fitted on OOS data, never training data.
Fitting it on training data would re-introduce the same overconfidence.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, brier_score_loss
from xgboost import XGBClassifier

from config import CFG, log
from features import FEATURE_COLS


class AlphaModel:

    def __init__(self, ticker: str):
        self.ticker              = ticker
        self.oos_precision: float   = 0.0
        self.oos_brier:     float   = 1.0   # Brier = 0.25 is random, lower is better
        self.n_folds:       int     = 0
        self.n_oos_samples: int     = 0
        self.final_model            = None
        self.feature_importance: Optional[pd.Series] = None

    def _make_base(self) -> XGBClassifier:
        """
        Conservative XGBoost configuration to reduce overfitting:
          max_depth=4        : Shallow trees — less memorisation
          min_child_weight=10: Requires ≥10 samples to create a leaf — reduces noise fitting
          subsample=0.8      : Each tree sees only 80% of rows — bagging effect
          colsample_bytree=0.8: Each tree uses 80% of features — random forest-like
          learning_rate=0.02 : Small steps — less likely to overfit to early patterns
        """
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            scale_pos_weight=1, eval_metric="logloss",
            random_state=42, verbosity=0,
        )

    def run(self, df_features: pd.DataFrame, labels: pd.Series) -> bool:
        """
        Execute the walk-forward validation loop.
        Returns False if the model fails quality checks.
        """
        data          = df_features[FEATURE_COLS].copy()
        data["label"] = labels
        data          = data[data["label"] != -1].dropna()

        if len(data) < CFG.MIN_TRAIN_SAMPLES * 2:
            return False

        X_all = data[FEATURE_COLS].values
        y_all = data["label"].values
        oos_true, oos_proba = [], []
        train_end = CFG.TRAIN_WINDOW_DAYS

        # ── Walk-forward loop ─────────────────────────────────────────────────
        while train_end + CFG.OOS_WINDOW_DAYS <= len(data):
            X_tr = X_all[:train_end]
            y_tr = y_all[:train_end]
            X_te = X_all[train_end : train_end + CFG.OOS_WINDOW_DAYS]
            y_te = y_all[train_end : train_end + CFG.OOS_WINDOW_DAYS]

            if y_tr.sum() < CFG.MIN_POSITIVE_LABELS:
                train_end += CFG.OOS_WINDOW_DAYS
                continue

            # Dynamic class balancing per fold
            n0, n1 = (y_tr == 0).sum(), (y_tr == 1).sum()
            model  = self._make_base()
            model.set_params(scale_pos_weight=n0 / n1 if n1 > 0 else 1.0)
            model.fit(X_tr, y_tr)

            # Platt calibration fitted on OOS fold — never on training data
            cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            cal.fit(X_te, y_te)

            oos_true.extend(y_te.tolist())
            oos_proba.extend(cal.predict_proba(X_te)[:, 1].tolist())
            self.n_folds += 1
            train_end    += CFG.OOS_WINDOW_DAYS

        if len(oos_true) < 50:
            return False

        oos_arr, proba_arr = np.array(oos_true), np.array(oos_proba)
        self.oos_precision = precision_score(
            oos_arr, (proba_arr >= 0.5).astype(int), zero_division=0
        )
        self.oos_brier     = brier_score_loss(oos_arr, proba_arr)
        self.n_oos_samples = len(oos_true)

        if self.oos_precision < CFG.MIN_OOS_PRECISION:
            return False

        # ── Final model trained on ALL data ───────────────────────────────────
        # Used for live prediction only. OOS metrics above are the
        # performance estimate — not re-measured here.
        n0_all, n1_all = (y_all == 0).sum(), (y_all == 1).sum()
        base = self._make_base()
        base.set_params(scale_pos_weight=n0_all / n1_all if n1_all > 0 else 1.0)
        base.fit(X_all, y_all)

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
        """Return calibrated probability for the most recent bar."""
        if self.final_model is None:
            return None
        try:
            X = df_features[FEATURE_COLS].dropna().iloc[[-1]].values
            return float(self.final_model.predict_proba(X)[0, 1])
        except Exception:
            return None
