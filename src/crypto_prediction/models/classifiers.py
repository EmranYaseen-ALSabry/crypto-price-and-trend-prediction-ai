"""Direction classifiers: predict P(up) for the forward return, ignoring moves smaller than trading costs.

The regression target (forward return) is tiny and noisy. Classifying direction on moves that are
large enough to matter (|return| > threshold, default = fees + slippage) is a better-posed problem and
maps directly to a trading decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_THRESHOLD = 0.0015  # fee 0.10% + slippage 0.05%, matches StrategyBacktester defaults


def make_direction_labels(y: np.ndarray | pd.Series, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Returns 1 for r > +threshold, 0 for r < -threshold, and -1 for the neutral band (excluded)."""
    r = np.asarray(y, dtype=np.float64)
    labels = np.full(r.shape, -1, dtype=np.int64)
    labels[r > threshold] = 1
    labels[r < -threshold] = 0
    return labels


class DirectionClassifier:
    """Wraps a sklearn classifier so it trains on return targets and outputs P(up)."""

    def __init__(self, name: str, estimator, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.name = name
        self.estimator = estimator
        self.threshold = threshold
        self.is_fitted = False
        self._constant_p: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y_return: pd.Series | np.ndarray) -> "DirectionClassifier":
        labels = make_direction_labels(y_return, self.threshold)
        keep = labels >= 0
        X_arr = np.asarray(X, dtype=np.float64)[keep]
        y_lab = labels[keep]
        if len(np.unique(y_lab)) < 2:
            # Degenerate window (only one class): fall back to the observed frequency.
            self._constant_p = float(y_lab.mean()) if len(y_lab) else 0.5
        else:
            self._constant_p = None
            self.estimator.fit(X_arr, y_lab)
        self.is_fitted = True
        return self

    def predict_proba_up(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Classifier is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        if self._constant_p is not None:
            return np.full(len(X_arr), self._constant_p)
        return self.estimator.predict_proba(X_arr)[:, 1]

    def predict_signal(self, X: pd.DataFrame | np.ndarray, p_long: float = 0.55) -> np.ndarray:
        """1 = long, 0 = stay in cash (spot, long-only)."""
        return (self.predict_proba_up(X) >= p_long).astype(np.int64)


def default_classifier_factories(seed: int = 42, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Name -> zero-argument factory returning a fresh, unfitted DirectionClassifier."""
    return {
        "LogisticRegression": lambda: DirectionClassifier(
            "LogisticRegression",
            make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500, random_state=seed)),
            threshold,
        ),
        "RandomForestClf": lambda: DirectionClassifier(
            "RandomForestClf",
            RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=20, random_state=seed, n_jobs=-1),
            threshold,
        ),
        "HistGradientBoostingClf": lambda: DirectionClassifier(
            "HistGradientBoostingClf",
            HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3,
                                           min_samples_leaf=30, l2_regularization=1.0, random_state=seed),
            threshold,
        ),
    }
