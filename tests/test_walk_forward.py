"""Tests for walk-forward folds, direction classifiers and the trading simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_prediction.models.classifiers import DirectionClassifier, default_classifier_factories, make_direction_labels
from crypto_prediction.models.baselines import MeanReturnBaseline
from crypto_prediction.models.linear import RidgeModel
from crypto_prediction.pipeline.walk_forward import (
    make_folds,
    run_walk_forward_classification,
    run_walk_forward_regression,
    run_walk_forward_trading,
)


def _frame(n: int = 3000, signal: float = 0.0, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.normal(0, 0.004, n), index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
    X = pd.DataFrame({
        "noise_a": rng.normal(size=n),
        "noise_b": rng.normal(size=n),
        "driver": y.to_numpy() * signal / 0.004 + rng.normal(size=n),
    }, index=y.index)
    return X, y


class TestFolds:
    def test_folds_are_disjoint_ordered_and_purged(self) -> None:
        folds = list(make_folds(n=3000, n_folds=4, test_window=300, purge_bars=5, min_train=500))
        assert len(folds) == 4
        prev_end = -1
        for tr, te in folds:
            assert tr.max() < te.min() - 5 + 1  # purge gap respected (train ends >= purge before test)
            assert te.min() > prev_end  # test windows never overlap and move forward
            prev_end = te.max()
        assert folds[-1][1].max() == 2999

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(ValueError):
            list(make_folds(n=1000, n_folds=6, test_window=500, purge_bars=1, min_train=2000))


class TestDirectionLabels:
    def test_neutral_band_excluded(self) -> None:
        labels = make_direction_labels([0.01, -0.01, 0.0005, -0.0005], threshold=0.0015)
        assert labels.tolist() == [1, 0, -1, -1]

    def test_single_class_window_falls_back_to_constant(self) -> None:
        from sklearn.linear_model import LogisticRegression

        clf = DirectionClassifier("x", LogisticRegression(), threshold=0.001)
        X = np.random.default_rng(0).normal(size=(50, 3))
        clf.fit(X, np.full(50, 0.01))  # only "up" moves
        assert np.allclose(clf.predict_proba_up(X), 1.0)


class TestWalkForwardSkill:
    def test_noise_shows_no_skill(self) -> None:
        X, y = _frame(signal=0.0)
        _, summary = run_walk_forward_regression(
            X, y, {"Ridge": lambda: RidgeModel(alpha=1.0), "Mean": lambda: MeanReturnBaseline()},
            n_folds=4, test_window=300, min_train=1000,
        )
        ridge = summary.set_index("model_name").loc["Ridge"]
        assert ridge["rmse_skill_mean"] < 0.01  # no meaningful skill on pure noise

    def test_planted_signal_is_detected(self) -> None:
        X, y = _frame(signal=0.5)
        _, summary = run_walk_forward_classification(
            X, y, default_classifier_factories(threshold=0.0015), n_folds=4, test_window=300, min_train=1000,
        )
        assert summary["auc_mean"].max() > 0.6

    def test_trading_output_has_reliability_flag(self) -> None:
        X, y = _frame(signal=0.5)
        trading = run_walk_forward_trading(
            X, y, default_classifier_factories(), n_folds=4, test_window=300, min_train=1000, min_trades=30,
        )
        assert {"total_trades", "cumulative_return", "statistically_reliable"} <= set(trading.columns)
        assert len(trading) == 3
