"""Automated test suite specifically targeting data leakage, causality, and purging."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from crypto_prediction.data.cleaner import DataCleaner
from crypto_prediction.features.engineer import FeatureEngineer
from crypto_prediction.validation.leakage_guard import LeakageGuard
from crypto_prediction.validation.splitter import ChronologicalSplitter, WalkForwardSplitter


class TestTemporalLeakagePrevention:
    """Verifies that no future data leaks into the past under any condition."""

    def test_future_price_perturbation_invariance(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Modifying future prices must NOT alter past features."""
        cleaner = DataCleaner()
        clean_df = cleaner.clean(sample_ohlcv_df)

        engineer = FeatureEngineer(
            return_lags=[1, 2, 4, 12, 24],
            rolling_windows=[6, 12, 24],
            rsi_periods=[7, 14],
        )

        cutoff = 300
        passed, max_diff, msg = LeakageGuard.run_perturbation_test(
            df_raw=clean_df,
            feature_engineer=engineer,
            cutoff_idx=cutoff,
            perturbation_factor=3.5,
        )

        assert passed is True, f"Leakage detected! {msg}"
        assert max_diff < 1e-10

    def test_target_isolation_from_model_features(self, sample_ohlcv_df: pd.DataFrame) -> None:
        cleaner = DataCleaner()
        clean_df = cleaner.clean(sample_ohlcv_df)

        engineer = FeatureEngineer(forecast_horizons=[1, 4, 24], primary_horizon=1)
        X, y, _ = engineer.prepare_dataset(clean_df)

        passed, msg = LeakageGuard.verify_target_isolation(list(X.columns))
        assert passed is True, msg
        assert "target" not in "".join(X.columns).lower()

    def test_chronological_splits_strictly_ordered(self, sample_ohlcv_df: pd.DataFrame) -> None:
        cleaner = DataCleaner()
        clean_df = cleaner.clean(sample_ohlcv_df)

        engineer = FeatureEngineer()
        X, y, meta = engineer.prepare_dataset(clean_df)

        splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4)
        splits = splitter.split(X, y, metadata=meta)

        # 1. Monotonic timestamps within partitions
        assert splits.X_train.index.is_monotonic_increasing
        assert splits.X_val.index.is_monotonic_increasing
        assert splits.X_test.index.is_monotonic_increasing

        # 2. Strict chronological separation: train < val < test
        assert splits.X_train.index[-1] < splits.X_val.index[0]
        assert splits.X_val.index[-1] < splits.X_test.index[0]

        # 3. Purging gap test: index difference >= purge_bars
        train_end_loc = X.index.get_loc(splits.X_train.index[-1])
        val_start_loc = X.index.get_loc(splits.X_val.index[0])
        assert (val_start_loc - train_end_loc) > 4

    def test_walk_forward_cv_has_zero_lookahead(self, sample_ohlcv_df: pd.DataFrame) -> None:
        cleaner = DataCleaner()
        clean_df = cleaner.clean(sample_ohlcv_df)

        engineer = FeatureEngineer()
        X, y, _ = engineer.prepare_dataset(clean_df)

        wf_splitter = WalkForwardSplitter(
            n_splits=3,
            train_window_bars=200,
            val_window_bars=50,
            step_bars=40,
            purge_bars=4,
        )

        for train_idx, val_idx in wf_splitter.split(X):
            # All training indices must strictly precede validation indices with at least purge_bars gap
            assert np.max(train_idx) < np.min(val_idx)
            gap = np.min(val_idx) - np.max(train_idx)
            assert gap > 4

    def test_preprocessing_isolation(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Verifies that standardizing features on train does not peek into test distribution."""
        cleaner = DataCleaner()
        clean_df = cleaner.clean(sample_ohlcv_df)

        engineer = FeatureEngineer()
        X, y, meta = engineer.prepare_dataset(clean_df)

        splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=1)
        splits = splitter.split(X, y, metadata=meta)

        scaler = StandardScaler()
        # Fit ONLY on train
        scaler.fit(splits.X_train)

        # Mean and var should be computed solely from train
        train_mean = splits.X_train.mean().to_numpy()
        assert np.allclose(scaler.mean_, train_mean, atol=1e-5)

        # Transform val and test without fitting
        scaled_val = scaler.transform(splits.X_val)
        scaled_test = scaler.transform(splits.X_test)

        assert not np.isnan(scaled_val).any()
        assert not np.isnan(scaled_test).any()

    def test_purging_gap_audit_detects_insufficient_gap(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Verifies that audit_splits actually flags when purging gap is smaller than required_purge_bars."""
        cleaner = DataCleaner()
        clean_df = cleaner.clean(sample_ohlcv_df)

        engineer = FeatureEngineer()
        X, y, meta = engineer.prepare_dataset(clean_df)

        # Split with 0 purge bars
        splitter_no_purge = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=0)
        splits_no_purge = splitter_no_purge.split(X, y, metadata=meta)

        # Audit expecting 4 purge bars -> must fail purging_passed
        mono_pass, purge_pass, issues = LeakageGuard.audit_splits(
            splits_no_purge.X_train.index,
            splits_no_purge.X_val.index,
            splits_no_purge.X_test.index,
            required_purge_bars=4,
            full_index=X.index,
        )
        assert mono_pass is True
        assert purge_pass is False
        assert any("less than required" in issue for issue in issues)

        # Split with 4 purge bars -> must pass purging_passed
        splitter_purged = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4)
        splits_purged = splitter_purged.split(X, y, metadata=meta)

        mono_pass_p, purge_pass_p, issues_p = LeakageGuard.audit_splits(
            splits_purged.X_train.index,
            splits_purged.X_val.index,
            splits_purged.X_test.index,
            required_purge_bars=4,
            full_index=X.index,
        )
        assert mono_pass_p is True
        assert purge_pass_p is True
        assert len(issues_p) == 0
