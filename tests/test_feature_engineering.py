"""Unit tests for feature engineering, indicators, and target alignment."""

import numpy as np
import pandas as pd
import pytest

from crypto_prediction.data.cleaner import DataCleaner
from crypto_prediction.features.engineer import FeatureEngineer
from crypto_prediction.features.registry import FeatureRegistry


class TestFeatureEngineering:
    """Verifies feature computations, shapes, and target alignment."""

    def test_indicator_computations_and_bounds(self, sample_ohlcv_df: pd.DataFrame) -> None:
        cleaner = DataCleaner()
        clean_df = cleaner.clean(sample_ohlcv_df)

        engineer = FeatureEngineer(
            return_lags=[1, 2, 4],
            rolling_windows=[6, 12, 24],
            rsi_periods=[7, 14],
            forecast_horizons=[1, 4],
        )

        features = engineer.compute_features(clean_df)

        # Check required categories exist
        assert "return_1h" in features.columns
        assert "return_lag_1" in features.columns
        assert "volatility_6h" in features.columns
        assert "rsi_14" in features.columns
        assert "macd_line" in features.columns
        assert "bb_percent_b" in features.columns
        assert "atr_normalized_14" in features.columns
        assert "hour_sin" in features.columns
        assert "day_cos" in features.columns

        # Verify RSI mathematical bounds: [0, 100]
        valid_rsi = features["rsi_14"].dropna()
        assert (valid_rsi >= 0.0).all()
        assert (valid_rsi <= 100.0).all()

    def test_target_alignment_exact_match(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Verifies forward return target at row t is strictly (Close_{t+h} - Close_t) / Close_t."""
        cleaner = DataCleaner()
        clean_df = cleaner.clean(sample_ohlcv_df)

        engineer = FeatureEngineer(forecast_horizons=[1, 4], primary_horizon=1)
        targets = engineer.create_targets(clean_df)

        close = clean_df["close"]

        # Horizon 1 check for several random indices
        for idx in [50, 100, 200]:
            expected_ret_1 = (close.iloc[idx + 1] - close.iloc[idx]) / close.iloc[idx]
            actual_ret_1 = targets["target_return_1"].iloc[idx]
            assert np.isclose(expected_ret_1, actual_ret_1, atol=1e-8)

            expected_price_1 = close.iloc[idx + 1]
            actual_price_1 = targets["target_price_1"].iloc[idx]
            assert np.isclose(expected_price_1, actual_price_1, atol=1e-8)

        # Horizon 4 check
        for idx in [50, 100, 200]:
            expected_ret_4 = (close.iloc[idx + 4] - close.iloc[idx]) / close.iloc[idx]
            actual_ret_4 = targets["target_return_4"].iloc[idx]
            assert np.isclose(expected_ret_4, actual_ret_4, atol=1e-8)

    def test_prepare_dataset_shapes_and_warmup_drop(self, sample_ohlcv_df: pd.DataFrame) -> None:
        cleaner = DataCleaner()
        clean_df = cleaner.clean(sample_ohlcv_df)

        engineer = FeatureEngineer(
            return_lags=[1, 2],
            rolling_windows=[6, 12, 24],
            forecast_horizons=[1, 4],
            primary_horizon=1,
        )

        X, y, metadata = engineer.prepare_dataset(clean_df, target_horizon=1, drop_warmup=True)

        assert len(X) == len(y)
        assert len(X) == len(metadata)
        assert not X.isna().any().any()
        assert not y.isna().any()
        assert "target_return_1" not in X.columns
        assert "target_price_1" not in X.columns
        assert "close" in metadata.columns

    def test_feature_registry_metadata_coverage(self) -> None:
        registry = FeatureRegistry()
        features = registry.all_features()
        assert len(features) > 10

        for f in features:
            assert f.name != ""
            assert f.category != ""
            assert f.formula != ""
            assert f.causality_proof != ""
