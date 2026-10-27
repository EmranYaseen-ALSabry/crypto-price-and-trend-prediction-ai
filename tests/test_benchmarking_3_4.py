"""Unit tests for Phase 3.4 Controlled Multi-Asset Model Benchmarking Suite."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crypto_prediction.pipeline.benchmarking import (
    ControlledBenchmarkEngine,
    WalkForwardFoldSpec,
)


def test_walk_forward_fold_spec_and_boundaries() -> None:
    """Verifies that walk-forward folds satisfy chronological ordering, purge gaps, and holdout isolation."""
    engine = ControlledBenchmarkEngine()
    total_bars = 33324
    folds = engine.define_walk_forward_folds(total_bars)

    assert len(folds) == 3, f"Expected 3 folds, got {len(folds)}"

    for i, fold in enumerate(folds):
        assert isinstance(fold, WalkForwardFoldSpec)
        assert fold.train_start_idx == 0
        assert fold.train_end_idx > fold.train_start_idx
        # Purge gap must be at least 24 bars to prevent target overlap leakage
        assert fold.purge_bars >= 24, f"Purge bars too small in {fold.name}: {fold.purge_bars}"
        assert fold.test_start_idx >= fold.train_end_idx + fold.purge_bars, (
            f"Leakage detected in {fold.name}: test starts at {fold.test_start_idx} "
            f"before train_end {fold.train_end_idx} + purge {fold.purge_bars}"
        )
        assert fold.test_end_idx > fold.test_start_idx
        # Test size should be 2000 bars
        assert fold.test_end_idx - fold.test_start_idx == 2000

    # Ensure train windows strictly expand chronologically
    assert folds[0].train_end_idx < folds[1].train_end_idx < folds[2].train_end_idx

    # Ensure the final holdout of ~7,300 bars remains strictly untouched
    max_test_idx = folds[-1].test_end_idx
    untouched_bars = total_bars - max_test_idx
    assert untouched_bars >= 7000, f"Final holdout violated! Remaining bars: {untouched_bars}"


def test_feature_partitioning() -> None:
    """Verifies that asset-specific and cross-asset features are properly separated."""
    all_cols = [
        "target_return_1",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
        "return_1h",
        "return_lag_1",
        "volatility_24h",
        "rsi_14",
        "macd",
        "btc_return_lag_1",
        "eth_return_lag_1",
        "sol_return_lag_1",
        "cross_corr_eth_24h",
        "cross_vol_ratio_sol_24h",
        "cross_mom_spread_eth_1h",
        "market_mean_return_lag_1",
        "market_return_dispersion_lag_1",
        "market_mean_volatility_24h",
    ]

    asset_cols, cross_cols = ControlledBenchmarkEngine.partition_feature_columns(all_cols)

    # Metadata and target must be excluded
    for excluded in ["target_return_1", "open", "high", "low", "close", "volume", "symbol"]:
        assert excluded not in asset_cols
        assert excluded not in cross_cols

    # Asset specific must NOT contain cross prefixes
    cross_prefixes = (
        "btc_return_lag",
        "eth_return_lag",
        "sol_return_lag",
        "cross_vol_ratio",
        "cross_corr",
        "cross_mom_spread",
        "market_mean_return",
        "market_return_dispersion",
        "market_mean_volatility",
    )
    for col in asset_cols:
        assert not col.startswith(cross_prefixes), f"Cross prefix found in asset_cols: {col}"

    assert "rsi_14" in asset_cols
    assert "volatility_24h" in asset_cols
    assert "return_lag_1" in asset_cols

    # Cross cols must contain all asset cols plus the cross features
    for col in asset_cols:
        assert col in cross_cols

    assert "btc_return_lag_1" in cross_cols
    assert "cross_corr_eth_24h" in cross_cols
    assert "market_mean_volatility_24h" in cross_cols


def test_metric_calculations() -> None:
    """Verifies precision and correctness of RMSE, MAE, OOS R2, and directional accuracy."""
    y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    y_train_mean = 0.005

    # 1. Perfect prediction
    metrics_perf = ControlledBenchmarkEngine.calculate_metrics(y_true, y_true, y_train_mean)
    assert metrics_perf["rmse"] == pytest.approx(0.0)
    assert metrics_perf["mae"] == pytest.approx(0.0)
    assert metrics_perf["r2_oos"] == pytest.approx(1.0)
    assert metrics_perf["directional_accuracy"] == pytest.approx(1.0)
    assert metrics_perf["coverage"] == pytest.approx(1.0)

    # 2. Constant zero prediction (ZeroReturn Baseline)
    y_zero = np.zeros_like(y_true)
    metrics_zero = ControlledBenchmarkEngine.calculate_metrics(y_true, y_zero, y_train_mean)
    assert metrics_zero["r2_zero"] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(metrics_zero["directional_accuracy"])
    assert metrics_zero["coverage"] == pytest.approx(0.0)
    assert metrics_zero["rmse"] == pytest.approx(np.sqrt(np.mean(y_true**2)))

    # 3. Completely inverted predictions
    y_inv = -1.0 * y_true
    metrics_inv = ControlledBenchmarkEngine.calculate_metrics(y_true, y_inv, y_train_mean)
    assert metrics_inv["directional_accuracy"] == pytest.approx(0.0)
    assert metrics_inv["coverage"] == pytest.approx(1.0)


def test_sample_weight_inverse_volatility() -> None:
    """Verifies that high-volatility samples receive lower weights."""
    vol_low = 0.01
    vol_high = 0.04
    w_low = 1.0 / (vol_low**2)
    w_high = 1.0 / (vol_high**2)
    assert w_low > w_high
    assert pytest.approx(w_low / w_high) == 16.0


def test_mini_benchmark_execution(tmp_path: Path) -> None:
    """Executes a fast end-to-end mini benchmark on synthetic features to verify pipeline wiring."""
    n_bars = 400
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")

    # Generate synthetic features for BTC and ETH
    np.random.seed(42)
    features_dir = tmp_path / "features"
    artifacts_dir = tmp_path / "artifacts"
    features_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    for sym in ["BTCUSDT", "ETHUSDT"]:
        df = pd.DataFrame(
            {
                "open_time": dates,
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 101.0,
                "volume": 1000.0,
                "symbol": sym,
                "target_return_1": np.random.normal(0.0001, 0.01, n_bars),
                "return_1h": np.random.normal(0.0001, 0.01, n_bars),
                "return_lag_1": np.random.normal(0.0001, 0.01, n_bars),
                "volatility_24h": np.full(n_bars, 0.02 if sym == "BTCUSDT" else 0.04),
                "rsi_14": np.random.uniform(30, 70, n_bars),
                "btc_return_lag_1": np.random.normal(0.0001, 0.01, n_bars),
                "eth_return_lag_1": np.random.normal(0.0001, 0.01, n_bars),
                "market_mean_return_lag_1": np.random.normal(0.0001, 0.01, n_bars),
                "market_mean_volatility_24h": np.full(n_bars, 0.03),
            }
        )
        if sym == "BTCUSDT":
            df["cross_corr_eth_24h"] = 0.8
            df["cross_vol_ratio_eth_24h"] = 0.5
            df["cross_mom_spread_eth_1h"] = 0.001
        else:
            df["cross_corr_btc_24h"] = 0.8
            df["cross_vol_ratio_btc_24h"] = 2.0
            df["cross_mom_spread_btc_1h"] = -0.001

        df.to_csv(features_dir / f"{sym}_1h_features.csv", index=False)

    class MiniBenchmarkEngine(ControlledBenchmarkEngine):
        def define_walk_forward_folds(self, total_bars: int) -> list[WalkForwardFoldSpec]:
            return [
                WalkForwardFoldSpec(
                    fold_id=1,
                    name="Mini Fold 1",
                    train_start_idx=0,
                    train_end_idx=200,
                    purge_bars=24,
                    test_start_idx=224,
                    test_end_idx=300,
                )
            ]

    engine = MiniBenchmarkEngine(
        features_dir=features_dir,
        artifacts_dir=artifacts_dir,
        symbols=["BTCUSDT", "ETHUSDT"],
        random_seed=42,
    )

    results = engine.run_benchmark(
        approaches=["A_single_asset_baseline", "C_multi_asset_baseline"],
        models=["ZeroReturn", "Ridge"],
        use_sample_weighting=True,
    )

    assert not results.empty
    assert len(results) == 1 * 2 * 2 * 2  # 1 fold * 2 approaches * 2 models * 2 symbols = 8 records
    expected_cols = [
        "fold_id",
        "fold_name",
        "approach",
        "model_family",
        "symbol",
        "rmse",
        "mae",
        "r2_oos",
        "directional_accuracy",
        "improvement_vs_zero_pct",
        "improvement_vs_approach_a_pct",
    ]
    for col in expected_cols:
        assert col in results.columns

    # Verify that artifact was created
    assert (artifacts_dir / "phase3_4_benchmark_results.csv").exists()
    assert (artifacts_dir / "phase3_4_benchmark_results.csv.meta.json").exists()
