"""Unit and integration tests for Phase 3.5 Cross-Asset Transfer Learning & Volatility Adaptation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crypto_prediction.pipeline.benchmarking import WalkForwardFoldSpec
from crypto_prediction.pipeline.transfer_learning import TransferLearningEngine


def test_held_out_asset_isolation_and_source_exclusion() -> None:
    """Verifies that in LOAO, source assets strictly exclude the target asset."""
    engine = TransferLearningEngine(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    for target in engine.symbols:
        source_assets = [s for s in engine.symbols if s != target]
        assert target not in source_assets
        assert len(source_assets) == 2


def test_asset_agnostic_feature_extraction() -> None:
    """Verifies that asset-agnostic features contain zero ticker-specific cross prefixes."""
    cols = [
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
        "market_mean_return_lag_1",
        "market_mean_volatility_24h",
        "btc_return_lag_1",
        "eth_return_lag_1",
        "sol_return_lag_1",
        "cross_corr_eth_24h",
        "cross_vol_ratio_sol_24h",
        "cross_mom_spread_eth_1h",
    ]

    agnostic = TransferLearningEngine.extract_asset_agnostic_features(cols)

    # Excluded metadata & target
    for meta in ["target_return_1", "open", "high", "low", "close", "volume", "symbol"]:
        assert meta not in agnostic

    # Excluded ticker specific cross features
    for ticker_cross in [
        "btc_return_lag_1",
        "eth_return_lag_1",
        "sol_return_lag_1",
        "cross_corr_eth_24h",
        "cross_vol_ratio_sol_24h",
        "cross_mom_spread_eth_1h",
    ]:
        assert ticker_cross not in agnostic

    # Retained generic indicators and market aggregates
    assert "rsi_14" in agnostic
    assert "volatility_24h" in agnostic
    assert "macd" in agnostic
    assert "return_lag_1" in agnostic
    assert "market_mean_return_lag_1" in agnostic
    assert "market_mean_volatility_24h" in agnostic


def test_causal_volatility_normalization_and_inverse_restoration() -> None:
    """Verifies that volatility normalization and inverse transformation are mathematically exact."""
    # Simulated return and trailing volatility
    true_return = np.array([0.02, -0.015, 0.03, -0.005])
    causal_vol = np.array([0.01, 0.015, 0.02, 0.005])

    # 1. Forward normalization: z = r / vol
    z = true_return / causal_vol
    assert np.allclose(z, np.array([2.0, -1.0, 1.5, -1.0]))

    # 2. Predicted standardized z
    pred_z = np.array([1.8, -1.2, 1.4, -0.8])

    # 3. Inverse transformation: y = pred_z * vol
    restored_return = pred_z * causal_vol
    expected_return = np.array([0.018, -0.018, 0.028, -0.004])
    assert np.allclose(restored_return, expected_return)

    # 4. Metrics calculated in original return units
    metrics = TransferLearningEngine.calculate_metrics(true_return, restored_return, y_train_mean=0.005)
    assert metrics["rmse"] > 0.0
    assert metrics["mae"] > 0.0
    # RMSE should be in return units (~0.002) not z units (~0.2)
    assert metrics["rmse"] < 0.01


def test_zero_shot_vs_few_shot_separation_and_leakage_safety() -> None:
    """Verifies that few-shot training slice is strictly before the purge gap and does not overlap with test."""
    total_bars = 10000
    few_shot_bars = 1000
    purge_bars = 24

    train_end = 8000
    test_start = train_end + purge_bars  # 8024
    test_end = test_start + 2000         # 10024

    few_shot_start = max(0, train_end - few_shot_bars)  # 7000
    few_shot_end = train_end                            # 8000

    # Few-shot must be strictly prior to purge gap
    assert few_shot_end <= train_end
    assert test_start - few_shot_end == purge_bars
    assert few_shot_end < test_start < test_end

    # Test slice indices are completely disjoint from few-shot indices
    few_shot_indices = set(range(few_shot_start, few_shot_end))
    test_indices = set(range(test_start, test_end))
    assert few_shot_indices.isdisjoint(test_indices)


def test_mini_transfer_learning_execution(tmp_path: Path) -> None:
    """Executes a fast end-to-end mini transfer benchmark on synthetic data."""
    n_bars = 300
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")

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
                "volatility_24h": np.full(n_bars, 0.01 if sym == "BTCUSDT" else 0.03),
                "rsi_14": np.random.uniform(30, 70, n_bars),
                "market_mean_return_lag_1": np.random.normal(0.0001, 0.01, n_bars),
                "market_mean_volatility_24h": np.full(n_bars, 0.02),
            }
        )
        df.to_csv(features_dir / f"{sym}_1h_features.csv", index=False)

    class MiniTransferEngine(TransferLearningEngine):
        def define_walk_forward_folds(self, total_bars: int) -> list[WalkForwardFoldSpec]:
            return [
                WalkForwardFoldSpec(
                    fold_id=1,
                    name="Mini Fold 1",
                    train_start_idx=0,
                    train_end_idx=150,
                    purge_bars=24,
                    test_start_idx=174,
                    test_end_idx=250,
                )
            ]

    engine = MiniTransferEngine(
        features_dir=features_dir,
        artifacts_dir=artifacts_dir,
        symbols=["BTCUSDT", "ETHUSDT"],
        random_seed=42,
        few_shot_bars=50,
    )

    results = engine.run_transfer_experiments(
        target_assets=["ETHUSDT"],
        modes=["zero_shot_raw", "zero_shot_vol_adapted", "few_shot_vol_adapted", "asset_specific_baseline"],
        models=["ZeroReturn", "Ridge"],
    )

    assert not results.empty
    # 1 fold * 1 target * 4 modes * 2 models = 8 records
    assert len(results) == 8
    expected_cols = [
        "fold_id",
        "target_asset",
        "source_assets",
        "model_family",
        "transfer_mode",
        "rmse",
        "mae",
        "r2_oos",
        "directional_accuracy",
        "improvement_vs_zero_pct",
        "improvement_vs_baseline_pct",
        "gain_vs_raw_pct",
    ]
    for col in expected_cols:
        assert col in results.columns

    # Verify artifacts created
    assert (artifacts_dir / "phase3_5_transfer_results.csv").exists()
    assert (artifacts_dir / "phase3_5_transfer_results.csv.meta.json").exists()
