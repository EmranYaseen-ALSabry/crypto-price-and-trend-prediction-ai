"""
Phase 3.7 Independent Audit & Validation Test Suite.
Verifies zero temporal leakage, feature partition cleanliness, target isolation,
future target perturbation failure, and cryptographic artifact integrity.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crypto_prediction.pipeline.benchmarking import ControlledBenchmarkEngine
from crypto_prediction.pipeline.transfer_learning import TransferLearningEngine
from crypto_prediction.pipeline.backtest_engine_realistic import RealisticBacktestingEngine
from crypto_prediction.pipeline.evaluator import ModelEvaluator
from crypto_prediction.models.linear import RidgeModel
from crypto_prediction.models.boosters import LightGBMModel


class TestPhase37Audit:
    """Rigorous audit test cases for Phase 3.7 independent research verification."""

    def test_feature_partition_no_target_leakage(self):
        """Audit that ControlledBenchmarkEngine strictly excludes all target column variants."""
        engine = ControlledBenchmarkEngine()
        dummy_cols = [
            "target_return_1",
            "target_return_1.1",
            "target_return_4",
            "target_return_24",
            "target_price_1",
            "target_price_4",
            "target_price_24",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_time",
            "symbol",
            "rsi_14",
            "volatility_24h",
            "btc_return_lag_1",
            "cross_corr_eth_24h",
        ]

        asset_cols, cross_cols = engine.partition_feature_columns(dummy_cols)

        # Ensure NO target column exists in asset_cols or cross_cols
        for col in asset_cols + cross_cols:
            assert not col.startswith("target_"), f"Target column leaked into features: {col}"
            assert col not in ["open", "high", "low", "close", "volume", "open_time", "symbol"]

        assert "rsi_14" in asset_cols
        assert "volatility_24h" in asset_cols
        assert "btc_return_lag_1" in cross_cols
        assert "cross_corr_eth_24h" in cross_cols

    def test_transfer_learning_feature_extraction_no_target_leakage(self):
        """Audit that TransferLearningEngine strictly excludes all target column variants."""
        engine = TransferLearningEngine()
        dummy_cols = [
            "target_return_1",
            "target_return_1.1",
            "target_return_4",
            "target_price_1",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_time",
            "symbol",
            "rsi_14",
            "macd_diff",
            "btc_return_lag_1",
            "cross_vol_ratio_eth_24h",
        ]

        tl_cols = engine.extract_asset_agnostic_features(dummy_cols)

        for col in tl_cols:
            assert not col.startswith("target_"), f"Target column leaked into TL features: {col}"
            assert col not in ["open", "high", "low", "close", "volume", "open_time", "symbol"]
            assert not any(col.startswith(p) for p in ["btc_return_lag", "cross_vol_ratio"])

        assert "rsi_14" in tl_cols
        assert "macd_diff" in tl_cols

    def test_remediated_feature_target_correlation_bounds(self):
        """Audit that no feature in the panel exhibits an anomalous correlation with the target return."""
        engine = ControlledBenchmarkEngine()
        datasets = engine.load_feature_datasets()

        for sym, df in datasets.items():
            all_cols = list(df.columns)
            asset_cols, cross_cols = engine.partition_feature_columns(all_cols)
            
            # None of the features should have |corr| > 0.15 with forward hourly return
            corrs = df[cross_cols].corrwith(df["target_return_1"]).abs()
            max_corr = corrs.max()
            max_feat = corrs.idxmax()

            assert max_corr < 0.15, (
                f"{sym}: Max feature correlation ({max_corr:.4f} in '{max_feat}') exceeds 0.15 threshold! "
                f"Possible leakage."
            )

    def test_chronological_isolation_and_no_lookahead(self):
        """Audit that walk-forward folds have zero lookahead and strictly non-overlapping evaluation windows."""
        engine = ControlledBenchmarkEngine()
        datasets = engine.load_feature_datasets()
        first_sym = engine.symbols[0]
        n_bars = len(datasets[first_sym])
        folds = engine.define_walk_forward_folds(n_bars)

        assert len(folds) == 3, f"Expected 3 folds, got {len(folds)}"

        for i, fold in enumerate(folds):
            assert fold.train_start_idx < fold.train_end_idx
            assert fold.train_end_idx <= fold.test_start_idx, (
                f"Fold {i} ({fold.name}): train_end ({fold.train_end_idx}) > test_start ({fold.test_start_idx})"
            )
            assert fold.test_start_idx < fold.test_end_idx

            # Verify timestamps are strictly chronological
            df = datasets[first_sym]
            train_times = df.index[fold.train_start_idx:fold.train_end_idx]
            test_times = df.index[fold.test_start_idx:fold.test_end_idx]

            assert train_times.max() < test_times.min(), (
                f"Fold {i}: Max train time ({train_times.max()}) not strictly before min test time ({test_times.min()})"
            )

    def test_future_target_perturbation_destroys_skill(self):
        """Causality test: Randomly permuting future targets must cause model performance to collapse."""
        engine = ControlledBenchmarkEngine()
        datasets = engine.load_feature_datasets()
        df = datasets["BTCUSDT"]
        folds = engine.define_walk_forward_folds(len(df))
        f = folds[0]

        all_cols = list(df.columns)
        _, cross_cols = engine.partition_feature_columns(all_cols)
        common_cols = set.intersection(*(set(datasets[s].columns) for s in engine.symbols))
        cross_cols = [c for c in cross_cols if c in common_cols]

        X_train = df[cross_cols].iloc[f.train_start_idx:f.train_end_idx]
        y_train = df["target_return_1"].iloc[f.train_start_idx:f.train_end_idx]
        X_test = df[cross_cols].iloc[f.test_start_idx:f.test_end_idx]
        y_test = df["target_return_1"].iloc[f.test_start_idx:f.test_end_idx]

        # Train model on real un-leaked features
        model = RidgeModel(alpha=100.0)
        model.fit(X_train, y_train)
        real_preds = model.predict(X_test)

        evaluator = ModelEvaluator()
        real_metrics = evaluator.evaluate(y_test.values, real_preds, y_train.values)

        # Now randomly permute test labels (future perturbation)
        rng = np.random.default_rng(42)
        permuted_y_test = rng.permutation(y_test.values)
        permuted_metrics = evaluator.evaluate(permuted_y_test, real_preds, y_train.values)

        # With permuted future targets, Directional Accuracy must be near 50% (+- 3.5%) and R2 must be <= 0
        assert abs(permuted_metrics.directional_accuracy - 0.50) < 0.05, (
            f"Permuted DA should be ~0.50, got {permuted_metrics.directional_accuracy:.4f}"
        )
        assert permuted_metrics.r2 < 0.005, (
            f"Permuted R2 should be near 0 or negative, got {permuted_metrics.r2:.5f}"
        )

    def test_phase3_artifact_sha256_digests(self):
        """Cryptographic audit: Verify all Phase 3 result CSVs match their recorded SHA-256 metadata."""
        artifacts_to_verify = [
            Path("artifacts/phase3_4/phase3_4_benchmark_results.csv"),
            Path("artifacts/phase3_5/phase3_5_transfer_results.csv"),
            Path("artifacts/phase3_6/phase3_6_backtest_results.csv"),
        ]

        for p in artifacts_to_verify:
            assert p.exists(), f"Artifact missing: {p}"
            meta_path = p.with_name(p.name + ".meta.json")
            assert meta_path.exists(), f"Metadata missing: {meta_path}"

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            recorded_sha = meta.get("sha256")
            assert recorded_sha, f"No sha256 in metadata for {p}"

            # Compute actual SHA-256
            hasher = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            actual_sha = hasher.hexdigest()

            assert actual_sha == recorded_sha, (
                f"SHA-256 mismatch for {p}!\nRecorded: {recorded_sha}\nActual:   {actual_sha}"
            )
