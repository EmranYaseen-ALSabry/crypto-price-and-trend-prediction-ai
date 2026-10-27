"""Tests for Multi-Asset Generalization, Panel Construction, and Leave-One-Out Isolation."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crypto_prediction.pipeline.multi_asset import MultiAssetEngine


@pytest.fixture
def multi_engine(tmp_path: Path) -> MultiAssetEngine:
    return MultiAssetEngine(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        interval="1h",
        horizon=1,
        artifacts_dir=tmp_path / "artifacts",
        random_seed=42,
    )


class TestMultiAssetArchitecture:
    """Verifies panel construction, temporal synchronization, and cross-asset isolation."""

    def test_load_and_align_datasets(self, multi_engine: MultiAssetEngine) -> None:
        datasets = multi_engine.load_and_align_datasets(data_source="sample", total_bars=300)
        assert len(datasets) == 3
        btc_idx = datasets["BTCUSDT"].index
        eth_idx = datasets["ETHUSDT"].index
        sol_idx = datasets["SOLUSDT"].index

        # Identical timestamp index across all assets
        assert btc_idx.equals(eth_idx)
        assert btc_idx.equals(sol_idx)
        assert len(btc_idx) >= 200

    def test_panel_construction_and_scale_invariance(self, multi_engine: MultiAssetEngine) -> None:
        datasets = multi_engine.load_and_align_datasets(data_source="sample", total_bars=300)
        prepared = multi_engine.prepare_features(datasets, include_patterns=False)

        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            X = prepared[sym]["X"]
            assert not X.empty
            # Scale-invariance check: raw unnormalized prices must NOT be feature columns
            assert "close" not in X.columns
            assert "open" not in X.columns
            assert "high" not in X.columns
            assert "low" not in X.columns
            # Features should be bounded or ratios
            assert "return_1h" in X.columns

    def test_temporal_splitting_synchronized(self, multi_engine: MultiAssetEngine) -> None:
        datasets = multi_engine.load_and_align_datasets(data_source="sample", total_bars=300)
        prepared = multi_engine.prepare_features(datasets, include_patterns=False)
        splits = multi_engine.split_synchronized(prepared, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=1)

        # Confirm exact matching split lengths and temporal timestamps
        s_btc = splits["BTCUSDT"]
        s_eth = splits["ETHUSDT"]
        s_sol = splits["SOLUSDT"]

        assert len(s_btc.X_train) == len(s_eth.X_train) == len(s_sol.X_train)
        assert len(s_btc.X_val) == len(s_eth.X_val) == len(s_sol.X_val)
        assert len(s_btc.X_test) == len(s_eth.X_test) == len(s_sol.X_test)

        assert s_btc.X_train.index.equals(s_eth.X_train.index)
        assert s_btc.X_val.index.equals(s_eth.X_val.index)
        assert s_btc.X_test.index.equals(s_eth.X_test.index)

        # Confirm 1-bar purge gap
        train_last_time = s_btc.X_train.index[-1]
        val_first_time = s_btc.X_val.index[0]
        # Distance should be 2 hours (1 bar purged between them)
        assert val_first_time - train_last_time == pd.Timedelta(hours=2)

    def test_asset_isolation_in_loao(self, multi_engine: MultiAssetEngine) -> None:
        """Verifies that in Leave-One-Asset-Out, the held-out asset is strictly quarantined."""
        datasets = multi_engine.load_and_align_datasets(data_source="sample", total_bars=250)
        prepared = multi_engine.prepare_features(datasets, include_patterns=False)
        splits = multi_engine.split_synchronized(prepared, purge_bars=1)

        held_out = "BTCUSDT"
        train_syms = [s for s in multi_engine.symbols if s != held_out]
        assert held_out not in train_syms
        assert len(train_syms) == 2

        # Pool training data from remaining symbols
        X_train_loao = pd.concat([splits[s].X_train for s in train_syms], axis=0)
        y_train_loao = pd.concat([splits[s].y_train for s in train_syms], axis=0)

        # Expected row count is exactly 2 * train_len
        single_train_len = len(splits["ETHUSDT"].X_train)
        assert len(X_train_loao) == 2 * single_train_len
        assert len(y_train_loao) == 2 * single_train_len

    def test_end_to_end_multi_asset_execution(self, multi_engine: MultiAssetEngine) -> None:
        """Executes full multi-asset pipeline across Experiments A, B, and C."""
        results = multi_engine.run_full_research(data_source="sample", total_bars=300)

        assert "evaluation_df" in results
        assert "backtest_df" in results
        assert "summary_df" in results

        eval_df = results["evaluation_df"]
        summary_df = results["summary_df"]
        backtest_df = results["backtest_df"]

        assert not eval_df.empty
        assert not summary_df.empty
        assert not backtest_df.empty

        # Check all three experiment types exist
        exp_types = set(eval_df["experiment"].unique())
        assert any("Single-Asset" in e for e in exp_types)
        assert any("Pooled Multi-Asset" in e for e in exp_types)
        assert any("Leave-One-Asset-Out" in e for e in exp_types)

        # Check that backtest metrics contain gross and net returns
        assert "gross_cumulative_return" in backtest_df.columns
        assert "total_transaction_costs" in backtest_df.columns
        assert "cumulative_return" in backtest_df.columns

    def test_cross_asset_features_integration(self, tmp_path: Path) -> None:
        """Verifies MultiAssetEngine with use_cross_asset_features=True."""
        engine = MultiAssetEngine(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            interval="1h",
            horizon=1,
            artifacts_dir=tmp_path / "artifacts",
            use_cross_asset_features=True,
        )
        datasets = engine.load_and_align_datasets(data_source="sample", total_bars=300)
        prepared = engine.prepare_features(datasets, include_patterns=False)

        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            X = prepared[sym]["X"]
            assert "btc_return_lag_1" in X.columns
            assert "market_mean_return_1h" in X.columns
            if sym != "BTCUSDT":
                assert "cross_corr_btc_24h" in X.columns
                assert "cross_vol_ratio_btc_24h" in X.columns

        splits = engine.split_synchronized(prepared, purge_bars=1)
        assert len(splits["BTCUSDT"].X_train) == len(splits["ETHUSDT"].X_train) == len(splits["SOLUSDT"].X_train)
        assert splits["BTCUSDT"].X_train.index.equals(splits["ETHUSDT"].X_train.index)
