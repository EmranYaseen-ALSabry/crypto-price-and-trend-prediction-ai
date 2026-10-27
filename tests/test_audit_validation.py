"""Phase 2.1 Audit Validation and Regression Tests.

Verifies:
1. Exact sample count reconciliation from raw bars to partitions.
2. 100% numerical reproduction of saved model artifacts and evaluation metrics.
3. Out-of-sample R2 formula and baseline reference behavior.
4. Open-to-close execution spread bounds vs. backtester slippage rate.
5. Statistical reliability flags on small-sample backtests.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crypto_prediction.data.cleaner import DataCleaner
from crypto_prediction.features.engineer import FeatureEngineer
from crypto_prediction.models.base import BaseModel
from crypto_prediction.pipeline.backtester import StrategyBacktester
from crypto_prediction.pipeline.multi_asset import MultiAssetEngine
from crypto_prediction.validation.splitter import ChronologicalSplitter


class TestPhase2ValidationAudit:
    """Rigorous audit checks and regression verifications."""

    def test_timeline_and_sample_count_reconciliation(self) -> None:
        """Verifies exact arithmetic of raw bars -> warmup -> target shift -> splits."""
        df = pd.read_csv("data/sample/BTCUSDT_1h.csv")
        assert len(df) == 1500

        clean_df = DataCleaner().clean(df)
        assert len(clean_df) == 1500
        assert clean_df.index[0] == pd.Timestamp("2024-01-01 00:00:00+0000", tz="UTC")
        assert clean_df.index[-1] == pd.Timestamp("2024-03-03 11:00:00+0000", tz="UTC")

        fe = FeatureEngineer(primary_horizon=1, include_patterns=True)
        features = fe.compute_features(clean_df)
        targets = fe.create_targets(clean_df)

        # Warmup rows (168-hour longest rolling window)
        warmup_dropped = int(features.isna().any(axis=1).sum())
        assert warmup_dropped == 168

        # Target shift rows (last h=1 row has no future close)
        target_dropped = int(targets["target_return_1"].isna().sum())
        assert target_dropped == 1

        # Prepared dataset
        X, y, meta = fe.prepare_dataset(clean_df, target_horizon=1, drop_warmup=True)
        assert len(X) == 1500 - warmup_dropped - target_dropped == 1331
        assert X.index[0] == pd.Timestamp("2024-01-08 00:00:00+0000", tz="UTC")
        assert X.index[-1] == pd.Timestamp("2024-03-03 10:00:00+0000", tz="UTC")

        # Chronological splits
        splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=1)
        splits = splitter.split(X, y, metadata=meta)

        assert len(splits.X_train) == 931
        assert len(splits.X_val) == 199
        assert len(splits.X_test) == 199

        # Purge gaps
        gap1 = splits.X_val.index[0] - splits.X_train.index[-1]
        gap2 = splits.X_test.index[0] - splits.X_val.index[-1]
        assert gap1 == pd.Timedelta(hours=2)  # 1 bar purged in between
        assert gap2 == pd.Timedelta(hours=2)  # 1 bar purged in between

        # Total partition sum + 2 purged bars == 1331
        total_accounted = len(splits.X_train) + 1 + len(splits.X_val) + 1 + len(splits.X_test)
        assert total_accounted == 1331

    def test_reproduction_of_model_predictions_and_metrics(self) -> None:
        """Verifies that saved model artifacts reproduce recorded metrics with zero deviation."""
        engine = MultiAssetEngine(artifacts_dir="artifacts/multi_asset")
        aligned_dfs = engine.load_and_align_datasets(data_source="sample", total_bars=1500)
        prepared = engine.prepare_features(aligned_dfs)
        splits = engine.split_synchronized(prepared)

        eval_csv_path = Path("artifacts/multi_asset/multi_asset_evaluation_metrics.csv")
        assert eval_csv_path.exists()
        eval_df = pd.read_csv(eval_csv_path)

        models_dir = Path("artifacts/multi_asset/models")

        # Test a representative sample of Single, Pooled, and LOAO models
        test_cases = [
            ("Experiment A (Single-Asset)", "BTCUSDT", "BTCUSDT", "XGBoost", "BTCUSDT_1h_Single_XGBoost_h1.joblib"),
            ("Experiment A (Single-Asset)", "ETHUSDT", "ETHUSDT", "CatBoost", "ETHUSDT_1h_Single_CatBoost_h1.joblib"),
            ("Experiment B (Pooled Multi-Asset)", "ALL_POOLED", "BTCUSDT", "XGBoost", "POOLED_1h_Pooled_XGBoost_h1.joblib"),
            ("Experiment B (Pooled Multi-Asset)", "ALL_POOLED", "SOLUSDT", "CatBoost", "POOLED_1h_Pooled_CatBoost_h1.joblib"),
            ("Experiment C (Leave-One-Asset-Out)", "EXCLUDE_BTCUSDT", "BTCUSDT", "RandomForest", "EXCLUDE_BTCUSDT_1h_LOAO_RandomForest_h1.joblib"),
            ("Experiment C (Leave-One-Asset-Out)", "EXCLUDE_SOLUSDT", "SOLUSDT", "CatBoost", "EXCLUDE_SOLUSDT_1h_LOAO_CatBoost_h1.joblib"),
        ]

        for exp, train_scope, eval_asset, model_name, file_name in test_cases:
            model_file = models_dir / file_name
            assert model_file.exists()

            model = BaseModel.load(model_file)
            X_test = splits[eval_asset].X_test
            y_test = splits[eval_asset].y_test

            if "Single" in exp:
                train_y = splits[eval_asset].y_train
            elif "Pooled" in exp:
                train_y = pd.concat([splits[s].y_train for s in engine.symbols], axis=0)
            else:
                train_syms = [s for s in engine.symbols if s != eval_asset]
                train_y = pd.concat([splits[s].y_train for s in train_syms], axis=0)

            preds = model.predict(X_test)
            recalculated = engine.compute_metrics(y_test, preds, train_y)

            # Match against row in CSV
            row = eval_df[
                (eval_df["experiment"] == exp)
                & (eval_df["evaluation_asset"] == eval_asset)
                & (eval_df["model_name"] == model_name)
            ].iloc[0]

            assert np.isclose(recalculated["rmse"], row["rmse"], atol=1e-6)
            assert np.isclose(recalculated["mae"], row["mae"], atol=1e-6)
            assert np.isclose(recalculated["r2_oos"], row["r2_oos"], atol=1e-6)
            assert np.isclose(recalculated["directional_accuracy"], row["directional_accuracy"], atol=1e-6)
            assert np.isclose(recalculated["rank_ic"], row["rank_ic"], atol=1e-6)

    def test_oos_r2_formula_and_training_mean_dependency(self) -> None:
        """Proves that variation in baseline R2 across experiments is mathematically

        caused by shifting training set means.
        """
        engine = MultiAssetEngine()
        aligned = engine.load_and_align_datasets(data_source="sample", total_bars=1500)
        prepared = engine.prepare_features(aligned)
        splits = engine.split_synchronized(prepared)

        y_test_btc = splits["BTCUSDT"].y_test.values
        ss_res_zero = np.sum(y_test_btc**2)

        mean_train_single = splits["BTCUSDT"].y_train.mean()
        mean_train_pooled = pd.concat([splits[s].y_train for s in engine.symbols]).mean()
        mean_train_loao = pd.concat([splits["ETHUSDT"].y_train, splits["SOLUSDT"].y_train]).mean()

        r2_single = 1.0 - ss_res_zero / np.sum((y_test_btc - mean_train_single) ** 2)
        r2_pooled = 1.0 - ss_res_zero / np.sum((y_test_btc - mean_train_pooled) ** 2)
        r2_loao = 1.0 - ss_res_zero / np.sum((y_test_btc - mean_train_loao) ** 2)

        # Matches reported values in CSV
        assert np.isclose(r2_single, +0.00044577, atol=1e-6)
        assert np.isclose(r2_pooled, -0.00033359, atol=1e-6)
        assert np.isclose(r2_loao, -0.00026267, atol=1e-6)

    def test_open_to_close_execution_spread_bounds(self) -> None:
        """Verifies that average open-to-close gap is strictly absorbed by backtester slippage rate."""
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            df = pd.read_csv(f"data/sample/{sym}_1h.csv")
            c_t = df["close"].iloc[:-1].values
            o_next = df["open"].iloc[1:].values
            rel_diff = np.abs(o_next - c_t) / c_t
            # Mean gap in 24/7 crypto is ~0.04%, well within the 0.05% slippage rate
            assert rel_diff.mean() < 0.0005

    def test_statistical_reliability_flag_enforcement(self) -> None:
        """Verifies that strategies with fewer than min_trades (30) are flagged unreliable."""
        backtester = StrategyBacktester(min_trades=30)
        y_true = np.array([0.01, -0.02, 0.01, 0.03])
        y_pred = np.array([0.02, -0.01, 0.01, 0.02])

        res = backtester.run_backtest(y_true, y_pred, model_name="SmallSampleTest")
        assert res.total_trades < 30
        assert res.statistically_reliable is False
        assert "not statistically meaningful" in res.summary()
