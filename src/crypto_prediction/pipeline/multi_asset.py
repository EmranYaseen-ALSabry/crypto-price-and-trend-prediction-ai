"""Multi-Asset Generalization, Transferability, and Cross-Asset Benchmarking Engine.

Supports:
- Experiment A: Single-Asset Models (independent training on each asset).
- Experiment B: Pooled Multi-Asset Models (joint training on synchronized panel).
- Experiment C: Leave-One-Asset-Out (LOAO) Generalization (zero-shot transfer to unseen assets).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from crypto_prediction.data.cleaner import DataCleaner
from crypto_prediction.data.collector import DataCollector
from crypto_prediction.data.synchronizer import MultiAssetSynchronizer
from crypto_prediction.data.validator import DataValidator
from crypto_prediction.features.cross_asset import CrossAssetFeatureEngineer
from crypto_prediction.features.engineer import FeatureEngineer
from crypto_prediction.models.base import BaseModel
from crypto_prediction.models.baselines import (
    LastValueBaseline,
    MeanReturnBaseline,
    ZeroReturnBaseline,
)
from crypto_prediction.models.boosters import CatBoostModel, LightGBMModel, XGBoostModel
from crypto_prediction.models.linear import LinearRegressionModel, RidgeModel
from crypto_prediction.models.lstm import LSTMModel
from crypto_prediction.models.trees import HistGradientBoostingModel, RandomForestModel
from crypto_prediction.pipeline.backtester import BacktestResult, StrategyBacktester
from crypto_prediction.pipeline.evaluator import EvaluationMetrics, ModelEvaluator
from crypto_prediction.utils.logger import get_logger
from crypto_prediction.utils.reproducibility import set_seed
from crypto_prediction.validation.splitter import ChronologicalSplitter, DatasetSplits

logger = get_logger("multi_asset_engine")


class MultiAssetEngine:
    """Orchestrates multi-asset cryptocurrency forecasting experiments and transfer evaluation."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        interval: str = "1h",
        horizon: int = 1,
        artifacts_dir: str | Path = "artifacts/multi_asset",
        random_seed: int = 42,
        use_cross_asset_features: bool = False,
    ) -> None:
        self.symbols = [s.upper() for s in (symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"])]
        self.interval = interval
        self.horizon = horizon
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = self.artifacts_dir / "models"
        self.figures_dir = self.artifacts_dir / "figures"
        self.use_cross_asset_features = use_cross_asset_features

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.seed = random_seed
        set_seed(self.seed)

    def load_and_align_datasets(
        self,
        data_source: str = "sample",
        total_bars: int = 1500,
    ) -> dict[str, pd.DataFrame]:
        """Loads and strictly aligns OHLCV datasets across all target assets.

        Args:
            data_source: 'sample' (static verified panel), 'raw' (cached raw), or 'live' (Binance REST).
            total_bars: Number of bars to load per asset.

        Returns:
            Dictionary mapping symbol -> synchronized cleaned DataFrame.
        """
        collector = DataCollector()
        cleaner = DataCleaner()
        validator = DataValidator(expected_interval=self.interval)

        raw_dfs: dict[str, pd.DataFrame] = {}

        for sym in self.symbols:
            if data_source == "sample":
                sample_path = Path("data/sample") / f"{sym}_{self.interval}.csv"
                if sample_path.exists():
                    df = collector.load_from_csv(sample_path)
                else:
                    logger.warning(f"Sample file {sample_path} not found. Falling back to collector.")
                    df = collector.fetch_historical_bars(symbol=sym, interval=self.interval, total_bars=total_bars)
            elif data_source == "panel_5000h":
                panel_path = Path("data/processed/panel_5000h") / f"{sym}_{self.interval}.csv"
                if panel_path.exists():
                    df = collector.load_from_csv(panel_path)
                else:
                    logger.warning(f"Panel file {panel_path} not found. Falling back to collector.")
                    df = collector.fetch_historical_bars(symbol=sym, interval=self.interval, total_bars=total_bars)
            elif data_source == "raw":
                raw_path = Path("data/raw") / f"{sym}_{self.interval}.csv"
                if raw_path.exists():
                    df = collector.load_from_csv(raw_path)
                else:
                    df = collector.fetch_historical_bars(symbol=sym, interval=self.interval, total_bars=total_bars)
            else:
                # Live Binance
                df = collector.fetch_historical_bars(symbol=sym, interval=self.interval, total_bars=total_bars)

            # Validate and clean
            val_rep = validator.validate(df)
            if not val_rep.is_valid:
                raise ValueError(f"Validation failed for {sym}: {val_rep.errors}")

            clean_df = cleaner.clean(df)
            raw_dfs[sym] = clean_df

        # Determine common overlapping DatetimeIndex
        common_index = raw_dfs[self.symbols[0]].index
        for sym in self.symbols[1:]:
            common_index = common_index.intersection(raw_dfs[sym].index)

        common_index = common_index.sort_values()
        if len(common_index) < 100:
            raise ValueError(
                f"Insufficient overlapping timestamps across assets ({len(common_index)} common bars). "
                f"Assets cannot be fairly compared."
            )

        logger.info(
            f"Synchronized {len(self.symbols)} assets over {len(common_index)} common bars "
            f"({common_index[0]} to {common_index[-1]} UTC)."
        )

        aligned_dfs: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            aligned_dfs[sym] = raw_dfs[sym].loc[common_index].copy()

        return aligned_dfs

    def prepare_features(
        self,
        datasets: dict[str, pd.DataFrame],
        include_patterns: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Extracts scale-invariant features for each aligned asset.

        Supports standard single-asset features or unified cross-asset feature matrices.

        Returns:
            Dictionary mapping symbol -> {'X': X, 'y': y, 'meta': metadata}.
        """
        if self.use_cross_asset_features:
            cross_engineer = CrossAssetFeatureEngineer(
                anchor_symbols=self.symbols,
                primary_horizon=self.horizon,
                forecast_horizons=[1, 4, 24],
                include_patterns=include_patterns,
            )
            return cross_engineer.prepare_dataset(datasets, target_horizon=self.horizon, drop_warmup=True)

        engineer = FeatureEngineer(
            primary_horizon=self.horizon,
            forecast_horizons=[1, 4, 24],
            include_patterns=include_patterns,
        )

        prepared: dict[str, dict[str, Any]] = {}
        for sym, df in datasets.items():
            X, y, meta = engineer.prepare_dataset(df, target_horizon=self.horizon, drop_warmup=True)
            prepared[sym] = {"X": X, "y": y, "meta": meta}
            logger.info(f"Engineered {len(X.columns)} features for {sym} ({len(X)} usable rows).")

        # Re-align post-warmup indices across all assets to guarantee exact alignment
        common_usable_index = prepared[self.symbols[0]]["X"].index
        for sym in self.symbols[1:]:
            common_usable_index = common_usable_index.intersection(prepared[sym]["X"].index)

        common_usable_index = common_usable_index.sort_values()
        for sym in self.symbols:
            prepared[sym]["X"] = prepared[sym]["X"].loc[common_usable_index]
            prepared[sym]["y"] = prepared[sym]["y"].loc[common_usable_index]
            prepared[sym]["meta"] = prepared[sym]["meta"].loc[common_usable_index]

        logger.info(f"Aligned post-warmup panel: {len(common_usable_index)} observations per asset.")
        return prepared

    def split_synchronized(
        self,
        prepared_data: dict[str, dict[str, Any]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        purge_bars: int = 1,
    ) -> dict[str, DatasetSplits]:
        """Applies identical chronological splitting and purging across all assets."""
        splitter = ChronologicalSplitter(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            purge_bars=purge_bars,
        )

        splits_per_asset: dict[str, DatasetSplits] = {}
        for sym in self.symbols:
            X = prepared_data[sym]["X"]
            y = prepared_data[sym]["y"]
            meta = prepared_data[sym]["meta"]
            splits = splitter.split(X, y, metadata=meta)
            splits_per_asset[sym] = splits

        first_sym = self.symbols[0]
        s0 = splits_per_asset[first_sym]
        logger.info(
            f"Synchronized Splits: Train={len(s0.X_train)}, Val={len(s0.X_val)}, Test={len(s0.X_test)} bars "
            f"(purge={purge_bars} bar)."
        )
        return splits_per_asset

    def instantiate_models(self) -> dict[str, BaseModel]:
        """Instantiates core candidate models with deterministic seeds."""
        models: dict[str, BaseModel] = {
            "ZeroReturnBaseline": ZeroReturnBaseline(),
            "MeanReturnBaseline": MeanReturnBaseline(),
            "Ridge": RidgeModel(alpha=1.0, random_state=self.seed),
            "RandomForest": RandomForestModel(n_estimators=100, max_depth=5, random_state=self.seed),
        }

        # Boosters
        try:
            models["XGBoost"] = XGBoostModel(
                n_estimators=150, learning_rate=0.03, max_depth=4, standardize_target=True, random_state=self.seed
            )
        except Exception as e:
            logger.warning(f"XGBoost unavailable: {e}")

        try:
            models["LightGBM"] = LightGBMModel(
                n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=15, standardize_target=True, random_state=self.seed
            )
        except Exception as e:
            logger.warning(f"LightGBM unavailable: {e}")

        try:
            models["CatBoost"] = CatBoostModel(
                iterations=150, learning_rate=0.03, depth=4, random_seed=self.seed, verbose=0
            )
        except Exception as e:
            logger.warning(f"CatBoost unavailable: {e}")

        return models

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray,
        train_y: np.ndarray | pd.Series,
    ) -> dict[str, float]:
        """Calculates MAE, RMSE, Out-of-Sample R2, Directional Accuracy, and Spearman Rank IC."""
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        valid = (~np.isnan(y_t)) & (~np.isnan(y_p))
        y_t = y_t[valid]
        y_p = y_p[valid]

        if len(y_t) == 0:
            return {"mae": np.nan, "rmse": np.nan, "r2_oos": np.nan, "directional_accuracy": np.nan, "rank_ic": np.nan}

        mae = float(np.mean(np.abs(y_t - y_p)))
        rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

        # Out-of-sample R2 against training mean
        y_train_arr = np.asarray(train_y, dtype=float)
        train_mean = float(np.mean(y_train_arr[~np.isnan(y_train_arr)])) if len(y_train_arr) > 0 else 0.0
        ss_res = float(np.sum((y_t - y_p) ** 2))
        ss_tot = float(np.sum((y_t - train_mean) ** 2))
        r2_oos = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0

        # Directional accuracy
        sign_t = np.sign(y_t)
        sign_p = np.sign(y_p)
        active = sign_p != 0
        da = float(np.mean(sign_t[active] == sign_p[active])) if np.any(active) else np.nan

        # Rank IC (Spearman correlation)
        if len(y_t) > 3 and np.std(y_p) > 1e-9:
            corr, _ = spearmanr(y_t, y_p)
            rank_ic = float(corr) if not np.isnan(corr) else 0.0
        else:
            rank_ic = 0.0

        return {
            "mae": mae,
            "rmse": rmse,
            "r2_oos": r2_oos,
            "directional_accuracy": da,
            "rank_ic": rank_ic,
        }

    def run_full_research(
        self,
        data_source: str = "sample",
        total_bars: int = 1500,
    ) -> dict[str, Any]:
        """Executes Experiments A, B, and C under identical temporal conditions.

        Returns:
            Dictionary containing metrics tables, backtest summaries, and findings.
        """
        logger.info("=== Starting Multi-Asset Generalization Research (Phase 2) ===")

        # 1. Ingest, align, and prepare
        aligned_dfs = self.load_and_align_datasets(data_source=data_source, total_bars=total_bars)
        prepared_data = self.prepare_features(aligned_dfs)
        splits = self.split_synchronized(prepared_data)

        backtester = StrategyBacktester(fee_rate=0.0010, slippage_rate=0.0005)

        evaluation_records: list[dict[str, Any]] = []
        backtest_records: list[dict[str, Any]] = []

        # -------------------------------------------------------------
        # EXPERIMENT A: Single-Asset Baseline Models
        # -------------------------------------------------------------
        logger.info("--- Executing Experiment A: Single-Asset Models ---")
        single_models: dict[str, dict[str, BaseModel]] = {}

        for sym in self.symbols:
            s = splits[sym]
            single_models[sym] = self.instantiate_models()

            for model_name, model in single_models[sym].items():
                # Fit strictly on this asset's train partition
                model.fit(s.X_train, s.y_train)

                # Save artifact
                art_name = f"{sym}_{self.interval}_Single_{model_name}_h{self.horizon}"
                model.save(self.models_dir / f"{art_name}.joblib")

                # Predict on test
                preds_test = model.predict(s.X_test)

                # Prediction Metrics
                m = self.compute_metrics(s.y_test, preds_test, s.y_train)
                rec = {
                    "experiment": "Experiment A (Single-Asset)",
                    "training_scope": sym,
                    "evaluation_asset": sym,
                    "model_name": model_name,
                    "train_samples": len(s.X_train),
                    "test_samples": len(s.X_test),
                    **m,
                }
                evaluation_records.append(rec)

                # Trading Backtest
                bt = backtester.run_backtest(
                    y_true=s.y_test,
                    y_pred=preds_test,
                    model_name=f"Single_{sym}_{model_name}",
                    horizon=self.horizon,
                )
                bt_dict = bt.to_dict()
                bt_dict.update({
                    "experiment": "Experiment A (Single-Asset)",
                    "training_scope": sym,
                    "evaluation_asset": sym,
                    "model_name": model_name,
                })
                backtest_records.append(bt_dict)

        # -------------------------------------------------------------
        # EXPERIMENT B: Pooled Multi-Asset Models
        # -------------------------------------------------------------
        logger.info("--- Executing Experiment B: Pooled Multi-Asset Models ---")
        # Construct pooled training partition across all assets
        X_train_pooled = pd.concat([splits[sym].X_train for sym in self.symbols], axis=0).reset_index(drop=True)
        y_train_pooled = pd.concat([splits[sym].y_train for sym in self.symbols], axis=0).reset_index(drop=True)

        pooled_models = self.instantiate_models()

        for model_name, model in pooled_models.items():
            model.fit(X_train_pooled, y_train_pooled)

            # Save pooled artifact
            art_name = f"POOLED_{self.interval}_Pooled_{model_name}_h{self.horizon}"
            model.save(self.models_dir / f"{art_name}.joblib")

            # Evaluate on each individual asset's test set
            for sym in self.symbols:
                s = splits[sym]
                preds_test = model.predict(s.X_test)
                m = self.compute_metrics(s.y_test, preds_test, y_train_pooled)

                evaluation_records.append({
                    "experiment": "Experiment B (Pooled Multi-Asset)",
                    "training_scope": "ALL_POOLED",
                    "evaluation_asset": sym,
                    "model_name": model_name,
                    "train_samples": len(X_train_pooled),
                    "test_samples": len(s.X_test),
                    **m,
                })

                bt = backtester.run_backtest(
                    y_true=s.y_test,
                    y_pred=preds_test,
                    model_name=f"Pooled_{sym}_{model_name}",
                    horizon=self.horizon,
                )
                bt_dict = bt.to_dict()
                bt_dict.update({
                    "experiment": "Experiment B (Pooled Multi-Asset)",
                    "training_scope": "ALL_POOLED",
                    "evaluation_asset": sym,
                    "model_name": model_name,
                })
                backtest_records.append(bt_dict)

        # -------------------------------------------------------------
        # EXPERIMENT C: Leave-One-Asset-Out (LOAO) Generalization
        # -------------------------------------------------------------
        logger.info("--- Executing Experiment C: Leave-One-Asset-Out Generalization ---")
        for held_out_sym in self.symbols:
            train_syms = [s for s in self.symbols if s != held_out_sym]
            logger.info(f"Training on {train_syms} -> Evaluating zero-shot on {held_out_sym}")

            X_train_loao = pd.concat([splits[s].X_train for s in train_syms], axis=0).reset_index(drop=True)
            y_train_loao = pd.concat([splits[s].y_train for s in train_syms], axis=0).reset_index(drop=True)

            loao_models = self.instantiate_models()

            for model_name, model in loao_models.items():
                model.fit(X_train_loao, y_train_loao)

                # Save artifact
                art_name = f"EXCLUDE_{held_out_sym}_{self.interval}_LOAO_{model_name}_h{self.horizon}"
                model.save(self.models_dir / f"{art_name}.joblib")

                # Evaluate on held-out asset's test set
                s_held = splits[held_out_sym]
                preds_test = model.predict(s_held.X_test)
                m = self.compute_metrics(s_held.y_test, preds_test, y_train_loao)

                evaluation_records.append({
                    "experiment": "Experiment C (Leave-One-Asset-Out)",
                    "training_scope": f"EXCLUDE_{held_out_sym}",
                    "evaluation_asset": held_out_sym,
                    "model_name": model_name,
                    "train_samples": len(X_train_loao),
                    "test_samples": len(s_held.X_test),
                    **m,
                })

                bt = backtester.run_backtest(
                    y_true=s_held.y_test,
                    y_pred=preds_test,
                    model_name=f"LOAO_{held_out_sym}_{model_name}",
                    horizon=self.horizon,
                )
                bt_dict = bt.to_dict()
                bt_dict.update({
                    "experiment": "Experiment C (Leave-One-Asset-Out)",
                    "training_scope": f"EXCLUDE_{held_out_sym}",
                    "evaluation_asset": held_out_sym,
                    "model_name": model_name,
                })
                backtest_records.append(bt_dict)

        # -------------------------------------------------------------
        # Compile and save research tables
        # -------------------------------------------------------------
        eval_df = pd.DataFrame(evaluation_records)
        eval_csv = self.artifacts_dir / "multi_asset_evaluation_metrics.csv"
        eval_df.to_csv(eval_csv, index=False)
        logger.info(f"Saved evaluation metrics to {eval_csv}")

        bt_df = pd.DataFrame(backtest_records)
        bt_csv = self.artifacts_dir / "multi_asset_trading_backtest.csv"
        bt_df.to_csv(bt_csv, index=False)
        logger.info(f"Saved backtest metrics to {bt_csv}")

        # Comparative summary across strategies (Single vs Pooled vs LOAO)
        summary_rows = []
        for sym in self.symbols:
            for model_name in ["Ridge", "RandomForest", "LightGBM", "XGBoost", "CatBoost", "ZeroReturnBaseline"]:
                row: dict[str, Any] = {"asset": sym, "model": model_name}
                # Single-Asset
                sub_single = eval_df[
                    (eval_df["experiment"].str.contains("Single"))
                    & (eval_df["evaluation_asset"] == sym)
                    & (eval_df["model_name"] == model_name)
                ]
                if not sub_single.empty:
                    row["single_rmse"] = sub_single["rmse"].iloc[0]
                    row["single_r2_oos"] = sub_single["r2_oos"].iloc[0]
                    row["single_da"] = sub_single["directional_accuracy"].iloc[0]

                # Pooled
                sub_pooled = eval_df[
                    (eval_df["experiment"].str.contains("Pooled"))
                    & (eval_df["evaluation_asset"] == sym)
                    & (eval_df["model_name"] == model_name)
                ]
                if not sub_pooled.empty:
                    row["pooled_rmse"] = sub_pooled["rmse"].iloc[0]
                    row["pooled_r2_oos"] = sub_pooled["r2_oos"].iloc[0]
                    row["pooled_da"] = sub_pooled["directional_accuracy"].iloc[0]

                # LOAO
                sub_loao = eval_df[
                    (eval_df["experiment"].str.contains("Leave-One-Asset-Out"))
                    & (eval_df["evaluation_asset"] == sym)
                    & (eval_df["model_name"] == model_name)
                ]
                if not sub_loao.empty:
                    row["loao_rmse"] = sub_loao["rmse"].iloc[0]
                    row["loao_r2_oos"] = sub_loao["r2_oos"].iloc[0]
                    row["loao_da"] = sub_loao["directional_accuracy"].iloc[0]

                summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = self.artifacts_dir / "multi_asset_strategy_comparison.csv"
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Saved strategy comparison table to {summary_csv}")

        # Metadata
        meta_dict = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "symbols": self.symbols,
            "interval": self.interval,
            "horizon": self.horizon,
            "data_source": data_source,
            "total_bars_per_asset": total_bars,
            "train_bars": len(splits[self.symbols[0]].X_train),
            "val_bars": len(splits[self.symbols[0]].X_val),
            "test_bars": len(splits[self.symbols[0]].X_test),
            "evaluation_records_count": len(eval_df),
            "backtest_records_count": len(bt_df),
        }
        with open(self.artifacts_dir / "multi_asset_experiment_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        logger.info("=== Completed Multi-Asset Generalization Research ===")
        return {
            "evaluation_df": eval_df,
            "backtest_df": bt_df,
            "summary_df": summary_df,
            "splits": splits,
            "metadata": meta_dict,
        }
