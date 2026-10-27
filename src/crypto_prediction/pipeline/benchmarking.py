"""Controlled Multi-Asset Model Benchmarking Suite (Phase 3.4).

Executes a 2x2 factorial controlled experiment comparing:
- Approach A: Single-Asset Baseline (Independent models, asset-specific features)
- Approach B: Single-Asset + Cross-Asset (Independent models, cross-asset features)
- Approach C: Multi-Asset Baseline (Shared pooled model, asset-specific features + asset ID)
- Approach D: Multi-Asset + Cross-Asset (Shared pooled model, cross-asset features + asset ID)

Protocols:
- Strict chronological walk-forward cross-validation across 3 distinct market regimes.
- 24-bar purge gap to prevent target overlap leakage.
- Training-only preprocessing and parameter fitting.
- Final holdout strictly untouched.
- Evaluation on identical test timestamps across all 4 approaches.
- Metric calculation: MAE, RMSE, OOS R2 (vs train mean and vs zero), Directional Accuracy,
  and relative baseline improvements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from crypto_prediction.data.collector import compute_file_checksum
from crypto_prediction.models.base import BaseModel
from crypto_prediction.models.baselines import MeanReturnBaseline, ZeroReturnBaseline
from crypto_prediction.models.boosters import CatBoostModel, LightGBMModel, XGBoostModel
from crypto_prediction.models.linear import RidgeModel
from crypto_prediction.utils.logger import get_logger
from crypto_prediction.utils.reproducibility import set_seed

logger = get_logger("benchmarking_3_4")


@dataclass
class WalkForwardFoldSpec:
    """Defines exact boundaries for a walk-forward chronological fold."""

    fold_id: int
    name: str
    train_start_idx: int
    train_end_idx: int
    purge_bars: int
    test_start_idx: int
    test_end_idx: int
    train_start_utc: str = ""
    train_end_utc: str = ""
    test_start_utc: str = ""
    test_end_utc: str = ""


class ControlledBenchmarkEngine:
    """Orchestrates Phase 3.4 controlled multi-asset model benchmarking."""

    MODEL_FACTORIES = {
        "ZeroReturn": lambda: ZeroReturnBaseline(),
        "MeanReturn": lambda: MeanReturnBaseline(),
        "Ridge": lambda: RidgeModel(alpha=10.0),
        "LightGBM": lambda: LightGBMModel(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        ),
        "CatBoost": lambda: CatBoostModel(
            iterations=150,
            learning_rate=0.03,
            depth=4,
            random_seed=42,
            verbose=0,
        ),
        "XGBoost": lambda: XGBoostModel(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
    }

    def __init__(
        self,
        features_dir: str | Path = "data/processed/features_panel",
        artifacts_dir: str | Path = "artifacts/phase3_4",
        symbols: list[str] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = self.artifacts_dir / "models"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.symbols = [s.upper() for s in (symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"])]
        self.seed = random_seed
        set_seed(self.seed)

    def load_feature_datasets(self) -> dict[str, pd.DataFrame]:
        """Loads synchronized cross-asset feature panels and populates self-reference constants."""
        datasets = {}
        for sym in self.symbols:
            csv_path = self.features_dir / f"{sym}_1h_features.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Feature dataset not found: {csv_path}")
            df = pd.read_csv(csv_path)
            s_clean = sym.replace("USDT", "").lower()
            for col in [
                f"cross_corr_{s_clean}_24h",
                f"cross_corr_{s_clean}_168h",
                f"cross_vol_ratio_{s_clean}_24h",
                f"cross_vol_ratio_{s_clean}_168h",
            ]:
                if col not in df.columns:
                    df[col] = 1.0
            for col in [f"cross_mom_spread_{s_clean}_1h", f"cross_mom_spread_{s_clean}_24h"]:
                if col not in df.columns:
                    df[col] = 0.0
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            datasets[sym] = df.set_index("open_time")
        return datasets

    @staticmethod
    def partition_feature_columns(all_columns: list[str]) -> tuple[list[str], list[str]]:
        """Separates asset-specific feature columns from cross-asset peer columns."""
        cross_prefixes = [
            "btc_return_lag",
            "eth_return_lag",
            "sol_return_lag",
            "cross_vol_ratio",
            "cross_corr",
            "cross_mom_spread",
            "market_mean_return",
            "market_return_dispersion",
            "market_mean_volatility",
        ]
        asset_specific = []
        cross_asset = []

        for col in all_columns:
            if col.startswith("target_") or col in ["open", "high", "low", "close", "volume", "symbol", "open_time"]:
                continue
            if any(col.startswith(p) for p in cross_prefixes):
                cross_asset.append(col)
            else:
                asset_specific.append(col)

        return sorted(asset_specific), sorted(asset_specific + cross_asset)

    def define_walk_forward_folds(self, total_bars: int) -> list[WalkForwardFoldSpec]:
        """Defines 3 chronological walk-forward folds spanning multiple market regimes.

        Reserve the final ~7,300 bars (2026) as untouched holdout.
        """
        # Multi-regime boundaries within the 33,324 bars
        folds = [
            WalkForwardFoldSpec(
                fold_id=1,
                name="Fold 1 (2023 Recovery Regime)",
                train_start_idx=0,
                train_end_idx=8000,
                purge_bars=24,
                test_start_idx=8024,
                test_end_idx=10024,
            ),
            WalkForwardFoldSpec(
                fold_id=2,
                name="Fold 2 (2024 ETF & Momentum Regime)",
                train_start_idx=0,
                train_end_idx=16000,
                purge_bars=24,
                test_start_idx=16024,
                test_end_idx=18024,
            ),
            WalkForwardFoldSpec(
                fold_id=3,
                name="Fold 3 (2025 High-Volatility Regime)",
                train_start_idx=0,
                train_end_idx=24000,
                purge_bars=24,
                test_start_idx=24024,
                test_end_idx=26024,
            ),
        ]
        return folds

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train_mean: float,
    ) -> dict[str, float]:
        """Computes RMSE, MAE, out-of-sample R2, and directional accuracy."""
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)

        valid = (~np.isnan(y_t)) & (~np.isnan(y_p))
        y_t = y_t[valid]
        y_p = y_p[valid]

        mae = float(np.mean(np.abs(y_t - y_p)))
        mse = float(np.mean((y_t - y_p) ** 2))
        rmse = float(np.sqrt(mse))

        # R2 vs train mean (financial OOS R2)
        ss_tot_train = float(np.sum((y_t - y_train_mean) ** 2))
        ss_res = float(np.sum((y_t - y_p) ** 2))
        r2_oos = float(1.0 - (ss_res / (ss_tot_train + 1e-12)))

        # R2 vs zero return
        ss_tot_zero = float(np.sum(y_t**2))
        r2_zero = float(1.0 - (ss_res / (ss_tot_zero + 1e-12)))

        # Directional accuracy
        active = (y_p != 0.0)
        if np.any(active):
            dir_acc = float(np.mean(np.sign(y_t[active]) == np.sign(y_p[active])))
            coverage = float(np.mean(active))
        else:
            dir_acc = float("nan")
            coverage = 0.0

        return {
            "n_samples": len(y_t),
            "mae": mae,
            "rmse": rmse,
            "r2_oos": r2_oos,
            "r2_zero": r2_zero,
            "directional_accuracy": dir_acc,
            "coverage": coverage,
            "pred_mean": float(np.mean(y_p)),
            "pred_std": float(np.std(y_p)),
        }

    def run_benchmark(
        self,
        approaches: list[str] | None = None,
        models: list[str] | None = None,
        use_sample_weighting: bool = True,
    ) -> pd.DataFrame:
        """Executes the full controlled 2x2 factorial experiment across all folds, assets, and models."""
        target_approaches = approaches or [
            "A_single_asset_baseline",
            "B_single_asset_cross_asset",
            "C_multi_asset_baseline",
            "D_multi_asset_cross_asset",
        ]
        target_models = models or ["ZeroReturn", "MeanReturn", "Ridge", "LightGBM", "CatBoost", "XGBoost"]

        datasets = self.load_feature_datasets()
        first_sym = self.symbols[0]
        n_bars = len(datasets[first_sym])
        timestamps = datasets[first_sym].index

        # Partition features
        all_feature_cols = [
            c for c in datasets[first_sym].columns
            if c not in ["target_return_1", "open", "high", "low", "close", "volume", "symbol"]
        ]
        asset_cols, cross_cols = self.partition_feature_columns(all_feature_cols)

        # Ensure only columns common to all datasets are retained
        common_cols = set.intersection(*(set(datasets[s].columns) for s in self.symbols))
        asset_cols = [c for c in asset_cols if c in common_cols]
        cross_cols = [c for c in cross_cols if c in common_cols]

        logger.info(
            f"Feature Partitioning: {len(asset_cols)} asset-specific features, "
            f"{len(cross_cols)} total cross-asset features."
        )

        folds = self.define_walk_forward_folds(n_bars)
        results_records = []

        # Run folds
        for fold in folds:
            fold.train_start_utc = str(timestamps[fold.train_start_idx])
            fold.train_end_utc = str(timestamps[fold.train_end_idx - 1])
            fold.test_start_utc = str(timestamps[fold.test_start_idx])
            fold.test_end_utc = str(timestamps[fold.test_end_idx - 1])

            logger.info(
                f"\n>>> Executing {fold.name} | Train: {fold.train_start_utc} to {fold.train_end_utc} "
                f"({fold.train_end_idx - fold.train_start_idx} bars) | "
                f"Test: {fold.test_start_utc} to {fold.test_end_utc} ({fold.test_end_idx - fold.test_start_idx} bars) <<<"
            )

            # Pre-extract data splits per asset for this fold
            fold_splits: dict[str, dict[str, Any]] = {}
            for sym in self.symbols:
                df_sym = datasets[sym]
                y_train = df_sym["target_return_1"].iloc[fold.train_start_idx:fold.train_end_idx]
                y_test = df_sym["target_return_1"].iloc[fold.test_start_idx:fold.test_end_idx]
                X_train_asset = df_sym[asset_cols].iloc[fold.train_start_idx:fold.train_end_idx]
                X_test_asset = df_sym[asset_cols].iloc[fold.test_start_idx:fold.test_end_idx]
                X_train_cross = df_sym[cross_cols].iloc[fold.train_start_idx:fold.train_end_idx]
                X_test_cross = df_sym[cross_cols].iloc[fold.test_start_idx:fold.test_end_idx]

                # Volatility-based sample weights: w = 1 / (vol_24h^2)
                vol_train = df_sym["volatility_24h"].iloc[fold.train_start_idx:fold.train_end_idx].clip(lower=1e-4)
                sample_weights = (1.0 / (vol_train**2)).to_numpy()

                fold_splits[sym] = {
                    "y_train": y_train,
                    "y_test": y_test,
                    "y_train_mean": float(y_train.mean()),
                    "X_train_asset": X_train_asset,
                    "X_test_asset": X_test_asset,
                    "X_train_cross": X_train_cross,
                    "X_test_cross": X_test_cross,
                    "sample_weights": sample_weights,
                }

            # Prepare Pooled Multi-Asset Training Data for Approaches C and D
            pooled_train_asset_list = []
            pooled_train_cross_list = []
            pooled_train_y_list = []
            pooled_weights_list = []

            for sym in self.symbols:
                s = fold_splits[sym]
                n_t = len(s["y_train"])

                # Add one-hot asset indicators
                df_a_hot = s["X_train_asset"].copy()
                df_c_hot = s["X_train_cross"].copy()
                for o_sym in self.symbols:
                    df_a_hot[f"asset_{o_sym}"] = 1.0 if o_sym == sym else 0.0
                    df_c_hot[f"asset_{o_sym}"] = 1.0 if o_sym == sym else 0.0

                pooled_train_asset_list.append(df_a_hot)
                pooled_train_cross_list.append(df_c_hot)
                pooled_train_y_list.append(s["y_train"])
                if use_sample_weighting:
                    # Normalize weights across assets so sum(w_i) is equal
                    norm_w = s["sample_weights"] / (np.sum(s["sample_weights"]) + 1e-12) * n_t
                    pooled_weights_list.append(norm_w)
                else:
                    pooled_weights_list.append(np.ones(n_t))

            pooled_X_train_asset = pd.concat(pooled_train_asset_list, axis=0)
            pooled_X_train_cross = pd.concat(pooled_train_cross_list, axis=0)
            pooled_y_train = pd.concat(pooled_train_y_list, axis=0)
            pooled_sample_weights = np.concatenate(pooled_weights_list, axis=0)
            pooled_train_mean = float(pooled_y_train.mean())

            # Evaluate each approach
            for approach in target_approaches:
                logger.info(f"Evaluating Approach: {approach}...")

                for model_name in target_models:
                    model_factory = self.MODEL_FACTORIES[model_name]

                    if approach == "A_single_asset_baseline":
                        # Approach A: Train independent models using asset-specific features
                        for sym in self.symbols:
                            s = fold_splits[sym]
                            model = model_factory()
                            model.fit(s["X_train_asset"], s["y_train"])
                            preds = model.predict(s["X_test_asset"])
                            metrics = self.calculate_metrics(s["y_test"].to_numpy(), preds, s["y_train_mean"])
                            results_records.append({
                                "fold_id": fold.fold_id,
                                "fold_name": fold.name,
                                "approach": approach,
                                "model_family": model_name,
                                "symbol": sym,
                                "features_type": "asset_specific",
                                "is_pooled": False,
                                "test_bars": len(s["y_test"]),
                                **metrics,
                            })

                    elif approach == "B_single_asset_cross_asset":
                        # Approach B: Train independent models using cross-asset features
                        for sym in self.symbols:
                            s = fold_splits[sym]
                            model = model_factory()
                            model.fit(s["X_train_cross"], s["y_train"])
                            preds = model.predict(s["X_test_cross"])
                            metrics = self.calculate_metrics(s["y_test"].to_numpy(), preds, s["y_train_mean"])
                            results_records.append({
                                "fold_id": fold.fold_id,
                                "fold_name": fold.name,
                                "approach": approach,
                                "model_family": model_name,
                                "symbol": sym,
                                "features_type": "cross_asset",
                                "is_pooled": False,
                                "test_bars": len(s["y_test"]),
                                **metrics,
                            })

                    elif approach == "C_multi_asset_baseline":
                        # Approach C: One shared model trained on pooled asset-specific features + asset ID
                        shared_model = model_factory()
                        fit_weights = pooled_sample_weights if use_sample_weighting and model_name not in ["ZeroReturn"] else None
                        shared_model.fit(pooled_X_train_asset, pooled_y_train, sample_weight=fit_weights)

                        # Evaluate on each asset's test slice
                        for sym in self.symbols:
                            s = fold_splits[sym]
                            # Construct test features with one-hot indicators
                            X_test_hot = s["X_test_asset"].copy()
                            for o_sym in self.symbols:
                                X_test_hot[f"asset_{o_sym}"] = 1.0 if o_sym == sym else 0.0
                            preds = shared_model.predict(X_test_hot)
                            metrics = self.calculate_metrics(s["y_test"].to_numpy(), preds, s["y_train_mean"])
                            results_records.append({
                                "fold_id": fold.fold_id,
                                "fold_name": fold.name,
                                "approach": approach,
                                "model_family": model_name,
                                "symbol": sym,
                                "features_type": "asset_specific",
                                "is_pooled": True,
                                "test_bars": len(s["y_test"]),
                                **metrics,
                            })

                    elif approach == "D_multi_asset_cross_asset":
                        # Approach D: One shared model trained on pooled cross-asset features + asset ID
                        shared_model = model_factory()
                        fit_weights = pooled_sample_weights if use_sample_weighting and model_name not in ["ZeroReturn"] else None
                        shared_model.fit(pooled_X_train_cross, pooled_y_train, sample_weight=fit_weights)

                        # Evaluate on each asset's test slice
                        for sym in self.symbols:
                            s = fold_splits[sym]
                            X_test_hot = s["X_test_cross"].copy()
                            for o_sym in self.symbols:
                                X_test_hot[f"asset_{o_sym}"] = 1.0 if o_sym == sym else 0.0
                            preds = shared_model.predict(X_test_hot)
                            metrics = self.calculate_metrics(s["y_test"].to_numpy(), preds, s["y_train_mean"])
                            results_records.append({
                                "fold_id": fold.fold_id,
                                "fold_name": fold.name,
                                "approach": approach,
                                "model_family": model_name,
                                "symbol": sym,
                                "features_type": "cross_asset",
                                "is_pooled": True,
                                "test_bars": len(s["y_test"]),
                                **metrics,
                            })

        results_df = pd.DataFrame(results_records)

        # Baseline Relative Improvements:
        # 1. vs ZeroReturn
        zero_rmse_map = (
            results_df[results_df["model_family"] == "ZeroReturn"]
            .set_index(["fold_id", "symbol"])["rmse"]
            .to_dict()
        )
        results_df["zero_rmse"] = results_df.apply(
            lambda r: zero_rmse_map.get((r["fold_id"], r["symbol"]), np.nan), axis=1
        )
        results_df["improvement_vs_zero_pct"] = (
            (results_df["zero_rmse"] - results_df["rmse"]) / results_df["zero_rmse"]
        ) * 100.0

        # 2. vs Approach A (Single-Asset Baseline for same fold, model, and symbol)
        approach_a_map = (
            results_df[results_df["approach"] == "A_single_asset_baseline"]
            .set_index(["fold_id", "model_family", "symbol"])["rmse"]
            .to_dict()
        )
        results_df["approach_a_rmse"] = results_df.apply(
            lambda r: approach_a_map.get((r["fold_id"], r["model_family"], r["symbol"]), np.nan), axis=1
        )
        results_df["improvement_vs_approach_a_pct"] = (
            (results_df["approach_a_rmse"] - results_df["rmse"]) / results_df["approach_a_rmse"]
        ) * 100.0

        # Save artifacts
        out_csv = self.artifacts_dir / "phase3_4_benchmark_results.csv"
        results_df.to_csv(out_csv, index=False)
        meta_dict = {
            "total_evaluations": len(results_df),
            "approaches": target_approaches,
            "models": target_models,
            "symbols": self.symbols,
            "n_folds": len(folds),
            "sha256": compute_file_checksum(out_csv),
            "generated_at_utc": str(pd.Timestamp.now(tz="UTC")),
        }
        with open(out_csv.with_name(f"{out_csv.name}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2, default=str)

        logger.info(
            f"Successfully executed Phase 3.4 Benchmarking ({len(results_df)} evaluations). "
            f"Results saved to: {out_csv}"
        )
        return results_df
