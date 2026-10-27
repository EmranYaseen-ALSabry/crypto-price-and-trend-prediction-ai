"""Cross-Asset Transfer Learning & Volatility Adaptation Suite (Phase 3.5).

Implements Leave-One-Asset-Out (LOAO) evaluation:
- Source: BTC + ETH -> Target: SOL
- Source: BTC + SOL -> Target: ETH
- Source: ETH + SOL -> Target: BTC

Protocols & Distinctions:
1. Strict Isolation: Held-out asset labels never influence source training or feature calibration.
2. Transfer Modes:
   - Zero-Shot Raw: Source-trained model evaluated directly on target asset with raw returns.
   - Zero-Shot Volatility-Adapted: Source-trained model on volatility-normalized returns (z = r / sigma_t),
     dynamically scaled at test time using target asset's trailing causal volatility.
   - Few-Shot Volatility-Adapted: Source pre-trained model with residual fine-tuning on a small historical
     training partition of the target asset (strictly before the purge gap).
   - Asset-Specific Baseline: Model trained solely on the target asset's historical training partition.
3. Chronological walk-forward validation across 3 market regimes with 24-bar purging.
4. Final 2026 holdout (~7,300 bars) strictly quarantined.
5. All predictions restored to original return units before computing RMSE and MAE.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from crypto_prediction.pipeline.benchmarking import WalkForwardFoldSpec
from crypto_prediction.utils.logger import get_logger
from crypto_prediction.utils.reproducibility import set_seed

logger = get_logger("transfer_learning_3_5")


class TransferLearningEngine:
    """Orchestrates Phase 3.5 Cross-Asset Transfer Learning & Volatility Adaptation."""

    SOURCE_MODEL_FACTORIES = {
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

    ADAPTER_MODEL_FACTORIES = {
        "ZeroReturn": lambda: ZeroReturnBaseline(),
        "MeanReturn": lambda: MeanReturnBaseline(),
        "Ridge": lambda: RidgeModel(alpha=50.0),
        "LightGBM": lambda: LightGBMModel(
            n_estimators=50,
            learning_rate=0.02,
            max_depth=3,
            num_leaves=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        ),
        "CatBoost": lambda: CatBoostModel(
            iterations=50,
            learning_rate=0.02,
            depth=3,
            random_seed=42,
            verbose=0,
        ),
        "XGBoost": lambda: XGBoostModel(
            n_estimators=50,
            learning_rate=0.02,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
    }

    def __init__(
        self,
        features_dir: str | Path = "data/processed/features_panel",
        artifacts_dir: str | Path = "artifacts/phase3_5",
        symbols: list[str] | None = None,
        random_seed: int = 42,
        few_shot_bars: int = 1000,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = self.artifacts_dir / "models"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.symbols = [s.upper() for s in (symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"])]
        self.seed = random_seed
        self.few_shot_bars = few_shot_bars
        set_seed(self.seed)

    def load_feature_datasets(self) -> dict[str, pd.DataFrame]:
        """Loads synchronized feature panels and populates self-reference constants."""
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
    def extract_asset_agnostic_features(all_columns: list[str]) -> list[str]:
        """Extracts scale-invariant, asset-agnostic feature columns suitable for zero-shot transfer."""
        cross_ticker_prefixes = [
            "btc_return_lag",
            "eth_return_lag",
            "sol_return_lag",
            "cross_vol_ratio",
            "cross_corr",
            "cross_mom_spread",
        ]
        asset_agnostic = []
        for col in all_columns:
            if col.startswith("target_") or col in ["open", "high", "low", "close", "volume", "symbol", "open_time"]:
                continue
            if not any(col.startswith(p) for p in cross_ticker_prefixes):
                asset_agnostic.append(col)
        return sorted(asset_agnostic)

    def define_walk_forward_folds(self, total_bars: int) -> list[WalkForwardFoldSpec]:
        """Defines 3 chronological walk-forward folds identical to Phase 3.4."""
        return [
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

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train_mean: float,
    ) -> dict[str, float]:
        """Computes RMSE, MAE, out-of-sample R2, and directional accuracy in original return units."""
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)

        valid = (~np.isnan(y_t)) & (~np.isnan(y_p))
        y_t = y_t[valid]
        y_p = y_p[valid]

        mae = float(np.mean(np.abs(y_t - y_p)))
        mse = float(np.mean((y_t - y_p) ** 2))
        rmse = float(np.sqrt(mse))

        # Financial OOS R2 (vs target asset train mean)
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

    def run_transfer_experiments(
        self,
        target_assets: list[str] | None = None,
        modes: list[str] | None = None,
        models: list[str] | None = None,
    ) -> pd.DataFrame:
        """Executes full Leave-One-Asset-Out transfer learning and volatility adaptation experiments."""
        target_assets_list = [s.upper() for s in (target_assets or self.symbols)]
        target_modes = modes or [
            "zero_shot_raw",
            "zero_shot_vol_adapted",
            "few_shot_vol_adapted",
            "asset_specific_baseline",
        ]
        target_models = models or ["ZeroReturn", "MeanReturn", "Ridge", "LightGBM", "CatBoost", "XGBoost"]

        datasets = self.load_feature_datasets()
        first_sym = self.symbols[0]
        n_bars = len(datasets[first_sym])
        timestamps = datasets[first_sym].index

        # Extract common asset-agnostic features
        common_cols = set.intersection(*(set(datasets[s].columns) for s in self.symbols))
        agnostic_cols = self.extract_asset_agnostic_features(list(common_cols))

        logger.info(
            f"Transfer Learning Engine initialized: {len(agnostic_cols)} asset-agnostic features, "
            f"{len(target_assets_list)} LOAO targets across {len(target_modes)} modes."
        )

        folds = self.define_walk_forward_folds(n_bars)
        results_records = []

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
                X_train = df_sym[agnostic_cols].iloc[fold.train_start_idx:fold.train_end_idx]
                X_test = df_sym[agnostic_cols].iloc[fold.test_start_idx:fold.test_end_idx]

                # Causal trailing 24h volatility at prediction bar close t
                vol_train = df_sym["volatility_24h"].iloc[fold.train_start_idx:fold.train_end_idx].clip(lower=1e-5)
                vol_test = df_sym["volatility_24h"].iloc[fold.test_start_idx:fold.test_end_idx].clip(lower=1e-5)

                # Volatility-normalized target: z = r / vol_24h
                z_train = y_train / vol_train
                z_test = y_test / vol_test

                # Few-shot historical slice: last few_shot_bars of target train
                few_idx_start = max(0, len(y_train) - self.few_shot_bars)
                X_train_few = X_train.iloc[few_idx_start:]
                y_train_few = y_train.iloc[few_idx_start:]
                z_train_few = z_train.iloc[few_idx_start:]
                vol_train_few = vol_train.iloc[few_idx_start:]

                fold_splits[sym] = {
                    "X_train": X_train,
                    "y_train": y_train,
                    "z_train": z_train,
                    "vol_train": vol_train,
                    "X_test": X_test,
                    "y_test": y_test,
                    "z_test": z_test,
                    "vol_test": vol_test,
                    "y_train_mean": float(y_train.mean()),
                    "X_train_few": X_train_few,
                    "y_train_few": y_train_few,
                    "z_train_few": z_train_few,
                    "vol_train_few": vol_train_few,
                }

            # Run LOAO experiments for each target asset
            for target_asset in target_assets_list:
                source_assets = [s for s in self.symbols if s != target_asset]
                logger.info(f"Target Asset: {target_asset} (Source Assets: {source_assets})")

                s_target = fold_splits[target_asset]
                y_target_test = s_target["y_test"].to_numpy()
                vol_target_test = s_target["vol_test"].to_numpy()
                y_target_train_mean = s_target["y_train_mean"]

                # Construct pooled source datasets (zero target data)
                X_source_train = pd.concat([fold_splits[s]["X_train"] for s in source_assets], axis=0)
                y_source_train = pd.concat([fold_splits[s]["y_train"] for s in source_assets], axis=0)
                z_source_train = pd.concat([fold_splits[s]["z_train"] for s in source_assets], axis=0)

                for model_name in target_models:
                    source_factory = self.SOURCE_MODEL_FACTORIES[model_name]
                    adapter_factory = self.ADAPTER_MODEL_FACTORIES[model_name]

                    # -------------------------------------------------------------
                    # Mode 1: Zero-Shot Raw (Unadapted Zero-Shot Transfer)
                    # -------------------------------------------------------------
                    if "zero_shot_raw" in target_modes:
                        model_raw = source_factory()
                        model_raw.fit(X_source_train, y_source_train)
                        preds_raw = model_raw.predict(s_target["X_test"])
                        m_raw = self.calculate_metrics(y_target_test, preds_raw, y_target_train_mean)
                        results_records.append({
                            "fold_id": fold.fold_id,
                            "fold_name": fold.name,
                            "target_asset": target_asset,
                            "source_assets": "+".join(source_assets),
                            "model_family": model_name,
                            "transfer_mode": "zero_shot_raw",
                            "volatility_adapted": False,
                            "few_shot_adapted": False,
                            "test_bars": len(y_target_test),
                            **m_raw,
                        })

                    # -------------------------------------------------------------
                    # Mode 2: Zero-Shot Volatility-Adapted (Standardized z, Causal Scale)
                    # -------------------------------------------------------------
                    if "zero_shot_vol_adapted" in target_modes:
                        model_vol = source_factory()
                        model_vol.fit(X_source_train, z_source_train)
                        pred_z = model_vol.predict(s_target["X_test"])
                        # Inverse transform dynamically using target asset's causal trailing volatility
                        pred_y_adapted = pred_z * vol_target_test
                        m_vol = self.calculate_metrics(y_target_test, pred_y_adapted, y_target_train_mean)
                        results_records.append({
                            "fold_id": fold.fold_id,
                            "fold_name": fold.name,
                            "target_asset": target_asset,
                            "source_assets": "+".join(source_assets),
                            "model_family": model_name,
                            "transfer_mode": "zero_shot_vol_adapted",
                            "volatility_adapted": True,
                            "few_shot_adapted": False,
                            "test_bars": len(y_target_test),
                            **m_vol,
                        })

                    # -------------------------------------------------------------
                    # Mode 3: Few-Shot Volatility-Adapted (Source Pre-train + Target Residual Adapt)
                    # -------------------------------------------------------------
                    if "few_shot_vol_adapted" in target_modes:
                        model_base = source_factory()
                        model_base.fit(X_source_train, z_source_train)

                        if model_name in ["ZeroReturn"]:
                            pred_few_y = np.zeros_like(y_target_test)
                        elif model_name in ["MeanReturn"]:
                            # Target mean return on few-shot historical slice
                            few_mean = float(s_target["y_train_few"].mean())
                            pred_few_y = np.full_like(y_target_test, few_mean)
                        else:
                            # Predict z on few-shot historical slice of target asset
                            pred_few_base = model_base.predict(s_target["X_train_few"])
                            residual_few = s_target["z_train_few"].to_numpy() - pred_few_base

                            # Fit adapter model on target residuals
                            adapter_model = adapter_factory()
                            adapter_model.fit(s_target["X_train_few"], residual_few)

                            # Predict on test: base prediction + adapted residual
                            pred_test_base = model_base.predict(s_target["X_test"])
                            pred_test_residual = adapter_model.predict(s_target["X_test"])
                            pred_z_adapted = pred_test_base + pred_test_residual

                            # Inverse transform using causal trailing volatility
                            pred_few_y = pred_z_adapted * vol_target_test

                        m_few = self.calculate_metrics(y_target_test, pred_few_y, y_target_train_mean)
                        results_records.append({
                            "fold_id": fold.fold_id,
                            "fold_name": fold.name,
                            "target_asset": target_asset,
                            "source_assets": "+".join(source_assets),
                            "model_family": model_name,
                            "transfer_mode": "few_shot_vol_adapted",
                            "volatility_adapted": True,
                            "few_shot_adapted": True,
                            "test_bars": len(y_target_test),
                            **m_few,
                        })

                    # -------------------------------------------------------------
                    # Mode 4: Asset-Specific Baseline (Counterfactual Baseline)
                    # -------------------------------------------------------------
                    if "asset_specific_baseline" in target_modes:
                        model_base_sym = source_factory()
                        model_base_sym.fit(s_target["X_train"], s_target["y_train"])
                        preds_sym = model_base_sym.predict(s_target["X_test"])
                        m_sym = self.calculate_metrics(y_target_test, preds_sym, y_target_train_mean)
                        results_records.append({
                            "fold_id": fold.fold_id,
                            "fold_name": fold.name,
                            "target_asset": target_asset,
                            "source_assets": target_asset,
                            "model_family": model_name,
                            "transfer_mode": "asset_specific_baseline",
                            "volatility_adapted": False,
                            "few_shot_adapted": False,
                            "test_bars": len(y_target_test),
                            **m_sym,
                        })

        df_results = pd.DataFrame(results_records)

        # Baseline Relative Improvements:
        # 1. vs ZeroReturn
        zero_rmse_map = (
            df_results[df_results["model_family"] == "ZeroReturn"]
            .set_index(["fold_id", "target_asset"])["rmse"]
            .to_dict()
        )
        df_results["zero_rmse"] = df_results.apply(
            lambda r: zero_rmse_map.get((r["fold_id"], r["target_asset"]), np.nan), axis=1
        )
        df_results["improvement_vs_zero_pct"] = (
            (df_results["zero_rmse"] - df_results["rmse"]) / df_results["zero_rmse"]
        ) * 100.0

        # 2. vs Asset-Specific Baseline (Same fold, model, and target asset)
        baseline_rmse_map = (
            df_results[df_results["transfer_mode"] == "asset_specific_baseline"]
            .set_index(["fold_id", "model_family", "target_asset"])["rmse"]
            .to_dict()
        )
        df_results["baseline_rmse"] = df_results.apply(
            lambda r: baseline_rmse_map.get((r["fold_id"], r["model_family"], r["target_asset"]), np.nan),
            axis=1,
        )
        df_results["improvement_vs_baseline_pct"] = (
            (df_results["baseline_rmse"] - df_results["rmse"]) / df_results["baseline_rmse"]
        ) * 100.0

        # 3. Adaptation Gain: vs Zero-Shot Raw
        raw_rmse_map = (
            df_results[df_results["transfer_mode"] == "zero_shot_raw"]
            .set_index(["fold_id", "model_family", "target_asset"])["rmse"]
            .to_dict()
        )
        df_results["raw_rmse"] = df_results.apply(
            lambda r: raw_rmse_map.get((r["fold_id"], r["model_family"], r["target_asset"]), np.nan),
            axis=1,
        )
        df_results["gain_vs_raw_pct"] = (
            (df_results["raw_rmse"] - df_results["rmse"]) / df_results["raw_rmse"]
        ) * 100.0

        # Save artifacts
        out_csv = self.artifacts_dir / "phase3_5_transfer_results.csv"
        df_results.to_csv(out_csv, index=False)
        meta_dict = {
            "total_evaluations": len(df_results),
            "target_assets": target_assets_list,
            "transfer_modes": target_modes,
            "models": target_models,
            "n_folds": len(folds),
            "few_shot_bars": self.few_shot_bars,
            "sha256": compute_file_checksum(out_csv),
            "generated_at_utc": str(pd.Timestamp.now(tz="UTC")),
        }
        with open(out_csv.with_name(f"{out_csv.name}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2, default=str)

        logger.info(
            f"Successfully executed Phase 3.5 Transfer Learning Benchmarking ({len(df_results)} evaluations). "
            f"Results saved to: {out_csv}"
        )
        return df_results
