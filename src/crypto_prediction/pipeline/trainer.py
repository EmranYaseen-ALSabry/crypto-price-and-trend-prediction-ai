"""End-to-end training, auditing, benchmarking, and artifacts pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd
import sklearn

from crypto_prediction.data.cleaner import DataCleaner
from crypto_prediction.data.collector import DataCollector
from crypto_prediction.data.validator import DataValidator
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
from crypto_prediction.pipeline.backtester import StrategyBacktester
from crypto_prediction.pipeline.evaluator import EvaluationMetrics, ModelEvaluator
from crypto_prediction.pipeline.optimizer import HyperparameterOptimizer
from crypto_prediction.utils.logger import get_logger
from crypto_prediction.utils.reproducibility import set_seed
from crypto_prediction.utils.visualization import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_model_comparison_bar,
    plot_price_and_volume,
    plot_residuals_distribution,
)
from crypto_prediction.validation.leakage_guard import LeakageGuard
from crypto_prediction.validation.splitter import ChronologicalSplitter

logger = get_logger("pipeline_trainer")


class PipelineTrainer:
    """Orchestrates end-to-end training and evaluation of forecasting models."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        artifacts_dir: str | Path = "artifacts",
    ) -> None:
        self.config = config or {}
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = self.artifacts_dir / "models"
        self.figures_dir = self.artifacts_dir / "figures"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.seed = self.config.get("random_seed", 42)
        set_seed(self.seed)

    def instantiate_models(self, custom_params: dict[str, dict[str, Any]] | None = None) -> dict[str, BaseModel]:
        """Instantiates all supported baselines, regression, and gradient boosting models."""
        custom = custom_params or {}
        models: dict[str, BaseModel] = {
            "ZeroReturnBaseline": ZeroReturnBaseline(),
            "LastValueBaseline": LastValueBaseline(),
            "MeanReturnBaseline": MeanReturnBaseline(),
            "LinearRegression": LinearRegressionModel(),
            "Ridge": RidgeModel(alpha=1.0, random_state=self.seed),
            "RandomForest": RandomForestModel(n_estimators=100, max_depth=5, random_state=self.seed),
            "HistGradientBoosting": HistGradientBoostingModel(max_iter=100, learning_rate=0.03, random_state=self.seed),
        }

        # Boosters (if available)
        try:
            xgb_kwargs = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "random_state": self.seed}
            xgb_kwargs.update(custom.get("XGBoost", {}))
            models["XGBoost"] = XGBoostModel(**xgb_kwargs)
        except Exception as e:
            logger.warning(f"Could not initialize XGBoost: {e}")

        try:
            lgb_kwargs = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "random_state": self.seed}
            lgb_kwargs.update(custom.get("LightGBM", {}))
            models["LightGBM"] = LightGBMModel(**lgb_kwargs)
        except Exception as e:
            logger.warning(f"Could not initialize LightGBM: {e}")

        try:
            cat_kwargs = {"iterations": 100, "learning_rate": 0.03, "depth": 4, "random_seed": self.seed}
            cat_kwargs.update(custom.get("CatBoost", {}))
            models["CatBoost"] = CatBoostModel(**cat_kwargs)
        except Exception as e:
            logger.warning(f"Could not initialize CatBoost: {e}")

        # LSTM (PyTorch or explicit fallback)
        allow_fallback = self.config.get("allow_mlp_fallback", False)
        try:
            lstm_model = LSTMModel(
                sequence_length=12,
                hidden_dim=32,
                epochs=15,
                allow_mlp_fallback=allow_fallback,
                random_state=self.seed,
            )
            key_name = "LSTM" if getattr(lstm_model, "is_genuine_lstm", False) else "MLP_Fallback"
            models[key_name] = lstm_model
        except ImportError as e:
            logger.info(f"LSTM not instantiated: {e}")
        except Exception as e:
            logger.warning(f"Could not initialize LSTM: {e}")

        return models

    def run_experiment(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        horizon: int = 1,
        total_bars: int = 1500,
        optimize_hyperparams: bool = False,
    ) -> dict[str, Any]:
        """Executes full experiment workflow:

        1. Ingestion & Validation
        2. Cleaning
        3. Feature engineering
        4. Leakage audit
        5. Chronological splitting
        6. Model training & Out-of-sample benchmarking
        7. Artifact generation

        Returns:
            Dictionary containing metrics, evaluation tables, and audit reports.
        """
        logger.info(f"=== Starting Experiment for {symbol} ({interval}, horizon={horizon}h) ===")

        # 1. Ingestion
        collector = DataCollector()
        raw_df = collector.fetch_historical_bars(symbol=symbol, interval=interval, total_bars=total_bars)

        # 2. Validation
        validator = DataValidator(expected_interval=interval)
        val_report = validator.validate(raw_df)
        logger.info(val_report.summary())
        if not val_report.is_valid:
            raise ValueError(f"Data validation failed for {symbol}: {val_report.errors}")

        # 3. Cleaning
        cleaner = DataCleaner()
        clean_df = cleaner.clean(raw_df)

        # Plot price & volume
        plot_price_and_volume(
            clean_df,
            symbol=symbol,
            save_path=self.figures_dir / f"{symbol}_{interval}_price_volume.png",
        )

        # 4. Feature Engineering
        engineer = FeatureEngineer(
            primary_horizon=horizon,
            forecast_horizons=[1, 4, 24],
        )
        X, y, metadata = engineer.prepare_dataset(clean_df, target_horizon=horizon)

        # 5. Chronological Splitting with Purging
        splitter = ChronologicalSplitter(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=horizon,  # Purge gap equal to horizon
        )
        splits = splitter.split(X, y, metadata=metadata)

        # 6. Leakage Audit
        guard = LeakageGuard()
        audit_report = guard.run_full_audit(
            df_raw=clean_df,
            feature_engineer=engineer,
            train_index=splits.X_train.index,
            val_index=splits.X_val.index,
            test_index=splits.X_test.index,
            required_purge_bars=horizon,
            full_index=X.index,
        )
        logger.info(audit_report.summary())
        if not audit_report.passed_all:
            raise RuntimeError(f"Data leakage audit failed: {audit_report.details}")

        # Optional hyperparameter tuning for LightGBM
        custom_params: dict[str, dict[str, Any]] = {}
        if optimize_hyperparams:
            optimizer = HyperparameterOptimizer(n_trials=15, timeout=120)
            best_lgb_params = optimizer.optimize_lightgbm(splits.X_train, splits.y_train)
            logger.info(f"Optimized LightGBM params: {best_lgb_params}")
            custom_params["LightGBM"] = best_lgb_params

        # 7. Model Training & Evaluation
        models = self.instantiate_models(custom_params=custom_params)
        evaluator = ModelEvaluator()
        all_metrics: list[EvaluationMetrics] = []
        predictions: dict[str, dict[str, np.ndarray]] = {}

        # Extract genuine contiguous historical warmup from the continuous timeline
        val_idx_loc = X.index.get_loc(splits.X_val.index[0])
        val_warmup = X.iloc[max(0, val_idx_loc - 12) : val_idx_loc]
        test_idx_loc = X.index.get_loc(splits.X_test.index[0])
        test_warmup = X.iloc[max(0, test_idx_loc - 12) : test_idx_loc]

        for name, model in models.items():
            logger.info(f"Training {name}...")
            # Fit strictly on training partition
            model.fit(splits.X_train, splits.y_train)

            # Save model artifact using documented naming convention: {symbol}_{interval}_{name}_h{horizon}.joblib
            artifact_stem = f"{symbol}_{interval}_{name}_h{horizon}"
            model_path = self.models_dir / f"{artifact_stem}.joblib"
            model.save(model_path)

            # Save metadata sidecar
            meta_path = self.models_dir / f"{artifact_stem}_meta.json"
            model_meta = {
                "model_name": name,
                "model_class": model.__class__.__name__,
                "symbol": symbol,
                "interval": interval,
                "horizon": horizon,
                "feature_names": list(splits.X_train.columns),
                "feature_count": len(splits.X_train.columns),
                "is_genuine_lstm": getattr(model, "is_genuine_lstm", False) if ("LSTM" in name or "MLP" in name) else None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(model_meta, mf, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save sidecar metadata for {name}: {e}")

            # Out-of-sample predictions
            preds_train = model.predict(splits.X_train)
            if getattr(model, "is_genuine_lstm", False):
                preds_val = model.predict(splits.X_val, warmup_history=val_warmup)
                preds_test = model.predict(splits.X_test, warmup_history=test_warmup)
            else:
                preds_val = model.predict(splits.X_val)
                preds_test = model.predict(splits.X_test)

            predictions[name] = {
                "train": preds_train,
                "val": preds_val,
                "test": preds_test,
            }

            # Evaluate on Test
            # Price reconstruction: P_hat_{t+h} = Close_t * (1 + r_hat_{t+h})
            test_metrics = evaluator.evaluate(
                y_true=splits.y_test,
                y_pred=preds_test,
                model_name=name,
                horizon=horizon,
                split_name="test",
                current_close=splits.meta_test["close"],
                true_future_price=splits.meta_test[f"target_price_{horizon}"],
            )
            all_metrics.append(test_metrics)

            # Evaluate on Train
            train_metrics = evaluator.evaluate(
                y_true=splits.y_train,
                y_pred=preds_train,
                model_name=name,
                horizon=horizon,
                split_name="train",
                current_close=splits.meta_train["close"],
                true_future_price=splits.meta_train[f"target_price_{horizon}"],
            )
            all_metrics.append(train_metrics)

            # Evaluate on Validation
            val_metrics = evaluator.evaluate(
                y_true=splits.y_val,
                y_pred=preds_val,
                model_name=name,
                horizon=horizon,
                split_name="validation",
                current_close=splits.meta_val["close"],
                true_future_price=splits.meta_val[f"target_price_{horizon}"],
            )
            all_metrics.append(val_metrics)

        # 8. Comparison table & figures
        comparison_df = evaluator.build_comparison_table(all_metrics)
        train_comparison = comparison_df[comparison_df["split_name"] == "train"].copy()
        val_comparison = comparison_df[comparison_df["split_name"] == "validation"].copy()
        test_comparison = comparison_df[comparison_df["split_name"] == "test"].copy()

        train_comparison.to_csv(self.artifacts_dir / f"{symbol}_train_metrics_h{horizon}.csv", index=False)
        val_comparison.to_csv(self.artifacts_dir / f"{symbol}_val_metrics_h{horizon}.csv", index=False)
        test_comparison.to_csv(self.artifacts_dir / f"{symbol}_test_metrics_h{horizon}.csv", index=False)

        # Comparative Bar Chart (Validation Set)
        plot_model_comparison_bar(
            val_comparison,
            metric="rmse",
            save_path=self.figures_dir / f"{symbol}_model_comparison_rmse_h{horizon}.png",
        )

        # Select best model strictly on Validation set to keep Test set completely unbiased
        best_model_name = val_comparison.sort_values(by="rmse").iloc[0]["model_name"]
        logger.info(f"Selected best model based on Validation RMSE: {best_model_name}")
        best_preds = predictions[best_model_name]["test"]

        plot_actual_vs_predicted(
            timestamps=splits.X_test.index,
            actual=splits.y_test,
            predicted=best_preds,
            model_name=best_model_name,
            target_type="Forward Return",
            save_path=self.figures_dir / f"{symbol}_{best_model_name}_actual_vs_pred_h{horizon}.png",
        )

        # Residuals
        residuals = splits.y_test.to_numpy() - best_preds
        plot_residuals_distribution(
            residuals=residuals,
            model_name=best_model_name,
            save_path=self.figures_dir / f"{symbol}_{best_model_name}_residuals_h{horizon}.png",
        )

        # Feature importance (if available on best model)
        best_model = models[best_model_name]
        importances = best_model.get_feature_importances()
        if importances is not None and len(importances) == len(splits.X_train.columns):
            plot_feature_importance(
                feature_names=list(splits.X_train.columns),
                importances=importances,
                model_name=best_model_name,
                top_n=15,
                save_path=self.figures_dir / f"{symbol}_{best_model_name}_feat_importance_h{horizon}.png",
            )

        # 9. Overfitting Diagnostics (Train vs Val vs Test RMSE Comparison)
        overfitting_df = evaluator.build_overfitting_diagnostics(all_metrics)
        overfitting_df.to_csv(self.artifacts_dir / f"{symbol}_overfitting_diagnostics_h{horizon}.csv", index=False)
        logger.info(f"Saved overfitting diagnostics to {symbol}_overfitting_diagnostics_h{horizon}.csv")

        # 10. Realistic Out-of-Sample Trading Simulation (Backtest with fees & slippage)
        backtester = StrategyBacktester(fee_rate=0.0010, slippage_rate=0.0005)
        backtest_records = []
        for name in models:
            test_preds_m = predictions[name]["test"]
            bt_res = backtester.run_backtest(
                y_true=splits.y_test,
                y_pred=test_preds_m,
                model_name=name,
                horizon=horizon,
            )
            backtest_records.append(bt_res.to_dict())

        backtest_df = pd.DataFrame(backtest_records)
        backtest_df.to_csv(self.artifacts_dir / f"{symbol}_trading_backtest_h{horizon}.csv", index=False)
        logger.info(f"Saved realistic trading backtest report to {symbol}_trading_backtest_h{horizon}.csv")

        # 11. Experiment Metadata & Data Provenance
        data_source = str(raw_df.attrs.get("data_source", "unknown"))
        is_synthetic = bool(raw_df.attrs.get("is_synthetic", False))
        meta_info = {
            "experiment_id": f"EXP_{symbol}_{horizon}h_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "interval": interval,
            "horizon": horizon,
            "data_source": data_source,
            "is_synthetic": is_synthetic,
            "market_claims_allowed": not is_synthetic,
            "start_time": str(clean_df.index[0]) if len(clean_df) > 0 else "",
            "last_bar_timestamp": str(clean_df.index[-1]) if len(clean_df) > 0 else "",
            "raw_row_count": len(raw_df),
            "clean_row_count": len(clean_df),
            "feature_count": len(X.columns),
            "train_row_count": len(splits.X_train),
            "val_row_count": len(splits.X_val),
            "test_row_count": len(splits.X_test),
            "random_seed": self.seed,
            "library_versions": {
                "python": sys.version.split()[0],
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
            },
            "best_model_by_validation_rmse": best_model_name,
        }
        if is_synthetic:
            logger.warning("[DATA POLICY ENFORCED] Synthetic dataset detected. Market claims are prohibited.")

        meta_path = self.artifacts_dir / f"{symbol}_experiment_meta_h{horizon}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_info, f, indent=2)
        logger.info(f"Saved experiment metadata to {meta_path.name}")

        logger.info(f"=== Completed Experiment for {symbol}. Best Model: {best_model_name} ===")
        return {
            "symbol": symbol,
            "horizon": horizon,
            "splits": splits,
            "models": models,
            "audit_report": audit_report,
            "comparison_df": comparison_df,
            "train_comparison": train_comparison,
            "val_comparison": val_comparison,
            "test_comparison": test_comparison,
            "overfitting_df": overfitting_df,
            "best_model_name": best_model_name,
            "predictions": predictions,
            "backtest_df": backtest_df,
            "experiment_meta": meta_info,
        }
