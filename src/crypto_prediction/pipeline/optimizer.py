"""Hyperparameter optimization using Optuna and causal walk-forward cross-validation."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from crypto_prediction.models.boosters import LightGBMModel
from crypto_prediction.utils.logger import get_logger
from crypto_prediction.validation.splitter import WalkForwardSplitter

logger = get_logger("optimizer")

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class HyperparameterOptimizer:
    """Optimizes forecasting model hyperparameters using Optuna and purged walk-forward CV."""

    def __init__(
        self,
        n_trials: int = 20,
        timeout: int = 180,
        n_splits: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_splits = n_splits
        self.random_state = random_state

        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna is not installed. Hyperparameter optimization will be disabled.")

    def optimize_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> dict[str, Any]:
        """Performs Bayesian optimization on LightGBM hyperparameters.

        Args:
            X_train: Training features.
            y_train: Training target.

        Returns:
            Dictionary of best hyperparameters found.
        """
        if not OPTUNA_AVAILABLE:
            logger.info("Optuna unavailable. Returning default LightGBM hyperparameters.")
            return {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "num_leaves": 15}

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        splitter = WalkForwardSplitter(
            n_splits=self.n_splits,
            train_window_bars=min(500, int(len(X_train) * 0.5)),
            val_window_bars=min(150, int(len(X_train) * 0.15)),
            step_bars=min(100, int(len(X_train) * 0.1)),
            purge_bars=24,
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "num_leaves": trial.suggest_int("num_leaves", 10, 35),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": self.random_state,
                "verbose": -1,
            }

            cv_rmses: list[float] = []
            for tr_idx, v_idx in splitter.split(X_train):
                X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_v, y_v = X_train.iloc[v_idx], y_train.iloc[v_idx]

                model = LightGBMModel(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_v)
                rmse = float(np.sqrt(mean_squared_error(y_v, preds)))
                cv_rmses.append(rmse)

            return float(np.mean(cv_rmses)) if cv_rmses else 999.0

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        # Reference: CV RMSE of always predicting the training-window mean return.
        baseline_rmses: list[float] = []
        for tr_idx, v_idx in splitter.split(X_train):
            y_tr, y_v = y_train.iloc[tr_idx], y_train.iloc[v_idx]
            baseline_rmses.append(float(np.sqrt(np.mean((y_v.to_numpy() - y_tr.mean()) ** 2))))
        baseline_rmse = float(np.mean(baseline_rmses)) if baseline_rmses else float("nan")

        logger.info(f"Optimization completed. Best CV RMSE: {study.best_value:.6f} (mean-return baseline: {baseline_rmse:.6f})")
        if not np.isnan(baseline_rmse) and study.best_value >= baseline_rmse:
            logger.warning("Tuned LightGBM does not beat the mean-return baseline in CV; treat its forecasts as no better than noise.")
        return study.best_params
