"""Gradient boosting estimators: XGBoost, LightGBM, and CatBoost."""

from __future__ import annotations

import numpy as np
import pandas as pd
from crypto_prediction.models.base import BaseModel
from crypto_prediction.utils.logger import get_logger

logger = get_logger("boosters")

# Optional imports with graceful availability detection
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class XGBoostModel(BaseModel):
    """XGBoost Regressor for financial time-series forecasting."""

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: float = 1.0,
        standardize_target: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(name="XGBoost")
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost package is not installed. Install via pip install xgboost.")

        self.standardize_target = standardize_target
        self.regressor = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> XGBoostModel:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        y_arr = np.asarray(y, dtype=np.float64)
        if self.standardize_target:
            self._y_mean = float(y_arr.mean())
            self._y_std = float(y_arr.std())
            if self._y_std < 1e-12:
                self._y_std = 1.0
            y_target = (y_arr - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
            y_target = y_arr

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float64)

        self.regressor.fit(X, y_target, **fit_kwargs)
        self.is_fitted = True

        # Diagnostic check for split collapse
        diag = self.get_split_diagnostics(X)
        if diag.get("is_collapsed_to_mean") or diag.get("is_constant_prediction"):
            logger.warning(
                f"[XGBoost DIAGNOSTIC ALERT] Model collapsed to constant/mean predictions "
                f"(total splits: {diag.get('total_splits', 0)})."
            )

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        y_std = getattr(self, "_y_std", 1.0)
        y_mean = getattr(self, "_y_mean", 0.0)
        return self.regressor.predict(X) * y_std + y_mean

    def get_feature_importances(self) -> np.ndarray | None:
        return self.regressor.feature_importances_

    def get_split_diagnostics(self, X: np.ndarray | pd.DataFrame | None = None) -> dict[str, Any]:
        """Inspects fitted tree structure to verify non-zero split counts and non-constant predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        booster = self.regressor.get_booster()
        dump = booster.get_dump()
        total_trees = len(dump)
        leaf_only_trees = sum(1 for t in dump if t.strip().startswith("0:leaf="))
        active_trees = total_trees - leaf_only_trees
        total_splits = sum(t.count("[") for t in dump)

        diagnostics: dict[str, Any] = {
            "model_name": self.name,
            "total_trees": total_trees,
            "active_split_trees": active_trees,
            "leaf_only_trees": leaf_only_trees,
            "total_splits": total_splits,
            "is_collapsed_to_mean": total_splits == 0,
        }

        if X is not None:
            preds = self.predict(X)
            pred_var = float(np.var(preds))
            pred_range = float(np.ptp(preds))
            is_const = bool(pred_range < 1e-9 or pred_var < 1e-12)
            diagnostics.update({
                "prediction_variance": pred_var,
                "prediction_range": pred_range,
                "is_constant_prediction": is_const,
                "is_degenerate_constant": is_const,
            })
        return diagnostics


class LightGBMModel(BaseModel):
    """LightGBM Regressor for fast gradient boosting."""

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        num_leaves: int = 15,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_samples: int = 15,
        min_split_gain: float = 0.0,
        standardize_target: bool = True,
        random_state: int = 42,
        verbose: int = -1,
    ) -> None:
        super().__init__(name="LightGBM")
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm package is not installed. Install via pip install lightgbm.")

        self.standardize_target = standardize_target
        self.regressor = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_samples=min_child_samples,
            min_split_gain=min_split_gain,
            random_state=random_state,
            verbose=verbose,
        )
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> LightGBMModel:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        y_arr = np.asarray(y, dtype=np.float64)
        if self.standardize_target:
            self._y_mean = float(y_arr.mean())
            self._y_std = float(y_arr.std())
            if self._y_std < 1e-12:
                self._y_std = 1.0
            y_target = (y_arr - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
            y_target = y_arr

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float64)

        self.regressor.fit(X, y_target, **fit_kwargs)
        self.is_fitted = True

        # Diagnostic check for split collapse
        diag = self.get_split_diagnostics(X)
        if diag.get("is_collapsed_to_mean") or diag.get("is_constant_prediction"):
            logger.warning(
                f"[LightGBM DIAGNOSTIC ALERT] Model collapsed to constant/mean predictions "
                f"(total splits: {diag.get('total_splits', 0)})."
            )

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        y_std = getattr(self, "_y_std", 1.0)
        y_mean = getattr(self, "_y_mean", 0.0)
        return self.regressor.predict(X) * y_std + y_mean

    def get_feature_importances(self) -> np.ndarray | None:
        return self.regressor.feature_importances_

    def get_split_diagnostics(self, X: np.ndarray | pd.DataFrame | None = None) -> dict[str, Any]:
        """Inspects fitted LightGBM booster structure for non-zero split counts and non-constant predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        booster = self.regressor.booster_
        dump = booster.dump_model()
        tree_info = dump.get("tree_info", [])
        total_trees = len(tree_info)
        total_splits = sum(t.get("num_leaves", 1) - 1 for t in tree_info)
        leaf_only_trees = sum(1 for t in tree_info if t.get("num_leaves", 1) <= 1)
        active_trees = total_trees - leaf_only_trees

        diagnostics: dict[str, Any] = {
            "model_name": self.name,
            "total_trees": total_trees,
            "active_split_trees": active_trees,
            "leaf_only_trees": leaf_only_trees,
            "total_splits": total_splits,
            "is_collapsed_to_mean": total_splits == 0,
        }

        if X is not None:
            preds = self.predict(X)
            pred_var = float(np.var(preds))
            pred_range = float(np.ptp(preds))
            is_const = bool(pred_range < 1e-9 or pred_var < 1e-12)
            diagnostics.update({
                "prediction_variance": pred_var,
                "prediction_range": pred_range,
                "is_constant_prediction": is_const,
                "is_degenerate_constant": is_const,
            })
        return diagnostics


class CatBoostModel(BaseModel):
    """CatBoost Regressor with symmetric decision trees."""

    def __init__(
        self,
        iterations: int = 150,
        learning_rate: float = 0.03,
        depth: int = 4,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        verbose: int = 0,
    ) -> None:
        super().__init__(name="CatBoost")
        if not CATBOOST_AVAILABLE:
            raise ImportError("catboost package is not installed. Install via pip install catboost.")

        self.regressor = cb.CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            verbose=verbose,
        )

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> CatBoostModel:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float64)
        self.regressor.fit(X, y, **fit_kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.regressor.predict(X)

    def get_feature_importances(self) -> np.ndarray | None:
        return self.regressor.get_feature_importance()

    def get_split_diagnostics(self, X: np.ndarray | pd.DataFrame | None = None) -> dict[str, Any]:
        """Inspects fitted CatBoost model structure for non-zero split counts and non-constant predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        tree_count = self.regressor.get_tree_count()
        diagnostics: dict[str, Any] = {
            "model_name": self.name,
            "total_trees": tree_count,
            "is_collapsed_to_mean": tree_count == 0,
        }
        if X is not None:
            preds = self.predict(X)
            pred_var = float(np.var(preds))
            pred_range = float(np.ptp(preds))
            is_const = bool(pred_range < 1e-9 or pred_var < 1e-12)
            diagnostics.update({
                "prediction_variance": pred_var,
                "prediction_range": pred_range,
                "is_constant_prediction": is_const,
                "is_degenerate_constant": is_const,
            })
        return diagnostics
