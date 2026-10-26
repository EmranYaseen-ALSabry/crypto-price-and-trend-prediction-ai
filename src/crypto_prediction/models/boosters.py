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
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(name="XGBoost")
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost package is not installed. Install via pip install xgboost.")

        self.regressor = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> XGBoostModel:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        # Returns are ~1e-3 in scale, so L1/L2 penalties and min-hessian rules are
        # meaningless on the raw target. Standardise y (train statistics only).
        y_arr = np.asarray(y, dtype=np.float64)
        self._y_mean = float(y_arr.mean())
        self._y_std = float(y_arr.std()) or 1.0
        self.regressor.fit(X, (y_arr - self._y_mean) / self._y_std)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        y_std = getattr(self, "_y_std", 1.0)
        y_mean = getattr(self, "_y_mean", 0.0)
        return self.regressor.predict(X) * y_std + y_mean

    def get_feature_importances(self) -> np.ndarray | None:
        return self.regressor.feature_importances_


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
        random_state: int = 42,
        verbose: int = -1,
    ) -> None:
        super().__init__(name="LightGBM")
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm package is not installed. Install via pip install lightgbm.")

        self.regressor = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            verbose=verbose,
        )
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> LightGBMModel:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        # Returns are ~1e-3 in scale, so L1/L2 penalties and min-hessian rules are
        # meaningless on the raw target. Standardise y (train statistics only).
        y_arr = np.asarray(y, dtype=np.float64)
        self._y_mean = float(y_arr.mean())
        self._y_std = float(y_arr.std()) or 1.0
        self.regressor.fit(X, (y_arr - self._y_mean) / self._y_std)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        y_std = getattr(self, "_y_std", 1.0)
        y_mean = getattr(self, "_y_mean", 0.0)
        return self.regressor.predict(X) * y_std + y_mean

    def get_feature_importances(self) -> np.ndarray | None:
        return self.regressor.feature_importances_


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

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> CatBoostModel:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        self.regressor.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.regressor.predict(X)

    def get_feature_importances(self) -> np.ndarray | None:
        return self.regressor.get_feature_importance()
