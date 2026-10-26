"""Decision-tree ensemble forecasting models: Random Forest and HistGradientBoosting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from crypto_prediction.models.base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest regressor model with tree ensembles."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 6,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(name="RandomForest")
        self.regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> RandomForestModel:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        self.regressor.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.regressor.predict(X)

    def get_feature_importances(self) -> np.ndarray | None:
        return self.regressor.feature_importances_


class HistGradientBoostingModel(BaseModel):
    """Scikit-Learn Histogram-based Gradient Boosting Regressor."""

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        random_state: int = 42,
    ) -> None:
        super().__init__(name="HistGradientBoosting")
        self.regressor = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> HistGradientBoostingModel:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        self.regressor.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.regressor.predict(X)
