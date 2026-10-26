"""Linear and regularized regression forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crypto_prediction.models.base import BaseModel


class LinearRegressionModel(BaseModel):
    """Ordinary Least Squares Linear Regression wrapped with StandardScaler."""

    def __init__(self, fit_intercept: bool = True) -> None:
        super().__init__(name="LinearRegression")
        self.fit_intercept = fit_intercept
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression(fit_intercept=fit_intercept)),
            ]
        )

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> LinearRegressionModel:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def get_feature_importances(self) -> np.ndarray | None:
        reg = self.pipeline.named_steps["regressor"]
        return np.abs(reg.coef_)


class RidgeModel(BaseModel):
    """L2 Regularized Ridge Regression wrapped with StandardScaler."""

    def __init__(self, alpha: float = 1.0, random_state: int = 42) -> None:
        super().__init__(name="Ridge")
        self.alpha = alpha
        self.random_state = random_state
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=alpha, random_state=random_state)),
            ]
        )

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> RidgeModel:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def get_feature_importances(self) -> np.ndarray | None:
        reg = self.pipeline.named_steps["regressor"]
        return np.abs(reg.coef_)
