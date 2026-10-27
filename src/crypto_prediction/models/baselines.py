"""Baseline forecasting models adhering to financial time-series standards."""

from __future__ import annotations

import numpy as np
import pandas as pd
from crypto_prediction.models.base import BaseModel


class ZeroReturnBaseline(BaseModel):
    """Zero-return baseline predicting strictly zero return (no price change) for all periods."""

    def __init__(self) -> None:
        super().__init__(name="ZeroReturnBaseline")

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> ZeroReturnBaseline:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        n_samples = len(X)
        return np.zeros(n_samples, dtype=np.float64)


class LastValueBaseline(BaseModel):
    """Last-value persistence baseline.

    When forecasting returns: predicting that price will not change is mathematically equivalent to 0.0 return.
    When predicting price directly: outputs the most recent available price.
    """

    def __init__(self, price_col_idx: int = 0) -> None:
        super().__init__(name="LastValueBaseline")
        self.price_col_idx = price_col_idx

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> LastValueBaseline:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        # For return prediction, persistence of price implies 0.0 return
        return np.zeros(len(X), dtype=np.float64)


class MeanReturnBaseline(BaseModel):
    """Historical mean baseline predicting the mean return observed on the training partition."""

    def __init__(self) -> None:
        super().__init__(name="MeanReturnBaseline")
        self.mean_return: float = 0.0

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> MeanReturnBaseline:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        y_arr = np.asarray(y, dtype=np.float64)
        if sample_weight is not None:
            w_arr = np.asarray(sample_weight, dtype=np.float64)
            valid = (~np.isnan(y_arr)) & (~np.isnan(w_arr))
            if np.sum(w_arr[valid]) > 0:
                self.mean_return = float(np.average(y_arr[valid], weights=w_arr[valid]))
            else:
                self.mean_return = float(np.nanmean(y_arr))
        else:
            self.mean_return = float(np.nanmean(y_arr))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        n_samples = len(X)
        return np.full(n_samples, self.mean_return, dtype=np.float64)
