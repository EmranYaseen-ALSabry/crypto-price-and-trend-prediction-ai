"""Forecasting models: baselines, linear regressions, tree ensembles, gradient boosting, and LSTM."""

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

__all__ = [
    "BaseModel",
    "LastValueBaseline",
    "ZeroReturnBaseline",
    "MeanReturnBaseline",
    "LinearRegressionModel",
    "RidgeModel",
    "RandomForestModel",
    "HistGradientBoostingModel",
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
    "LSTMModel",
]
