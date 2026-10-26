"""Crypto Price Prediction AI: Production-grade time-series forecasting system for cryptocurrencies."""

__version__ = "1.0.0"

from crypto_prediction.data.cleaner import DataCleaner
from crypto_prediction.data.collector import DataCollector
from crypto_prediction.data.validator import DataValidator, ValidationReport
from crypto_prediction.features.engineer import FeatureEngineer
from crypto_prediction.features.registry import FeatureMetadata, FeatureRegistry
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
from crypto_prediction.pipeline.evaluator import EvaluationMetrics, ModelEvaluator
from crypto_prediction.pipeline.trainer import PipelineTrainer
from crypto_prediction.validation.leakage_guard import LeakageAuditReport, LeakageGuard
from crypto_prediction.validation.splitter import (
    ChronologicalSplitter,
    DatasetSplits,
    WalkForwardSplitter,
)

__all__ = [
    "__version__",
    "DataCollector",
    "DataValidator",
    "ValidationReport",
    "DataCleaner",
    "FeatureEngineer",
    "FeatureRegistry",
    "FeatureMetadata",
    "BaseModel",
    "ZeroReturnBaseline",
    "LastValueBaseline",
    "MeanReturnBaseline",
    "LinearRegressionModel",
    "RidgeModel",
    "RandomForestModel",
    "HistGradientBoostingModel",
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
    "LSTMModel",
    "ChronologicalSplitter",
    "WalkForwardSplitter",
    "DatasetSplits",
    "LeakageGuard",
    "LeakageAuditReport",
    "ModelEvaluator",
    "EvaluationMetrics",
    "PipelineTrainer",
]
