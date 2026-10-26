"""Base model interface for all time-series forecasting estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class defining standardized API for forecasting models."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.is_fitted = False
        self.feature_names: list[str] = []

    @abstractmethod
    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> BaseModel:
        """Trains the estimator on input features X and target y."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Generates point forecasts for input features X."""
        pass

    def get_feature_importances(self) -> np.ndarray | None:
        """Returns feature importance array if supported by estimator, else None."""
        return None

    def save(self, file_path: str | Path) -> Path:
        """Serializes the model to disk using joblib."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, file_path: str | Path) -> BaseModel:
        """Loads a serialized model instance from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")
        return joblib.load(path)
