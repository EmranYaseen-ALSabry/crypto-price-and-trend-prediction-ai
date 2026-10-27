"""Shared pytest fixtures for Crypto Price Prediction AI test suite."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")

from crypto_prediction.data.collector import DataCollector


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Generates deterministic clean OHLCV DataFrame."""
    collector = DataCollector()
    df = collector.generate_synthetic_ohlcv(symbol="BTCUSDT", n_bars=600, seed=42)
    return df


@pytest.fixture
def corrupted_ohlcv_df(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Creates an OHLCV DataFrame with injected data defects for validation tests."""
    df = sample_ohlcv_df.copy()
    # 1. Non-positive price
    df.loc[10, "close"] = -50.0
    # 2. Negative volume
    df.loc[20, "volume"] = -10.0
    # 3. Invalid candlestick: High < Low
    df.loc[30, "high"] = df.loc[30, "low"] - 100.0
    # 4. Duplicate timestamp
    df.loc[40, "open_time"] = df.loc[39, "open_time"]
    # 5. Non-monotonic shuffle
    temp = df.iloc[50].copy()
    df.iloc[50] = df.iloc[52].copy()
    df.iloc[52] = temp
    return df
