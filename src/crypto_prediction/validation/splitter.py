"""Chronological and Walk-Forward Time-Series Splitters with Purging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator
import numpy as np
import pandas as pd


@dataclass
class DatasetSplits:
    """Container holding strictly partitioned train, validation, and test sets."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    meta_train: pd.DataFrame
    meta_val: pd.DataFrame
    meta_test: pd.DataFrame


class ChronologicalSplitter:
    """Partitions time series chronologically into Train, Validation, and Test partitions

    with optional purging gap to eliminate multi-step target overlap.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        purge_bars: int = 0,
        embargo_bars: int = 0,
    ) -> None:
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0, atol=1e-5):
            raise ValueError(f"Split ratios must sum to 1.0 (got {total:.4f}).")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.total_gap = purge_bars + embargo_bars

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: pd.DataFrame | None = None,
    ) -> DatasetSplits:
        """Splits datasets chronologically.

        Args:
            X: Input feature DataFrame.
            y: Target Series.
            metadata: Associated metadata DataFrame (e.g. underlying prices).

        Returns:
            DatasetSplits object with partitioned subsets.
        """
        n = len(X)
        if n < 50:
            raise ValueError(f"Insufficient samples for reliable 3-way time-series split (got {n}).")

        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        # Train partition
        train_end = n_train

        # Val partition begins after total_gap (purge + embargo)
        val_start = min(train_end + self.total_gap, n - 2)
        val_end = min(val_start + n_val, n - 1)

        # Test partition begins after total_gap from val_end
        test_start = min(val_end + self.total_gap, n)

        if val_start >= val_end or test_start >= n:
            raise ValueError(
                f"Dataset size ({n}) too small for requested split ratios and total_gap={self.total_gap}."
            )

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[val_start:val_end], y.iloc[val_start:val_end]
        X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]

        meta = metadata if metadata is not None else pd.DataFrame(index=X.index)
        meta_train = meta.iloc[:train_end]
        meta_val = meta.iloc[val_start:val_end]
        meta_test = meta.iloc[test_start:]

        return DatasetSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
        )


class WalkForwardSplitter:
    """Expanding or rolling walk-forward time-series CV splitter with purging and embargo."""

    def __init__(
        self,
        n_splits: int = 4,
        train_window_bars: int = 800,
        val_window_bars: int = 200,
        step_bars: int = 150,
        purge_bars: int = 24,
        embargo_bars: int = 0,
        expanding: bool = True,
    ) -> None:
        self.n_splits = n_splits
        self.train_window_bars = train_window_bars
        self.val_window_bars = val_window_bars
        self.step_bars = step_bars
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.total_gap = purge_bars + embargo_bars
        self.expanding = expanding

    def split(
        self,
        X: pd.DataFrame,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yields (train_indices, val_indices) for each walk-forward fold."""
        n = len(X)
        for i in range(self.n_splits):
            val_end = n - (self.n_splits - 1 - i) * self.step_bars
            val_start = val_end - self.val_window_bars
            train_end = val_start - self.total_gap

            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, train_end - self.train_window_bars)

            if train_end <= train_start or val_start <= 0 or val_end > n:
                continue

            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)

            yield train_idx, val_idx
