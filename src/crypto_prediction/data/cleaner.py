"""Data cleaning and pre-processing pipeline for crypto OHLCV series."""

from __future__ import annotations

import pandas as pd
from crypto_prediction.utils.logger import get_logger

logger = get_logger("data_cleaner")


class DataCleaner:
    """Cleans, sorts, deduplicates, and aligns cryptocurrency OHLCV time series."""

    def __init__(self, max_consecutive_fill: int = 5) -> None:
        self.max_consecutive_fill = max_consecutive_fill

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the input OHLCV DataFrame without future data leakage.

        Args:
            df: Raw DataFrame containing OHLCV records.

        Returns:
            Cleaned DataFrame indexed by UTC DatetimeIndex.
        """
        if df.empty:
            raise ValueError("Cannot clean an empty DataFrame.")

        cleaned = df.copy()

        # Parse timestamp column
        if "open_time" in cleaned.columns:
            cleaned["open_time"] = pd.to_datetime(cleaned["open_time"], utc=True)
        elif not isinstance(cleaned.index, pd.DatetimeIndex):
            raise KeyError("DataFrame must contain 'open_time' column or have a DatetimeIndex.")

        # If open_time is a column, sort and deduplicate before setting index
        if "open_time" in cleaned.columns:
            # Sort chronologically
            cleaned = cleaned.sort_values(by="open_time").reset_index(drop=True)

            # Drop duplicates by open_time (keep the last recorded observation)
            initial_count = len(cleaned)
            cleaned = cleaned.drop_duplicates(subset=["open_time"], keep="last")
            dropped_dups = initial_count - len(cleaned)
            if dropped_dups > 0:
                logger.info(f"Dropped {dropped_dups} duplicate timestamp rows.")

            cleaned = cleaned.set_index("open_time")

        # Sort index chronologically to guarantee monotonicity
        cleaned = cleaned.sort_index()

        # Enforce numeric types
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

        # Remove rows where close or open is non-positive or NaN
        valid_price_mask = (cleaned["close"] > 0) & (cleaned["open"] > 0)
        invalid_prices = (~valid_price_mask).sum()
        if invalid_prices > 0:
            logger.warning(f"Dropping {invalid_prices} rows with invalid/non-positive close or open prices.")
            cleaned = cleaned[valid_price_mask]

        # Handle missing volume: replace negative or NaN volume with 0.0
        if "volume" in cleaned.columns:
            cleaned["volume"] = cleaned["volume"].fillna(0.0).clip(lower=0.0)

        # Forward fill prices within maximum allowed threshold (never backward fill!)
        # Backward filling would introduce future leakage!
        price_cols = [c for c in ["open", "high", "low", "close"] if c in cleaned.columns]
        cleaned[price_cols] = cleaned[price_cols].ffill(limit=self.max_consecutive_fill)

        # Drop any remaining unfillable NaNs at the very beginning of the series
        cleaned = cleaned.dropna(subset=price_cols)

        # Ensure high >= max(open, close) and low <= min(open, close)
        cleaned["high"] = cleaned[["high", "open", "close"]].max(axis=1)
        cleaned["low"] = cleaned[["low", "open", "close"]].min(axis=1)

        return cleaned
