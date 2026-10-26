"""Validation module for cryptocurrency OHLCV time-series datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class ValidationReport:
    """Structured report containing data validation diagnostics."""

    is_valid: bool
    total_rows: int
    missing_values: dict[str, int]
    duplicate_timestamps: int
    non_monotonic_timestamps: int
    invalid_ohlc_bars: int
    non_positive_prices: int
    negative_volumes: int
    timestamp_gaps: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Returns human-readable validation summary."""
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"=== Data Validation Report [{status}] ===",
            f"Total Rows: {self.total_rows}",
            f"Duplicate Timestamps: {self.duplicate_timestamps}",
            f"Non-Monotonic Timestamps: {self.non_monotonic_timestamps}",
            f"Invalid OHLC Bars: {self.invalid_ohlc_bars}",
            f"Non-Positive Prices: {self.non_positive_prices}",
            f"Negative Volumes: {self.negative_volumes}",
            f"Detected Time Gaps: {self.timestamp_gaps}",
        ]
        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  - [ERROR] {err}")
        if self.warnings:
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  - [WARNING] {warn}")
        return "\n".join(lines)


class DataValidator:
    """Validates raw and processed cryptocurrency market data against financial invariants."""

    REQUIRED_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]

    def __init__(self, expected_interval: str = "1h") -> None:
        self.expected_interval = expected_interval

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Executes full validation suite on the input dataframe.

        Args:
            df: Input DataFrame containing OHLCV records.

        Returns:
            ValidationReport with detailed counts and diagnostic messages.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if df.empty:
            return ValidationReport(
                is_valid=False,
                total_rows=0,
                missing_values={},
                duplicate_timestamps=0,
                non_monotonic_timestamps=0,
                invalid_ohlc_bars=0,
                non_positive_prices=0,
                negative_volumes=0,
                timestamp_gaps=0,
                errors=["Dataset is completely empty."],
            )

        # Check required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationReport(
                is_valid=False,
                total_rows=len(df),
                missing_values={},
                duplicate_timestamps=0,
                non_monotonic_timestamps=0,
                invalid_ohlc_bars=0,
                non_positive_prices=0,
                negative_volumes=0,
                timestamp_gaps=0,
                errors=errors,
            )

        # Ensure timestamps are parsed
        if not pd.api.types.is_datetime64_any_dtype(df["open_time"]):
            try:
                open_time = pd.to_datetime(df["open_time"], utc=True)
            except Exception as e:
                errors.append(f"Could not parse 'open_time' as datetime: {e}")
                open_time = df["open_time"]
        else:
            open_time = df["open_time"]

        total_rows = len(df)
        missing_values = df[self.REQUIRED_COLUMNS].isna().sum().to_dict()
        for col, count in missing_values.items():
            if count > 0:
                warnings.append(f"Column '{col}' contains {count} missing (NaN) values.")

        # Check duplicate timestamps
        duplicates = int(open_time.duplicated().sum())
        if duplicates > 0:
            errors.append(f"Detected {duplicates} duplicate timestamps.")

        # Check monotonic ordering
        is_monotonic = open_time.is_monotonic_increasing
        non_monotonic = 0 if is_monotonic else 1
        if not is_monotonic:
            errors.append("Timestamps are not strictly monotonically increasing.")

        # Price sanity: strictly positive prices
        price_cols = ["open", "high", "low", "close"]
        non_positive_mask = (df[price_cols] <= 0).any(axis=1)
        non_positive_count = int(non_positive_mask.sum())
        if non_positive_count > 0:
            errors.append(f"Detected {non_positive_count} rows with non-positive price values (<= 0).")

        # Volume sanity: non-negative volume
        negative_vol_mask = df["volume"] < 0
        negative_vol_count = int(negative_vol_mask.sum())
        if negative_vol_count > 0:
            errors.append(f"Detected {negative_vol_count} rows with negative volume (< 0).")

        # OHLC logical consistency:
        # High must be >= Open, Close, Low.
        # Low must be <= Open, Close, High.
        invalid_ohlc_mask = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )
        invalid_ohlc_count = int(invalid_ohlc_mask.sum())
        if invalid_ohlc_count > 0:
            errors.append(
                f"Detected {invalid_ohlc_count} rows violating candlestick boundaries (e.g. High < Low or High < Close)."
            )

        # Gap detection
        time_gaps = 0
        if is_monotonic and duplicates == 0 and len(open_time) > 1:
            diffs = open_time.diff().dropna()
            median_delta = diffs.median()
            # If any step is > 1.5x median delta, it is a gap
            gap_mask = diffs > (1.5 * median_delta)
            time_gaps = int(gap_mask.sum())
            if time_gaps > 0:
                warnings.append(f"Detected {time_gaps} temporal gaps exceeding expected bar interval {median_delta}.")

        is_valid = len(errors) == 0

        return ValidationReport(
            is_valid=is_valid,
            total_rows=total_rows,
            missing_values=missing_values,
            duplicate_timestamps=duplicates,
            non_monotonic_timestamps=non_monotonic,
            invalid_ohlc_bars=invalid_ohlc_count,
            non_positive_prices=non_positive_count,
            negative_volumes=negative_vol_count,
            timestamp_gaps=time_gaps,
            errors=errors,
            warnings=warnings,
        )
