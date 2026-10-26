"""Unit tests for data validation and cleaning components."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crypto_prediction.data.cleaner import DataCleaner
from crypto_prediction.data.validator import DataValidator


class TestDataValidation:
    """Tests validation of clean and defective market datasets."""

    def test_clean_data_passes_validation(self, sample_ohlcv_df: pd.DataFrame) -> None:
        validator = DataValidator(expected_interval="1h")
        report = validator.validate(sample_ohlcv_df)

        assert report.is_valid is True
        assert report.duplicate_timestamps == 0
        assert report.non_monotonic_timestamps == 0
        assert report.invalid_ohlc_bars == 0
        assert report.non_positive_prices == 0
        assert report.negative_volumes == 0
        assert len(report.errors) == 0

    def test_corrupted_data_detected(self, corrupted_ohlcv_df: pd.DataFrame) -> None:
        validator = DataValidator(expected_interval="1h")
        report = validator.validate(corrupted_ohlcv_df)

        assert report.is_valid is False
        assert report.non_positive_prices >= 1
        assert report.negative_volumes >= 1
        assert report.invalid_ohlc_bars >= 1
        assert report.duplicate_timestamps >= 1
        assert len(report.errors) > 0

    def test_empty_dataframe_fails(self) -> None:
        validator = DataValidator()
        report = validator.validate(pd.DataFrame())

        assert report.is_valid is False
        assert "empty" in report.errors[0].lower()

    def test_missing_required_columns(self, sample_ohlcv_df: pd.DataFrame) -> None:
        validator = DataValidator()
        incomplete_df = sample_ohlcv_df.drop(columns=["volume", "high"])
        report = validator.validate(incomplete_df)

        assert report.is_valid is False
        assert any("Missing required columns" in err for err in report.errors)


class TestDataCleaner:
    """Tests sorting, deduplication, and forward-filling rules."""

    def test_cleaner_fixes_duplicates_and_sorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        cleaner = DataCleaner(max_consecutive_fill=3)

        # Inject duplicate and out-of-order row
        dirty = sample_ohlcv_df.copy()
        extra_row = dirty.iloc[10:11].copy()
        dirty = pd.concat([dirty, extra_row], ignore_index=True)

        cleaned = cleaner.clean(dirty)

        assert isinstance(cleaned.index, pd.DatetimeIndex)
        assert cleaned.index.is_monotonic_increasing
        assert cleaned.index.duplicated().sum() == 0
        assert len(cleaned) == len(sample_ohlcv_df)

    def test_cleaner_handles_candlestick_invariants(self, sample_ohlcv_df: pd.DataFrame) -> None:
        cleaner = DataCleaner()
        df = sample_ohlcv_df.copy()
        # Set high lower than close
        df.loc[5, "high"] = df.loc[5, "close"] - 10.0
        # Set low higher than open
        df.loc[6, "low"] = df.loc[6, "open"] + 10.0

        cleaned = cleaner.clean(df)

        assert (cleaned["high"] >= cleaned["open"]).all()
        assert (cleaned["high"] >= cleaned["close"]).all()
        assert (cleaned["low"] <= cleaned["open"]).all()
        assert (cleaned["low"] <= cleaned["close"]).all()

    def test_cleaner_drops_non_positive_prices(self, sample_ohlcv_df: pd.DataFrame) -> None:
        cleaner = DataCleaner()
        df = sample_ohlcv_df.copy()
        df.loc[15, "close"] = -100.0

        cleaned = cleaner.clean(df)
        assert (cleaned["close"] > 0).all()
        assert len(cleaned) == len(sample_ohlcv_df) - 1


class TestDataCollector:
    """Tests data acquisition, fallback priority order, and synthetic isolation."""

    def test_collector_cached_raw_priority_over_sample(self, tmp_path: Path, sample_ohlcv_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        from crypto_prediction.data.collector import DataCollector

        collector = DataCollector(base_dir=tmp_path)
        # Mock API failure to test offline fallback hierarchy
        monkeypatch.setattr(collector, "fetch_binance_klines", lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError("Simulated API failure")))

        # Create raw file with 10 rows
        raw_df = sample_ohlcv_df.iloc[:10].copy()
        collector.save_to_csv(raw_df, collector.raw_dir / "BTCUSDT_1h.csv")

        # Create sample file with 20 rows
        sample_df = sample_ohlcv_df.iloc[:20].copy()
        collector.save_to_csv(sample_df, collector.sample_dir / "BTCUSDT_1h.csv")

        # Live fetch fails -> must pick cached raw file, NOT sample file
        df = collector.fetch_historical_bars(symbol="BTCUSDT", interval="1h", total_bars=10, fallback_to_sample=True)
        assert df.attrs.get("data_source") == "cached_raw_csv"
        assert len(df) == 10

    def test_collector_synthetic_isolation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from crypto_prediction.data.collector import DataCollector

        collector = DataCollector(base_dir=tmp_path)
        # Mock API failure to test synthetic generation fallback
        monkeypatch.setattr(collector, "fetch_binance_klines", lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError("Simulated API failure")))

        # No raw, no sample -> triggers synthetic
        df = collector.fetch_historical_bars(symbol="SOLUSDT", interval="1h", total_bars=50, fallback_to_sample=True)

        assert df.attrs.get("data_source") == "synthetic"
        assert df.attrs.get("is_synthetic") is True

        # Assert synthetic file is saved ONLY in synthetic_dir, NOT sample_dir
        assert (collector.synthetic_dir / "SOLUSDT_1h_synthetic.csv").exists()
        assert not (collector.sample_dir / "SOLUSDT_1h.csv").exists()
