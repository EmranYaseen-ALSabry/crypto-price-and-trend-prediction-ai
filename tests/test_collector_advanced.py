"""Unit and integration tests for advanced data collector capabilities:
- Forward date range pagination
- Incremental updates and resumable collection
- Atomic file writes and checksums
- Rate limit handling and backoff
- Spot market isolation
- Metadata sidecar generation
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import requests

from crypto_prediction.data.collector import DataCollector, compute_file_checksum
from crypto_prediction.data.validator import DataValidator


@pytest.fixture
def mock_kline_generator():
    """Generates mock Binance REST kline batches."""
    def _gen_batch(start_ts_ms: int, count: int, interval_ms: int = 3600000):
        batch = []
        for i in range(count):
            open_time = start_ts_ms + (i * interval_ms)
            close_time = open_time + interval_ms - 1
            batch.append([
                open_time,
                "50000.00",
                "50500.00",
                "49800.00",
                "50200.00",
                "150.500",
                close_time,
                "7550000.00",
                1000,
                "75.250",
                "3775000.00",
                "0",
            ])
        return batch
    return _gen_batch


class TestDataCollectorAdvanced:
    """Verifies pagination, atomic writes, resumability, and error handling."""

    def test_atomic_save_csv(self, tmp_path: Path):
        collector = DataCollector(base_dir=tmp_path)
        df = pd.DataFrame({
            "open_time": pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"),
            "open": [10.0] * 5,
            "high": [12.0] * 5,
            "low": [9.0] * 5,
            "close": [11.0] * 5,
            "volume": [100.0] * 5,
        })
        target_path = tmp_path / "atomic_test.csv"
        saved_path = collector.save_to_csv(df, target_path)

        assert saved_path.exists()
        loaded = pd.read_csv(saved_path)
        assert len(loaded) == 5

        # Verify no temporary files left in directory
        temp_files = list(tmp_path.glob(".*.tmp_*"))
        assert len(temp_files) == 0

    def test_compute_file_checksum_and_metadata(self, tmp_path: Path):
        collector = DataCollector(base_dir=tmp_path)
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\n1,2\n3,4\n", encoding="utf-8")

        sha = compute_file_checksum(test_file)
        assert isinstance(sha, str)
        assert len(sha) == 64

        meta_path = collector.save_metadata(test_file, {"symbol": "BTCUSDT", "bars": 2})
        assert meta_path.exists()

        loaded_meta = collector.load_metadata(test_file)
        assert loaded_meta["sha256"] == sha
        assert loaded_meta["symbol"] == "BTCUSDT"
        assert loaded_meta["file_size_bytes"] == test_file.stat().st_size

    def test_spot_market_type_enforcement(self, tmp_path: Path):
        collector = DataCollector(base_dir=tmp_path)
        with pytest.raises(ValueError, match="Only 'spot' is supported"):
            collector.fetch_historical_range(symbol="BTCUSDT", market_type="futures")

        with pytest.raises(ValueError, match="Only 'spot' is supported"):
            collector.fetch_historical_bars(symbol="BTCUSDT", market_type="perpetual")

    def test_forward_pagination_multiple_pages(self, tmp_path: Path, mock_kline_generator):
        collector = DataCollector(base_dir=tmp_path, rate_limit_delay=0.0)

        start_ms = 1704067200000  # 2024-01-01 00:00:00 UTC
        # Generate two pages: 1000 candles and 500 candles
        page1 = mock_kline_generator(start_ms, 1000)
        page2_start = start_ms + (1000 * 3600000)
        page2 = mock_kline_generator(page2_start, 500)

        with patch("requests.get") as mock_get:
            mock_resp1 = MagicMock()
            mock_resp1.status_code = 200
            mock_resp1.json.return_value = page1

            mock_resp2 = MagicMock()
            mock_resp2.status_code = 200
            mock_resp2.json.return_value = page2

            mock_get.side_effect = [mock_resp1, mock_resp2]

            df = collector.fetch_historical_range(
                symbol="BTCUSDT",
                interval="1h",
                start_time=start_ms,
                end_time=page2_start + (500 * 3600000),
                exclude_unclosed_bar=False,
            )

            assert len(df) == 1500
            assert mock_get.call_count == 2
            # Verify chronological ordering
            assert df["open_time"].is_monotonic_increasing
            assert (tmp_path / "raw" / "BTCUSDT_1h.csv").exists()
            assert (tmp_path / "raw" / "BTCUSDT_1h.csv.meta.json").exists()

    def test_resumable_incremental_update(self, tmp_path: Path, mock_kline_generator):
        collector = DataCollector(base_dir=tmp_path, rate_limit_delay=0.0)

        # Pre-populate raw CSV with 100 bars
        start_ms = 1704067200000  # 2024-01-01 00:00:00 UTC
        initial_bars = mock_kline_generator(start_ms, 100)
        initial_df = pd.DataFrame([{
            "open_time": pd.to_datetime(k[0], unit="ms", utc=True),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        } for k in initial_bars])

        raw_file = tmp_path / "raw" / "BTCUSDT_1h.csv"
        collector.save_to_csv(initial_df, raw_file)

        # New bars starting at bar 101 (20 bars)
        new_start_ms = start_ms + (100 * 3600000)
        new_bars = mock_kline_generator(new_start_ms, 20)

        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = new_bars
            mock_get.return_value = mock_resp

            df = collector.fetch_historical_range(
                symbol="BTCUSDT",
                interval="1h",
                start_time=start_ms,
                end_time=new_start_ms + (20 * 3600000),
                resumable=True,
                exclude_unclosed_bar=False,
            )

            # Total should be 100 + 20 = 120 bars
            assert len(df) == 120
            # Ensure only 1 call was made since it resumed from bar 101
            assert mock_get.call_count == 1
            call_params = mock_get.call_args[1]["params"]
            assert call_params["startTime"] > start_ms

    def test_rate_limit_and_server_error_backoff(self, tmp_path: Path, mock_kline_generator):
        collector = DataCollector(base_dir=tmp_path, rate_limit_delay=0.0, max_retries=3)

        success_batch = mock_kline_generator(1704067200000, 10)

        with patch("requests.get") as mock_get, patch("time.sleep") as mock_sleep:
            # 1st attempt: 429 rate limit
            resp_429 = MagicMock()
            resp_429.status_code = 429
            resp_429.headers = {"Retry-After": "1"}

            # 2nd attempt: 502 bad gateway
            resp_502 = MagicMock()
            resp_502.status_code = 502

            # 3rd attempt: 200 success
            resp_200 = MagicMock()
            resp_200.status_code = 200
            resp_200.json.return_value = success_batch

            mock_get.side_effect = [resp_429, resp_502, resp_200]

            df = collector.fetch_binance_klines(symbol="BTCUSDT", limit=10)
            assert len(df) == 10
            assert mock_get.call_count == 3
            assert mock_sleep.call_count >= 2

    def test_validator_gap_and_anomaly_detection(self):
        validator = DataValidator(expected_interval="1h")

        # Create time series with intentional gap and extreme price spike
        timestamps = list(pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC"))
        # Inject 10-hour gap between item 4 and 5
        timestamps[5] = timestamps[4] + pd.Timedelta(hours=10)
        for i in range(6, 10):
            timestamps[i] = timestamps[i-1] + pd.Timedelta(hours=1)

        df = pd.DataFrame({
            "open_time": timestamps,
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [98.0] * 10,
            "close": [102.0] * 10,
            "volume": [1000.0] * 10,
        })
        # Inject extreme spike at index 7: +50% move
        df.loc[7, "close"] = 150.0
        df.loc[7, "high"] = 155.0

        report = validator.validate(df)
        assert report.timestamp_gaps >= 1
        assert len(report.gap_details) >= 1
        assert report.gap_details[0]["duration_hours"] >= 9.0
        assert report.extreme_anomalies >= 1
        assert len(report.anomaly_details) >= 1
