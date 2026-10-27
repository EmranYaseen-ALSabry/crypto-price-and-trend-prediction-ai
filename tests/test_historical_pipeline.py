"""Unit tests verifying Phase 3.2 data pipeline outputs:
- Pilot dataset integrity
- Synchronized multi-asset panel integrity
- Provenance metadata and SHA-256 validation
"""

from pathlib import Path
import json
import pandas as pd
import pytest

from crypto_prediction.data.collector import compute_file_checksum
from crypto_prediction.data.validator import DataValidator


class TestHistoricalPipelineOutputs:
    """Verifies that ingested datasets satisfy the Phase 3.2 quality specification."""

    BASE_DIR = Path("data")

    def test_pilot_dataset_integrity(self):
        pilot_dir = self.BASE_DIR / "processed" / "pilot"
        validator = DataValidator(expected_interval="1h")
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

        for sym in symbols:
            csv_path = pilot_dir / f"{sym}_1h_pilot.csv"
            meta_path = pilot_dir / f"{sym}_1h_pilot.csv.meta.json"

            assert csv_path.exists(), f"Pilot CSV missing for {sym}"
            assert meta_path.exists(), f"Pilot metadata missing for {sym}"

            df = pd.read_csv(csv_path)
            assert len(df) == 744, f"Expected 744 bars in pilot dataset, got {len(df)}"

            # Invariant validation
            report = validator.validate(df)
            assert report.is_valid is True
            assert report.duplicate_timestamps == 0
            assert report.invalid_ohlc_bars == 0
            assert report.non_positive_prices == 0
            assert report.negative_volumes == 0

            # Provenance checksum check
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            assert meta["sha256"] == compute_file_checksum(csv_path)
            assert meta["market_type"] == "spot"

    def test_panel_synchronization_and_alignment(self):
        panel_dir = self.BASE_DIR / "processed" / "panel_5000h"
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        dfs = {}

        for sym in symbols:
            csv_path = panel_dir / f"{sym}_1h.csv"
            meta_path = panel_dir / f"{sym}_1h.csv.meta.json"

            assert csv_path.exists(), f"Panel CSV missing for {sym}"
            assert meta_path.exists(), f"Panel metadata missing for {sym}"

            df = pd.read_csv(csv_path)
            assert len(df) >= 5000, f"Expected >= 5000 bars in panel, got {len(df)}"
            dfs[sym] = df

            # Validate checksum
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            assert meta["sha256"] == compute_file_checksum(csv_path)
            assert meta["market_type"] == "spot"

        # Assert identical length across all symbols
        btc_len = len(dfs["BTCUSDT"])
        assert len(dfs["ETHUSDT"]) == btc_len
        assert len(dfs["SOLUSDT"]) == btc_len

        # Assert exact timestamp alignment
        btc_ts = dfs["BTCUSDT"]["open_time"]
        assert (btc_ts == dfs["ETHUSDT"]["open_time"]).all()
        assert (btc_ts == dfs["SOLUSDT"]["open_time"]).all()

        # Assert chronological monotonicity
        assert pd.to_datetime(btc_ts, utc=True).is_monotonic_increasing
