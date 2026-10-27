"""Unit tests for MultiAssetSynchronizer:
- Timestamp semantics (open, close, available)
- Panel key uniqueness (open_time, symbol)
- Intersection vs Union alignment
- Non-fabrication policy and staleness tracking
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crypto_prediction.data.synchronizer import MultiAssetSynchronizer, SynchronizationReport


@pytest.fixture
def sample_multi_asset_dfs() -> dict[str, pd.DataFrame]:
    """Generates synthetic 3-asset dataset with known overlapping and non-overlapping periods."""
    ts_btc = pd.date_range("2024-01-01 00:00:00", periods=50, freq="1h", tz="UTC")
    ts_eth = pd.date_range("2024-01-01 00:00:00", periods=50, freq="1h", tz="UTC")
    # SOL starts 10 hours later and has a 3-hour gap
    ts_sol_part1 = pd.date_range("2024-01-01 10:00:00", periods=15, freq="1h", tz="UTC")
    ts_sol_part2 = pd.date_range("2024-01-02 04:00:00", periods=22, freq="1h", tz="UTC")
    ts_sol = ts_sol_part1.union(ts_sol_part2)

    def _make_df(ts, base_price):
        n = len(ts)
        return pd.DataFrame({
            "open_time": ts,
            "open": [base_price] * n,
            "high": [base_price * 1.01] * n,
            "low": [base_price * 0.99] * n,
            "close": [base_price * 1.005] * n,
            "volume": [1000.0] * n,
        })

    return {
        "BTCUSDT": _make_df(ts_btc, 40000.0),
        "ETHUSDT": _make_df(ts_eth, 2500.0),
        "SOLUSDT": _make_df(ts_sol, 100.0),
    }


class TestMultiAssetSynchronizer:
    """Verifies temporal semantics, non-fabrication, and alignment modes."""

    def test_timestamp_semantics_addition(self, sample_multi_asset_dfs):
        sync = MultiAssetSynchronizer(interval="1h")
        btc_df = sample_multi_asset_dfs["BTCUSDT"]
        enriched = sync.add_timestamp_semantics(btc_df, "BTCUSDT")

        assert "open_time" in enriched.columns
        assert "close_time" in enriched.columns
        assert "available_time" in enriched.columns
        assert "symbol" in enriched.columns

        first_row = enriched.iloc[0]
        # 12:00 -> close 12:59:59.999 -> available 13:00:00
        assert first_row["close_time"] == first_row["open_time"] + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
        assert first_row["available_time"] == first_row["open_time"] + pd.Timedelta(hours=1)
        assert first_row["symbol"] == "BTCUSDT"

    def test_intersection_synchronization(self, sample_multi_asset_dfs):
        sync = MultiAssetSynchronizer(interval="1h")
        aligned, panel, report = sync.synchronize(sample_multi_asset_dfs, mode="intersection")

        assert report.mode == "intersection"
        assert report.is_aligned is True

        # Intersection length should equal SOL's row count (since SOL is a subset of BTC and ETH)
        sol_len = len(sample_multi_asset_dfs["SOLUSDT"])
        assert len(aligned["BTCUSDT"]) == sol_len
        assert len(aligned["ETHUSDT"]) == sol_len
        assert len(aligned["SOLUSDT"]) == sol_len

        # Identical indices
        assert aligned["BTCUSDT"].index.equals(aligned["ETHUSDT"].index)
        assert aligned["BTCUSDT"].index.equals(aligned["SOLUSDT"].index)

        # Panel MultiIndex uniqueness
        assert panel.index.names == ["open_time", "symbol"]
        assert not panel.index.duplicated().any()

        # In intersection mode, all rows must be available
        assert (aligned["BTCUSDT"]["is_available"] == True).all()
        assert (aligned["SOLUSDT"]["staleness_bars"] == 0).all()

    def test_union_synchronization_and_non_fabrication(self, sample_multi_asset_dfs):
        sync = MultiAssetSynchronizer(interval="1h")
        aligned, panel, report = sync.synchronize(sample_multi_asset_dfs, mode="union")

        assert report.mode == "union"
        # Total timestamps should equal union of all timestamps (50 hours)
        assert len(aligned["BTCUSDT"]) == 50
        assert len(aligned["ETHUSDT"]) == 50
        assert len(aligned["SOLUSDT"]) == 50

        # SOL was unlisted/missing in the first 10 hours
        sol_df = aligned["SOLUSDT"]
        first_10_avail = sol_df["is_available"].iloc[:10]
        assert (first_10_avail == False).all()

        # Non-fabrication check: prices MUST NOT be forward-filled!
        first_10_close = sol_df["close"].iloc[:10]
        assert first_10_close.isna().all()

        # Staleness counter must increment during missing periods
        assert sol_df["staleness_bars"].iloc[0] == 1
        assert sol_df["staleness_bars"].iloc[9] == 10
        # When bar 10 arrives, staleness resets to 0
        assert sol_df["staleness_bars"].iloc[10] == 0
        assert sol_df["is_available"].iloc[10] == True

    def test_unsupported_interval_raises(self):
        with pytest.raises(ValueError, match="Unsupported interval"):
            MultiAssetSynchronizer(interval="3d")

    def test_empty_datasets_raises(self):
        sync = MultiAssetSynchronizer(interval="1h")
        with pytest.raises(ValueError, match="cannot be empty"):
            sync.synchronize({})
