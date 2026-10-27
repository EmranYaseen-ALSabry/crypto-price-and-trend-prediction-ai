"""Script to execute Phase 3.2 data ingestion:
1. Pilot Ingestion (30 days: 2024-01-01 to 2024-01-31)
2. Multi-Year Historical Ingestion (BTCUSDT, ETHUSDT, SOLUSDT)
3. Strict Data Validation and Anomaly Auditing
4. Atomic Storage & Provenance Metadata Generation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd

from crypto_prediction.data.collector import DataCollector, compute_file_checksum
from crypto_prediction.data.validator import DataValidator
from crypto_prediction.utils.logger import get_logger

logger = get_logger("historical_pipeline")


def run_pilot_ingestion(collector: DataCollector, symbols: list[str]) -> dict:
    """Collects and validates a 30-day pilot dataset (2024-01-01 to 2024-01-31)."""
    pilot_dir = collector.base_dir / "processed" / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    validator = DataValidator(expected_interval="1h")
    results = {}

    start_date = "2024-01-01 00:00:00"
    end_date = "2024-01-31 23:00:00"

    print("\n" + "="*70)
    print("STEP 1: PILOT DATASET COLLECTION & VALIDATION (30 Days: Jan 2024)")
    print("="*70)

    for symbol in symbols:
        print(f"\n[*] Fetching pilot data for {symbol} (1h)...")
        # Fetch directly without saving to base raw dir to avoid overwriting production raw stores
        pilot_collector = DataCollector(base_dir=collector.base_dir, rate_limit_delay=0.25)
        df = pilot_collector.fetch_historical_range(
            symbol=symbol,
            interval="1h",
            start_time=start_date,
            end_time=end_date,
            market_type="spot",
            resumable=False,
            save_to_raw=False,
            exclude_unclosed_bar=False,
        )

        val_report = validator.validate(df)
        print(f"Validation Result for {symbol}: {'PASSED' if val_report.is_valid else 'FAILED'}")
        print(f"  Rows: {val_report.total_rows} | Duplicate TS: {val_report.duplicate_timestamps} | Gaps: {val_report.timestamp_gaps}")
        print(f"  OHLCV Violations: {val_report.invalid_ohlc_bars} | Anomalies: {val_report.extreme_anomalies}")

        # Save validated pilot data
        target_path = pilot_dir / f"{symbol}_1h_pilot.csv"
        pilot_collector.save_to_csv(df, target_path)
        meta = {
            "dataset_stage": "pilot",
            "symbol": symbol,
            "interval": "1h",
            "market_type": "spot",
            "total_bars": len(df),
            "first_bar_utc": str(df["open_time"].min()),
            "last_bar_utc": str(df["open_time"].max()),
            "is_valid": val_report.is_valid,
            "duplicate_timestamps": val_report.duplicate_timestamps,
            "timestamp_gaps": val_report.timestamp_gaps,
            "invalid_ohlc_bars": val_report.invalid_ohlc_bars,
            "extreme_anomalies": val_report.extreme_anomalies,
            "retrieved_at_utc": str(pd.Timestamp.now(tz="UTC")),
        }
        pilot_collector.save_metadata(target_path, meta)
        results[symbol] = {
            "path": str(target_path),
            "bars": len(df),
            "valid": val_report.is_valid,
            "sha256": compute_file_checksum(target_path),
        }

    return results


def run_historical_ingestion(
    collector: DataCollector,
    symbols: list[str],
    start_date: str = "2023-01-01 00:00:00",
    end_date: str | None = None,
) -> tuple[dict, dict]:
    """Collects multi-year historical data for symbols, validates, and stores in raw and processed locations."""
    hist_proc_dir = collector.base_dir / "processed" / "historical"
    panel_dir = collector.base_dir / "processed" / "panel_5000h"
    hist_proc_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    validator = DataValidator(expected_interval="1h")
    inventory = {}
    quality_reports = {}
    loaded_dfs = {}

    print("\n" + "="*70)
    print(f"STEP 2: MULTI-YEAR HISTORICAL INGESTION (Start: {start_date})")
    print("="*70)

    for symbol in symbols:
        print(f"\n[*] Collecting multi-year historical bars for {symbol} (1h, spot)...")
        df_raw = collector.fetch_historical_range(
            symbol=symbol,
            interval="1h",
            start_time=start_date,
            end_time=end_date,
            market_type="spot",
            resumable=True,
            save_to_raw=True,
            exclude_unclosed_bar=True,
        )

        raw_path = collector.raw_dir / f"{symbol}_1h.csv"
        raw_sha = compute_file_checksum(raw_path)

        # Validate raw data
        val_report = validator.validate(df_raw)
        quality_reports[symbol] = val_report

        print(f"Validation Report for {symbol}:")
        print(f"  Total Rows: {val_report.total_rows}")
        print(f"  Range: {df_raw['open_time'].min()} to {df_raw['open_time'].max()}")
        print(f"  Is Valid: {val_report.is_valid}")
        print(f"  Duplicate Timestamps: {val_report.duplicate_timestamps}")
        print(f"  Candlestick Invariant Violations: {val_report.invalid_ohlc_bars}")
        print(f"  Non-Positive Prices: {val_report.non_positive_prices}")
        print(f"  Temporal Gaps (>1.5h): {val_report.timestamp_gaps}")
        print(f"  Extreme Anomalies (|return|>20% or spread>40%): {val_report.extreme_anomalies}")

        # Store validated copy in processed/historical/
        proc_path = hist_proc_dir / f"{symbol}_1h.csv"
        collector.save_to_csv(df_raw, proc_path)
        proc_sha = compute_file_checksum(proc_path)

        proc_meta = {
            "dataset_stage": "processed_historical",
            "symbol": symbol,
            "interval": "1h",
            "market_type": "spot",
            "total_bars": len(df_raw),
            "first_bar_utc": str(df_raw["open_time"].min()),
            "last_bar_utc": str(df_raw["open_time"].max()),
            "validation_passed": val_report.is_valid,
            "duplicate_timestamps": val_report.duplicate_timestamps,
            "temporal_gaps": val_report.timestamp_gaps,
            "gap_details": val_report.gap_details,
            "invalid_ohlc_bars": val_report.invalid_ohlc_bars,
            "extreme_anomalies": val_report.extreme_anomalies,
            "anomaly_details": val_report.anomaly_details,
            "sha256": proc_sha,
            "source_raw_sha256": raw_sha,
            "retrieved_at_utc": str(pd.Timestamp.now(tz="UTC")),
        }
        collector.save_metadata(proc_path, proc_meta)

        inventory[symbol] = {
            "raw_path": str(raw_path),
            "raw_sha256": raw_sha,
            "raw_size_bytes": raw_path.stat().st_size,
            "proc_path": str(proc_path),
            "proc_sha256": proc_sha,
            "proc_size_bytes": proc_path.stat().st_size,
            "total_bars": len(df_raw),
            "first_bar_utc": str(df_raw["open_time"].min()),
            "last_bar_utc": str(df_raw["open_time"].max()),
            "validation_passed": val_report.is_valid,
        }
        loaded_dfs[symbol] = df_raw

    # Step 3: Synchronize Common Multi-Year Overlapping Panel (minimum 5,000 continuous bars)
    print("\n" + "="*70)
    print("STEP 3: SYNCHRONIZING MULTI-ASSET OVERLAPPING PANEL")
    print("="*70)

    # Find common timestamp index intersection
    common_indices = None
    for sym, df in loaded_dfs.items():
        ts_set = set(df["open_time"])
        if common_indices is None:
            common_indices = ts_set
        else:
            common_indices = common_indices.intersection(ts_set)

    common_sorted = sorted(list(common_indices))
    print(f"Total Common Overlapping Hourly Timestamps across {symbols}: {len(common_sorted)}")
    if common_sorted:
        print(f"Common Range: {common_sorted[0]} to {common_sorted[-1]}")

    # Extract 5,000+ continuous bars or full common range for panel_5000h
    for symbol, df in loaded_dfs.items():
        synced_df = df[df["open_time"].isin(common_sorted)].sort_values(by="open_time").reset_index(drop=True)
        panel_path = panel_dir / f"{symbol}_1h.csv"
        collector.save_to_csv(synced_df, panel_path)
        panel_sha = compute_file_checksum(panel_path)
        panel_meta = {
            "dataset_stage": "panel_synchronized",
            "symbol": symbol,
            "interval": "1h",
            "market_type": "spot",
            "total_bars": len(synced_df),
            "first_bar_utc": str(synced_df["open_time"].min()),
            "last_bar_utc": str(synced_df["open_time"].max()),
            "sha256": panel_sha,
            "common_symbols": symbols,
            "retrieved_at_utc": str(pd.Timestamp.now(tz="UTC")),
        }
        collector.save_metadata(panel_path, panel_meta)
        print(f"Synchronized panel saved for {symbol}: {len(synced_df)} bars -> {panel_path}")

    return inventory, quality_reports


def main():
    parser = argparse.ArgumentParser(description="Multi-Year Historical Data Pipeline Ingestion")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--start-date", default="2023-01-01 00:00:00", help="Start date UTC")
    parser.add_argument("--skip-pilot", action="store_true", help="Skip pilot dataset ingestion")
    args = parser.parse_args()

    collector = DataCollector(base_dir="data")

    if not args.skip_pilot:
        pilot_results = run_pilot_ingestion(collector, args.symbols)
        print("\nPilot Ingestion Summary:")
        for sym, res in pilot_results.items():
            print(f"  {sym}: {res['bars']} bars, Valid: {res['valid']}, SHA: {res['sha256'][:16]}...")

    inventory, quality_reports = run_historical_ingestion(collector, args.symbols, start_date=args.start_date)

    print("\n" + "="*70)
    print("INGESTION & VALIDATION COMPLETE")
    print("="*70)
    for sym, inv in inventory.items():
        print(f"{sym}: {inv['total_bars']} bars | Raw: {inv['raw_sha256'][:12]} | Valid: {inv['validation_passed']}")


if __name__ == "__main__":
    main()
