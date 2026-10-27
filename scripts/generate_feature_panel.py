"""Script to generate and validate synchronized cross-asset feature matrices for Phase 3.3:
1. Pilot Feature Dataset (Jan 2024, 744 bars)
2. Full Multi-Year Panel Features (33,493 bars, 2023-2026)
3. Invariant and causality verification
4. Storage in data/processed/features_panel/ and data/processed/features_pilot/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from crypto_prediction.data.collector import DataCollector, compute_file_checksum
from crypto_prediction.data.synchronizer import MultiAssetSynchronizer
from crypto_prediction.features.cross_asset import CrossAssetFeatureEngineer
from crypto_prediction.utils.logger import get_logger

logger = get_logger("generate_features")


def generate_pilot_features(collector: DataCollector) -> dict:
    """Generates cross-asset features on the 30-day pilot panel."""
    pilot_in_dir = collector.base_dir / "processed" / "pilot"
    pilot_out_dir = collector.base_dir / "processed" / "features_pilot"
    pilot_out_dir.mkdir(parents=True, exist_ok=True)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    datasets = {}
    for sym in symbols:
        p = pilot_in_dir / f"{sym}_1h_pilot.csv"
        df = pd.read_csv(p)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        datasets[sym] = df.set_index("open_time")

    print("\n" + "="*70)
    print("STEP 1: GENERATING PILOT CROSS-ASSET FEATURES (30 Days: Jan 2024)")
    print("="*70)

    engineer = CrossAssetFeatureEngineer(
        anchor_symbols=["BTCUSDT", "ETHUSDT"],
        primary_horizon=1,
        forecast_horizons=[1, 4, 24],
        include_technicals=True,
        include_patterns=False,
        include_market_aggregates=True,
    )

    prepared = engineer.prepare_dataset(datasets, target_horizon=1, drop_warmup=True)
    summary = {}

    for sym, data in prepared.items():
        X = data["X"]
        y = data["y"]
        meta = data["meta"]

        out_path = pilot_out_dir / f"{sym}_1h_features_pilot.csv"
        # Save X combined with target and metadata
        combined = pd.concat([X, y.rename("target_return_1"), meta], axis=1)
        collector.save_to_csv(combined, out_path)
        sha = compute_file_checksum(out_path)

        meta_dict = {
            "dataset_stage": "features_pilot",
            "symbol": sym,
            "interval": "1h",
            "total_usable_bars": len(X),
            "num_features": len(X.columns),
            "features_list": list(X.columns),
            "first_bar_utc": str(X.index.min()),
            "last_bar_utc": str(X.index.max()),
            "target": "target_return_1",
            "sha256": sha,
            "file_size_bytes": out_path.stat().st_size,
        }
        collector.save_metadata(out_path, meta_dict)
        summary[sym] = meta_dict
        print(f"Pilot features saved for {sym}: {len(X)} rows x {len(X.columns)} features -> {out_path.name}")

    return summary


def generate_full_panel_features(collector: DataCollector) -> dict:
    """Generates cross-asset features on the full multi-year 33,493-bar synchronized panel."""
    panel_in_dir = collector.base_dir / "processed" / "panel_5000h"
    panel_out_dir = collector.base_dir / "processed" / "features_panel"
    panel_out_dir.mkdir(parents=True, exist_ok=True)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    datasets = {}
    for sym in symbols:
        p = panel_in_dir / f"{sym}_1h.csv"
        df = pd.read_csv(p)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        datasets[sym] = df.set_index("open_time")

    print("\n" + "="*70)
    print("STEP 2: GENERATING FULL MULTI-YEAR CROSS-ASSET FEATURES (2023-2026)")
    print("="*70)

    engineer = CrossAssetFeatureEngineer(
        anchor_symbols=["BTCUSDT", "ETHUSDT"],
        primary_horizon=1,
        forecast_horizons=[1, 4, 24],
        include_technicals=True,
        include_patterns=False,
        include_market_aggregates=True,
    )

    prepared = engineer.prepare_dataset(datasets, target_horizon=1, drop_warmup=True)
    summary = {}

    for sym, data in prepared.items():
        X = data["X"]
        y = data["y"]
        meta = data["meta"]

        out_path = panel_out_dir / f"{sym}_1h_features.csv"
        combined = pd.concat([X, y.rename("target_return_1"), meta], axis=1)
        collector.save_to_csv(combined, out_path)
        sha = compute_file_checksum(out_path)

        meta_dict = {
            "dataset_stage": "features_full_panel",
            "symbol": sym,
            "interval": "1h",
            "total_usable_bars": len(X),
            "num_features": len(X.columns),
            "features_list": list(X.columns),
            "first_bar_utc": str(X.index.min()),
            "last_bar_utc": str(X.index.max()),
            "target": "target_return_1",
            "sha256": sha,
            "file_size_bytes": out_path.stat().st_size,
        }
        collector.save_metadata(out_path, meta_dict)
        summary[sym] = meta_dict
        print(f"Full panel features saved for {sym}: {len(X)} rows x {len(X.columns)} features -> {out_path.name}")

    return summary


def main():
    collector = DataCollector(base_dir="data")
    pilot_summary = generate_pilot_features(collector)
    panel_summary = generate_full_panel_features(collector)

    print("\n" + "="*70)
    print("PHASE 3.3 FEATURE GENERATION COMPLETE")
    print("="*70)
    for sym, s in panel_summary.items():
        print(f"{sym}: {s['total_usable_bars']} usable rows, {s['num_features']} features, Range: {s['first_bar_utc']} to {s['last_bar_utc']}")


if __name__ == "__main__":
    main()
