"""CLI Entry point for Phase 2: Multi-Asset Generalization & Transferability Experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src to sys.path
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from crypto_prediction.pipeline.multi_asset import MultiAssetEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Asset Cryptocurrency Generalization Engine")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="List of crypto assets to evaluate (default: BTCUSDT ETHUSDT SOLUSDT)",
    )
    parser.add_argument("--interval", type=str, default="1h", help="Timeframe interval (default: 1h)")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon (default: 1 bar)")
    parser.add_argument(
        "--data-source",
        type=str,
        default="sample",
        choices=["sample", "raw", "live"],
        help="Data source: 'sample' (verified benchmark panel), 'raw' (cached), or 'live' (Binance)",
    )
    parser.add_argument("--total-bars", type=int, default=1500, help="Total bars per asset to load")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/multi_asset",
        help="Output directory for artifacts and reports",
    )
    args = parser.parse_args()

    engine = MultiAssetEngine(
        symbols=args.symbols,
        interval=args.interval,
        horizon=args.horizon,
        artifacts_dir=args.artifacts_dir,
    )

    results = engine.run_full_research(
        data_source=args.data_source,
        total_bars=args.total_bars,
    )

    print("\n" + "=" * 78)
    print("  MULTI-ASSET GENERALIZATION & TRANSFERABILITY EXPERIMENT RESULTS")
    print("=" * 78)

    summary_df = results["summary_df"]
    print("\n[*] Comparative Model Performance Across Training Strategies (Test Partition):")
    print("-" * 78)
    # Format table nicely
    cols_to_print = ["asset", "model", "single_rmse", "pooled_rmse", "loao_rmse", "single_da", "pooled_da", "loao_da"]
    present_cols = [c for c in cols_to_print if c in summary_df.columns]
    print(summary_df[present_cols].to_string(index=False))

    print("\n" + "=" * 78)
    print(f"Artifacts saved to: {Path(args.artifacts_dir).resolve()}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
