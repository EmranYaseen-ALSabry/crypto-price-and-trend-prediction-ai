"""Execution script for Phase 3.4 Controlled Multi-Asset Model Benchmarking.

Runs the 2x2 factorial walk-forward benchmark across:
- 4 Approaches (A: Single-Asset, B: Single+Cross, C: Multi-Asset, D: Multi+Cross)
- 6 Model families (ZeroReturn, MeanReturn, Ridge, LightGBM, CatBoost, XGBoost)
- 3 Multi-regime walk-forward folds with 24-bar purge gaps
- 3 Synchronized Assets (BTCUSDT, ETHUSDT, SOLUSDT)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from crypto_prediction.pipeline.benchmarking import ControlledBenchmarkEngine
from crypto_prediction.utils.logger import get_logger

logger = get_logger("run_phase3_4_benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3.4 Controlled Multi-Asset Benchmarking.")
    parser.add_argument(
        "--features-dir",
        type=str,
        default="data/processed/features_panel",
        help="Directory containing synchronized cross-asset feature panels.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/phase3_4",
        help="Directory where benchmark results and models will be saved.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="List of symbols to evaluate.",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=[
            "A_single_asset_baseline",
            "B_single_asset_cross_asset",
            "C_multi_asset_baseline",
            "D_multi_asset_cross_asset",
        ],
        help="List of approaches to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ZeroReturn", "MeanReturn", "Ridge", "LightGBM", "CatBoost", "XGBoost"],
        help="List of model families to benchmark.",
    )
    parser.add_argument(
        "--no-sample-weights",
        action="store_true",
        help="Disable volatility-inverse sample weighting for pooled models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PHASE 3.4: CONTROLLED MULTI-ASSET BENCHMARKING RUNNER")
    logger.info("=" * 80)
    logger.info(f"Features directory: {args.features_dir}")
    logger.info(f"Artifacts directory: {args.artifacts_dir}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Approaches: {args.approaches}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Use sample weighting: {not args.no_sample_weights}")
    logger.info(f"Random seed: {args.seed}")

    start_time = time.time()
    engine = ControlledBenchmarkEngine(
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
        symbols=args.symbols,
        random_seed=args.seed,
    )

    df_results = engine.run_benchmark(
        approaches=args.approaches,
        models=args.models,
        use_sample_weighting=not args.no_sample_weights,
    )
    elapsed = time.time() - start_time

    logger.info("\n" + "=" * 80)
    logger.info(f"BENCHMARK COMPLETED in {elapsed:.1f}s ({len(df_results)} evaluations)")
    logger.info("=" * 80)

    # Print summary pivot by Approach and Model Family
    pivot_rmse = df_results.pivot_table(
        index="model_family",
        columns="approach",
        values="rmse",
        aggfunc="mean",
    )
    logger.info("\nMean RMSE by Model and Approach (Lower is Better):")
    print(pivot_rmse.to_string())

    pivot_dir_acc = df_results.pivot_table(
        index="model_family",
        columns="approach",
        values="directional_accuracy",
        aggfunc="mean",
    )
    logger.info("\nMean Directional Accuracy by Model and Approach (Higher is Better):")
    print(pivot_dir_acc.to_string())

    pivot_impr_a = df_results.pivot_table(
        index="model_family",
        columns="approach",
        values="improvement_vs_approach_a_pct",
        aggfunc="mean",
    )
    logger.info("\nMean Relative Improvement vs Approach A (%) (Higher is Better):")
    print(pivot_impr_a.to_string())


if __name__ == "__main__":
    main()
