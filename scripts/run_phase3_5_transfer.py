"""Execution script for Phase 3.5 Cross-Asset Transfer Learning & Volatility Adaptation.

Runs Leave-One-Asset-Out (LOAO) transfer experiments across:
- 3 Target Assets (BTC, ETH, SOL)
- 4 Modes (zero_shot_raw, zero_shot_vol_adapted, few_shot_vol_adapted, asset_specific_baseline)
- 6 Model families (ZeroReturn, MeanReturn, Ridge, LightGBM, CatBoost, XGBoost)
- 3 Multi-regime walk-forward folds with 24-bar purge gaps
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from crypto_prediction.pipeline.transfer_learning import TransferLearningEngine
from crypto_prediction.utils.logger import get_logger

logger = get_logger("run_phase3_5_transfer")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 3.5 Cross-Asset Transfer Learning & Volatility Adaptation."
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="data/processed/features_panel",
        help="Directory containing synchronized cross-asset feature panels.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/phase3_5",
        help="Directory where transfer experiment results and models will be saved.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="List of symbols to evaluate.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "zero_shot_raw",
            "zero_shot_vol_adapted",
            "few_shot_vol_adapted",
            "asset_specific_baseline",
        ],
        help="List of transfer learning modes to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ZeroReturn", "MeanReturn", "Ridge", "LightGBM", "CatBoost", "XGBoost"],
        help="List of model families to benchmark.",
    )
    parser.add_argument(
        "--few-shot-bars",
        type=int,
        default=1000,
        help="Number of historical training bars of target asset for few-shot adaptation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PHASE 3.5: CROSS-ASSET TRANSFER LEARNING & VOLATILITY ADAPTATION RUNNER")
    logger.info("=" * 80)
    logger.info(f"Features directory: {args.features_dir}")
    logger.info(f"Artifacts directory: {args.artifacts_dir}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Modes: {args.modes}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Few-shot bars: {args.few_shot_bars}")
    logger.info(f"Random seed: {args.seed}")

    start_time = time.time()
    engine = TransferLearningEngine(
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
        symbols=args.symbols,
        random_seed=args.seed,
        few_shot_bars=args.few_shot_bars,
    )

    df_results = engine.run_transfer_experiments(
        target_assets=args.symbols,
        modes=args.modes,
        models=args.models,
    )
    elapsed = time.time() - start_time

    logger.info("\n" + "=" * 80)
    logger.info(f"TRANSFER EXPERIMENTS COMPLETED in {elapsed:.1f}s ({len(df_results)} evaluations)")
    logger.info("=" * 80)

    # Print summary pivot by Mode and Model Family
    pivot_rmse = df_results.pivot_table(
        index="model_family",
        columns="transfer_mode",
        values="rmse",
        aggfunc="mean",
    )
    logger.info("\nMean RMSE by Model and Mode (Lower is Better):")
    print(pivot_rmse.to_string())

    pivot_dir_acc = df_results.pivot_table(
        index="model_family",
        columns="transfer_mode",
        values="directional_accuracy",
        aggfunc="mean",
    )
    logger.info("\nMean Directional Accuracy by Model and Mode (Higher is Better):")
    print(pivot_dir_acc.to_string())

    pivot_gain_raw = df_results.pivot_table(
        index="model_family",
        columns="transfer_mode",
        values="gain_vs_raw_pct",
        aggfunc="mean",
    )
    logger.info("\nMean Volatility Adaptation Gain vs Raw (%) (Higher is Better):")
    print(pivot_gain_raw.to_string())

    pivot_impr_base = df_results.pivot_table(
        index="model_family",
        columns="transfer_mode",
        values="improvement_vs_baseline_pct",
        aggfunc="mean",
    )
    logger.info("\nMean Improvement vs Asset-Specific Baseline (%) (Higher is Better):")
    print(pivot_impr_base.to_string())


if __name__ == "__main__":
    main()
