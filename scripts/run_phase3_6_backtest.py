"""Execution script for Phase 3.6 Realistic Trading Backtesting & Cost Sensitivity.

Runs the complete backtesting matrix across:
- Approaches: Approach A (Single), Approach D (Multi+Cross), LOAO (Zero-Shot)
- Models: CatBoost, LightGBM, Ridge, XGBoost, and Benchmarks (Buy & Hold, Cash)
- Policies: threshold_filter, pure_sign, high_conviction
- Cost Scenarios: Zero Cost, Institutional, Standard Retail, High Friction
- Multi-regime walk-forward folds (Fold 1, Fold 2, Fold 3)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from crypto_prediction.pipeline.backtest_engine_realistic import RealisticBacktestingEngine
from crypto_prediction.utils.logger import get_logger

logger = get_logger("run_phase3_6_backtest")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 3.6 Realistic Trading Backtesting & Cost Sensitivity."
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
        default="artifacts/phase3_6",
        help="Directory where backtest results and metrics will be saved.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="List of symbols to backtest.",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=["Approach_A_Single", "Approach_D_MultiCross", "LOAO_ZeroShot"],
        help="List of approaches to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["CatBoost", "LightGBM", "Ridge", "XGBoost"],
        help="List of forecasting models to benchmark.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["zero_cost", "institutional", "retail_standard", "high_friction"],
        help="List of cost scenarios to evaluate.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["threshold_filter", "pure_sign", "high_conviction"],
        help="List of decision policies to backtest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PHASE 3.6: REALISTIC TRADING BACKTESTING & COST SENSITIVITY RUNNER")
    logger.info("=" * 80)
    logger.info(f"Features directory: {args.features_dir}")
    logger.info(f"Artifacts directory: {args.artifacts_dir}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Approaches: {args.approaches}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Cost Scenarios: {args.scenarios}")
    logger.info(f"Policies: {args.policies}")
    logger.info(f"Random seed: {args.seed}")

    start_time = time.time()
    engine = RealisticBacktestingEngine(
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
        symbols=args.symbols,
        random_seed=args.seed,
    )

    df_results = engine.run_full_backtest_matrix(
        approaches=args.approaches,
        models=args.models,
        cost_scenarios=args.scenarios,
        policies=args.policies,
    )
    elapsed = time.time() - start_time

    logger.info("\n" + "=" * 80)
    logger.info(f"BACKTEST SIMULATIONS COMPLETED in {elapsed:.1f}s ({len(df_results)} simulations)")
    logger.info("=" * 80)

    # 1. Performance Summary across Cost Scenarios (for Threshold Filter Policy)
    df_thresh = df_results[df_results["policy"] == "threshold_filter"]
    pivot_net_ret = df_thresh.pivot_table(
        index=["approach", "model_name"],
        columns="cost_scenario",
        values="net_return",
        aggfunc="mean",
    )
    logger.info("\nMean Net Return (%) by Cost Scenario (Threshold Filter Policy):")
    print(pivot_net_ret.round(2).to_string())

    # 2. Annualized Sharpe Ratio by Cost Scenario
    pivot_sharpe = df_thresh.pivot_table(
        index=["approach", "model_name"],
        columns="cost_scenario",
        values="annualized_sharpe",
        aggfunc="mean",
    )
    logger.info("\nMean Annualized Sharpe Ratio by Cost Scenario:")
    print(pivot_sharpe.round(2).to_string())

    # 3. Policy Comparison under Retail Standard Costs
    df_retail = df_results[df_results["cost_scenario"] == "Standard Retail (Binance VIP0)"]
    pivot_pol = df_retail.pivot_table(
        index=["approach", "model_name"],
        columns="policy",
        values="net_return",
        aggfunc="mean",
    )
    logger.info("\nPolicy Comparison (Net Return %) under Standard Retail Costs (30 bps RT):")
    print(pivot_pol.round(2).to_string())

    # 4. Benchmark Performance (Buy & Hold vs Cash)
    df_bench = df_results[df_results["approach"] == "Benchmark"]
    pivot_bench = df_bench.pivot_table(
        index="model_name",
        columns="cost_scenario",
        values=["net_return", "annualized_sharpe", "max_drawdown"],
        aggfunc="mean",
    )
    logger.info("\nBenchmark Performance (Buy & Hold vs Cash):")
    print(pivot_bench.round(2).to_string())


if __name__ == "__main__":
    main()
