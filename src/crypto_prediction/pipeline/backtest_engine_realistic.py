"""Realistic Cryptocurrency Trading Strategy Backtesting and Cost Sensitivity Suite (Phase 3.6).

Features:
- Strict causal timestamp alignment: Feature close t -> Order submission t -> Execution at t+1.
- State-machine position tracking (Cash = 0, Long = 1, Short = -1).
- Multi-scenario cost accounting: Maker/taker exchange fees, bid-ask spread, and execution slippage.
- Turnover tracking: Entry (|1 - 0| = 1), Exit (|0 - 1| = 1), Reversal (|-1 - 1| = 2).
- Final liquidation to cash at simulation close.
- Compares Model Strategies against Buy-and-Hold and No-Trade Cash Baselines.
- Cost sensitivity analysis from 0 bps to 50 bps round-trip friction.
- Annualized Sharpe (hourly annualization factor sqrt(8760) ~ 93.595).
- Statistical reliability auditing (flags trade count < 30).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from crypto_prediction.data.collector import compute_file_checksum
from crypto_prediction.pipeline.benchmarking import ControlledBenchmarkEngine, WalkForwardFoldSpec
from crypto_prediction.pipeline.transfer_learning import TransferLearningEngine
from crypto_prediction.utils.logger import get_logger
from crypto_prediction.utils.reproducibility import set_seed

logger = get_logger("backtest_engine_realistic")


@dataclass
class CostScenario:
    """Trading cost configuration for fees and slippage."""

    name: str
    fee_rate: float
    slippage_rate: float
    description: str

    @property
    def unit_cost(self) -> float:
        """One-way friction per order (fee + slippage)."""
        return self.fee_rate + self.slippage_rate

    @property
    def round_trip_cost(self) -> float:
        """Round-trip friction for opening and closing a position."""
        return 2.0 * self.unit_cost


@dataclass
class RealisticBacktestReport:
    """Detailed performance and risk metrics for a backtested strategy."""

    model_name: str
    approach: str
    symbol: str
    fold_id: int
    fold_name: str
    cost_scenario: str
    policy: str
    total_bars: int
    total_trades: int
    turnover: float
    exposure_pct: float
    gross_return: float
    total_cost_drag: float
    net_return: float
    benchmark_return: float
    net_alpha: float
    annualized_sharpe: float
    annualized_sortino: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    statistically_reliable: bool
    breakeven_cost_bps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "approach": self.approach,
            "symbol": self.symbol,
            "fold_id": self.fold_id,
            "fold_name": self.fold_name,
            "cost_scenario": self.cost_scenario,
            "policy": self.policy,
            "total_bars": self.total_bars,
            "total_trades": self.total_trades,
            "turnover": self.turnover,
            "exposure_pct": self.exposure_pct,
            "gross_return": self.gross_return,
            "total_cost_drag": self.total_cost_drag,
            "net_return": self.net_return,
            "benchmark_return": self.benchmark_return,
            "net_alpha": self.net_alpha,
            "annualized_sharpe": self.annualized_sharpe,
            "annualized_sortino": self.annualized_sortino,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "statistically_reliable": self.statistically_reliable,
            "breakeven_cost_bps": self.breakeven_cost_bps,
        }


class RealisticBacktestingEngine:
    """Orchestrates Phase 3.6 Realistic Trading Backtesting & Cost Sensitivity."""

    COST_SCENARIOS = {
        "zero_cost": CostScenario(
            name="Zero Cost (Gross)",
            fee_rate=0.0,
            slippage_rate=0.0,
            description="Theoretical gross returns without friction (0 bps RT).",
        ),
        "institutional": CostScenario(
            name="Institutional / VIP",
            fee_rate=0.0005,
            slippage_rate=0.0002,
            description="5 bps fee + 2 bps slippage/spread (14 bps RT).",
        ),
        "retail_standard": CostScenario(
            name="Standard Retail (Binance VIP0)",
            fee_rate=0.0010,
            slippage_rate=0.0005,
            description="10 bps taker fee + 5 bps slippage/spread (30 bps RT).",
        ),
        "high_friction": CostScenario(
            name="High Friction / Stress Test",
            fee_rate=0.0015,
            slippage_rate=0.0010,
            description="15 bps fee + 10 bps slippage/spread (50 bps RT).",
        ),
    }

    def __init__(
        self,
        features_dir: str | Path = "data/processed/features_panel",
        artifacts_dir: str | Path = "artifacts/phase3_6",
        symbols: list[str] | None = None,
        random_seed: int = 42,
        annualization_factor: float = np.sqrt(8760),
        min_trades: int = 30,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.symbols = [s.upper() for s in (symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"])]
        self.seed = random_seed
        self.annualization_factor = annualization_factor
        self.min_trades = min_trades
        set_seed(self.seed)

    @staticmethod
    def simulate_policy_execution(
        r_true: np.ndarray,
        r_pred: np.ndarray,
        cost: CostScenario,
        policy: str = "threshold_filter",
        allow_short: bool = False,
        annualization_factor: float = np.sqrt(8760),
        min_trades: int = 30,
    ) -> dict[str, Any]:
        """Simulates causal trading execution under a specific decision policy and cost scenario."""
        y_t = np.asarray(r_true, dtype=np.float64)
        y_p = np.asarray(r_pred, dtype=np.float64)
        n = len(y_t)

        unit_cost = cost.unit_cost
        round_trip_cost = cost.round_trip_cost

        # Determine target threshold based on policy
        if policy == "threshold_filter":
            # Only enter if expected return exceeds round-trip friction
            entry_threshold = round_trip_cost
            exit_threshold = 0.0
        elif policy == "pure_sign":
            # Enter on any positive forecast
            entry_threshold = 0.0
            exit_threshold = 0.0
        elif policy == "high_conviction":
            # Enter only on high conviction moves (e.g. 25 bps)
            entry_threshold = max(0.0025, round_trip_cost)
            exit_threshold = 0.0
        elif policy == "buy_and_hold":
            # Always long 1.0
            entry_threshold = -999.0
            exit_threshold = -999.0
        elif policy == "cash_baseline":
            # Always cash 0.0
            entry_threshold = 999.0
            exit_threshold = 999.0
        else:
            raise ValueError(f"Unknown policy: {policy}")

        positions = np.zeros(n, dtype=np.float64)
        curr_pos = 0.0

        if policy == "buy_and_hold":
            positions.fill(1.0)
        elif policy == "cash_baseline":
            positions.fill(0.0)
        else:
            for t in range(n):
                signal = y_p[t]
                if curr_pos == 0.0:
                    if signal > entry_threshold:
                        curr_pos = 1.0
                    elif allow_short and signal < -entry_threshold:
                        curr_pos = -1.0
                elif curr_pos == 1.0:
                    if allow_short and signal < -entry_threshold:
                        curr_pos = -1.0
                    elif signal <= exit_threshold:
                        curr_pos = 0.0
                elif curr_pos == -1.0:
                    if signal > entry_threshold:
                        curr_pos = 1.0
                    elif signal >= -exit_threshold:
                        curr_pos = 0.0
                positions[t] = curr_pos

        # Calculate position changes (turnover)
        pos_prev = np.concatenate([[0.0], positions[:-1]])
        order_turnover = np.abs(positions - pos_prev)
        total_trades = int(np.count_nonzero(order_turnover))

        # Transaction friction incurred per bar
        friction_per_bar = order_turnover * unit_cost

        # Gross & Net period returns
        gross_period_returns = positions * y_t
        net_period_returns = gross_period_returns - friction_per_bar

        # Final liquidation to cash at simulation end
        final_turnover = 0.0
        if positions[-1] != 0.0:
            final_turnover = float(np.abs(positions[-1]))
            final_cost = final_turnover * unit_cost
            net_period_returns[-1] -= final_cost
            friction_per_bar[-1] += final_cost
            total_trades += 1

        total_cost_drag = float(np.sum(friction_per_bar))
        total_turnover = float(np.sum(order_turnover) + final_turnover)

        # Wealth curves starting from initial equity = 1.0
        equity_curve = np.concatenate([[1.0], np.cumprod(np.maximum(0.0, 1.0 + net_period_returns))])
        net_cum_ret = float(equity_curve[-1] - 1.0) if len(equity_curve) > 0 else 0.0

        gross_equity = np.concatenate([[1.0], np.cumprod(np.maximum(0.0, 1.0 + gross_period_returns))])
        gross_cum_ret = float(gross_equity[-1] - 1.0) if len(gross_equity) > 0 else 0.0

        # Buy & Hold benchmark starting from initial equity = 1.0
        # Incurs 2x unit_cost (entry and exit)
        bench_ret_series = y_t.copy()
        if len(bench_ret_series) > 0:
            bench_ret_series[0] -= unit_cost
            bench_ret_series[-1] -= unit_cost
        bench_curve = np.concatenate([[1.0], np.cumprod(np.maximum(0.0, 1.0 + bench_ret_series))])
        bench_cum_ret = float(bench_curve[-1] - 1.0) if len(bench_curve) > 0 else 0.0

        excess_alpha = net_cum_ret - bench_cum_ret

        # Exposure (% of time with active position)
        exposure_pct = float(np.mean(positions != 0.0)) * 100.0

        # Win rate on active trading periods
        active_periods = (positions != 0.0)
        if np.any(active_periods):
            win_rate = float(np.mean(net_period_returns[active_periods] > 0.0)) * 100.0
        else:
            win_rate = 0.0

        # Sharpe ratio
        mean_r = float(np.mean(net_period_returns))
        std_r = float(np.std(net_period_returns))
        if std_r > 1e-9:
            sharpe = float((mean_r / std_r) * annualization_factor)
        else:
            sharpe = 0.0

        # Sortino ratio
        downside = net_period_returns[net_period_returns < 0.0]
        downside_std = float(np.std(downside)) if len(downside) > 0 else 0.0
        if downside_std > 1e-9:
            sortino = float((mean_r / downside_std) * annualization_factor)
        else:
            sortino = 0.0

        # Maximum Drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / (running_max + 1e-12)
        max_dd = float(np.min(drawdowns)) * 100.0 if len(drawdowns) > 0 else 0.0

        # Profit Factor (Gross Profits / Gross Losses)
        pos_rets = net_period_returns[net_period_returns > 0.0]
        neg_rets = net_period_returns[net_period_returns < 0.0]
        gross_win = float(np.sum(pos_rets)) if len(pos_rets) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(neg_rets))) if len(neg_rets) > 0 else 0.0
        if gross_loss > 1e-9:
            profit_factor = float(gross_win / gross_loss)
        else:
            profit_factor = 99.0 if gross_win > 0.0 else 0.0

        # Breakeven cost calculation:
        # At what unit cost does gross_return == total_cost_drag?
        if total_turnover > 1e-9:
            breakeven_unit_cost = gross_cum_ret / total_turnover
            breakeven_bps = float(breakeven_unit_cost * 10000.0 * 2.0)  # Round-trip bps
        else:
            breakeven_bps = 0.0

        return {
            "total_bars": n,
            "total_trades": total_trades,
            "turnover": total_turnover,
            "exposure_pct": exposure_pct,
            "gross_return": gross_cum_ret * 100.0,
            "total_cost_drag": total_cost_drag * 100.0,
            "net_return": net_cum_ret * 100.0,
            "benchmark_return": bench_cum_ret * 100.0,
            "net_alpha": excess_alpha * 100.0,
            "annualized_sharpe": sharpe,
            "annualized_sortino": sortino,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "statistically_reliable": bool(total_trades >= min_trades),
            "breakeven_cost_bps": breakeven_bps,
        }

    def run_full_backtest_matrix(
        self,
        approaches: list[str] | None = None,
        models: list[str] | None = None,
        cost_scenarios: list[str] | None = None,
        policies: list[str] | None = None,
    ) -> pd.DataFrame:
        """Runs the complete backtesting and cost sensitivity matrix across models, folds, and scenarios."""
        target_approaches = approaches or ["Approach_A_Single", "Approach_D_MultiCross", "LOAO_ZeroShot"]
        target_models = models or ["CatBoost", "LightGBM", "Ridge", "XGBoost"]
        target_scenarios = cost_scenarios or list(self.COST_SCENARIOS.keys())
        target_policies = policies or ["threshold_filter", "pure_sign", "high_conviction"]

        logger.info("Initializing Backtest Engine: Loading features panel...")
        bench_engine = ControlledBenchmarkEngine(
            features_dir=self.features_dir,
            symbols=self.symbols,
            random_seed=self.seed,
        )
        datasets = bench_engine.load_feature_datasets()
        first_sym = self.symbols[0]
        n_bars = len(datasets[first_sym])
        folds = bench_engine.define_walk_forward_folds(n_bars)

        # Feature partitioning
        all_cols = [c for c in datasets[first_sym].columns if not c.startswith("target_") and c not in ["open", "high", "low", "close", "volume", "symbol"]]
        asset_cols, cross_cols = bench_engine.partition_feature_columns(all_cols)
        common_cols = set.intersection(*(set(datasets[s].columns) for s in self.symbols))
        asset_cols = [c for c in asset_cols if c in common_cols]
        cross_cols = [c for c in cross_cols if c in common_cols]

        records: list[dict[str, Any]] = []

        # Generate walk-forward out-of-sample predictions
        for fold in folds:
            logger.info(f"\n--- Backtesting {fold.name} ({fold.test_start_idx} to {fold.test_end_idx}) ---")

            # Extract splits per symbol
            splits = {}
            for sym in self.symbols:
                df = datasets[sym]
                y_train = df["target_return_1"].iloc[fold.train_start_idx:fold.train_end_idx]
                y_test = df["target_return_1"].iloc[fold.test_start_idx:fold.test_end_idx]
                X_train_a = df[asset_cols].iloc[fold.train_start_idx:fold.train_end_idx]
                X_test_a = df[asset_cols].iloc[fold.test_start_idx:fold.test_end_idx]
                X_train_c = df[cross_cols].iloc[fold.train_start_idx:fold.train_end_idx]
                X_test_c = df[cross_cols].iloc[fold.test_start_idx:fold.test_end_idx]
                splits[sym] = {
                    "y_train": y_train,
                    "y_test": y_test,
                    "X_train_a": X_train_a,
                    "X_test_a": X_test_a,
                    "X_train_c": X_train_c,
                    "X_test_c": X_test_c,
                }

            # Pre-compute pooled datasets for Approach D
            pooled_train_cross = []
            pooled_train_y = []
            for sym in self.symbols:
                df_c_hot = splits[sym]["X_train_c"].copy()
                for o_sym in self.symbols:
                    df_c_hot[f"asset_{o_sym}"] = 1.0 if o_sym == sym else 0.0
                pooled_train_cross.append(df_c_hot)
                pooled_train_y.append(splits[sym]["y_train"])
            pooled_X_c = pd.concat(pooled_train_cross, axis=0)
            pooled_y_c = pd.concat(pooled_train_y, axis=0)

            # Fit and collect predictions per approach and model
            for model_name in target_models:
                m_factory = ControlledBenchmarkEngine.MODEL_FACTORIES[model_name]

                # 1. Approach A: Single-Asset Baseline
                if "Approach_A_Single" in target_approaches:
                    for sym in self.symbols:
                        s = splits[sym]
                        m = m_factory()
                        m.fit(s["X_train_a"], s["y_train"])
                        preds = m.predict(s["X_test_a"])

                        # Run backtest across scenarios and policies
                        for scen_key in target_scenarios:
                            scen = self.COST_SCENARIOS[scen_key]
                            for pol in target_policies:
                                res = self.simulate_policy_execution(
                                    r_true=s["y_test"].to_numpy(),
                                    r_pred=preds,
                                    cost=scen,
                                    policy=pol,
                                    annualization_factor=self.annualization_factor,
                                    min_trades=self.min_trades,
                                )
                                records.append({
                                    "model_name": model_name,
                                    "approach": "Approach_A_Single",
                                    "symbol": sym,
                                    "fold_id": fold.fold_id,
                                    "fold_name": fold.name,
                                    "cost_scenario": scen.name,
                                    "policy": pol,
                                    **res,
                                })

                # 2. Approach D: Multi-Asset + Cross-Asset (Pooled)
                if "Approach_D_MultiCross" in target_approaches:
                    m_shared = m_factory()
                    m_shared.fit(pooled_X_c, pooled_y_c)

                    for sym in self.symbols:
                        s = splits[sym]
                        X_test_hot = s["X_test_c"].copy()
                        for o_sym in self.symbols:
                            X_test_hot[f"asset_{o_sym}"] = 1.0 if o_sym == sym else 0.0
                        preds_d = m_shared.predict(X_test_hot)

                        for scen_key in target_scenarios:
                            scen = self.COST_SCENARIOS[scen_key]
                            for pol in target_policies:
                                res = self.simulate_policy_execution(
                                    r_true=s["y_test"].to_numpy(),
                                    r_pred=preds_d,
                                    cost=scen,
                                    policy=pol,
                                    annualization_factor=self.annualization_factor,
                                    min_trades=self.min_trades,
                                )
                                records.append({
                                    "model_name": model_name,
                                    "approach": "Approach_D_MultiCross",
                                    "symbol": sym,
                                    "fold_id": fold.fold_id,
                                    "fold_name": fold.name,
                                    "cost_scenario": scen.name,
                                    "policy": pol,
                                    **res,
                                })

                # 3. LOAO Zero-Shot Transfer
                if "LOAO_ZeroShot" in target_approaches:
                    for target_sym in self.symbols:
                        source_syms = [s for s in self.symbols if s != target_sym]
                        X_source = pd.concat([splits[s]["X_train_a"] for s in source_syms], axis=0)
                        y_source = pd.concat([splits[s]["y_train"] for s in source_syms], axis=0)

                        m_loao = m_factory()
                        m_loao.fit(X_source, y_source)
                        preds_loao = m_loao.predict(splits[target_sym]["X_test_a"])

                        for scen_key in target_scenarios:
                            scen = self.COST_SCENARIOS[scen_key]
                            for pol in target_policies:
                                res = self.simulate_policy_execution(
                                    r_true=splits[target_sym]["y_test"].to_numpy(),
                                    r_pred=preds_loao,
                                    cost=scen,
                                    policy=pol,
                                    annualization_factor=self.annualization_factor,
                                    min_trades=self.min_trades,
                                )
                                records.append({
                                    "model_name": model_name,
                                    "approach": "LOAO_ZeroShot",
                                    "symbol": target_sym,
                                    "fold_id": fold.fold_id,
                                    "fold_name": fold.name,
                                    "cost_scenario": scen.name,
                                    "policy": pol,
                                    **res,
                                })

            # Also evaluate Buy-and-Hold & Cash baselines for this fold
            for sym in self.symbols:
                s = splits[sym]
                for scen_key in target_scenarios:
                    scen = self.COST_SCENARIOS[scen_key]
                    # Buy and Hold
                    res_bh = self.simulate_policy_execution(
                        r_true=s["y_test"].to_numpy(),
                        r_pred=np.ones_like(s["y_test"].to_numpy()),
                        cost=scen,
                        policy="buy_and_hold",
                        annualization_factor=self.annualization_factor,
                        min_trades=self.min_trades,
                    )
                    records.append({
                        "model_name": "BuyAndHold",
                        "approach": "Benchmark",
                        "symbol": sym,
                        "fold_id": fold.fold_id,
                        "fold_name": fold.name,
                        "cost_scenario": scen.name,
                        "policy": "buy_and_hold",
                        **res_bh,
                    })

                    # Cash Baseline
                    res_cash = self.simulate_policy_execution(
                        r_true=s["y_test"].to_numpy(),
                        r_pred=np.zeros_like(s["y_test"].to_numpy()),
                        cost=scen,
                        policy="cash_baseline",
                        annualization_factor=self.annualization_factor,
                        min_trades=self.min_trades,
                    )
                    records.append({
                        "model_name": "CashBaseline",
                        "approach": "Benchmark",
                        "symbol": sym,
                        "fold_id": fold.fold_id,
                        "fold_name": fold.name,
                        "cost_scenario": scen.name,
                        "policy": "cash_baseline",
                        **res_cash,
                    })

        df_results = pd.DataFrame(records)

        # Save artifacts
        out_csv = self.artifacts_dir / "phase3_6_backtest_results.csv"
        df_results.to_csv(out_csv, index=False)
        meta_dict = {
            "total_simulations": len(df_results),
            "approaches": target_approaches,
            "models": target_models,
            "symbols": self.symbols,
            "n_folds": len(folds),
            "cost_scenarios": list(self.COST_SCENARIOS.keys()),
            "policies": target_policies,
            "sha256": compute_file_checksum(out_csv),
            "generated_at_utc": str(pd.Timestamp.now(tz="UTC")),
        }
        with open(out_csv.with_name(f"{out_csv.name}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2, default=str)

        logger.info(
            f"Successfully executed Phase 3.6 Realistic Backtesting ({len(df_results)} simulations). "
            f"Results saved to: {out_csv}"
        )
        return df_results
