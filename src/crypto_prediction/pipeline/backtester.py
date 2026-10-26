"""Realistic Cryptocurrency Trading Strategy Backtester with Fees and Slippage."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from crypto_prediction.utils.logger import get_logger

logger = get_logger("backtester")


@dataclass
class BacktestResult:
    """Encapsulates realistic financial performance of a predictive trading strategy."""

    model_name: str
    total_trades: int
    win_rate: float
    cumulative_return: float
    benchmark_return: float
    excess_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    profit_factor: float
    fee_rate: float
    slippage_rate: float
    min_trades: int = 30
    statistically_reliable: bool = False

    def summary(self) -> str:
        return (
            f"--- Backtest Performance for {self.model_name} (Fee: {self.fee_rate*100:.2f}%, Slippage: {self.slippage_rate*100:.2f}%) ---\n"
            f"Total Trades:        {self.total_trades}\n"
            f"Trade Win Rate:      {self.win_rate*100:.2f}%\n"
            f"Strategy Net Return: {self.cumulative_return*100:+.2f}%\n"
            f"Buy & Hold Return:   {self.benchmark_return*100:+.2f}%\n"
            f"Excess Alpha:        {self.excess_return*100:+.2f}%\n"
            f"Annualized Sharpe:   {self.sharpe_ratio:.2f}\n"
            f"Max Drawdown:        {self.max_drawdown*100:.2f}%\n"
            f"Profit Factor:       {self.profit_factor:.2f}\n"
            + (
                ""
                if self.statistically_reliable
                else f"NOTE: only {self.total_trades} trades (< {self.min_trades}); Sharpe/win-rate are not statistically meaningful.\n"
            )
        )

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "model_name": self.model_name,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "cumulative_return": self.cumulative_return,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "statistically_reliable": self.statistically_reliable,
        }


class StrategyBacktester:
    """Simulates realistic out-of-sample trading execution with friction (fees + slippage)."""

    def __init__(
        self,
        fee_rate: float = 0.0010,  # 0.10% standard Binance Spot taker/maker fee
        slippage_rate: float = 0.0005,  # 0.05% realistic execution slippage
        allow_short: bool = False,  # Spot trading (Long or Cash only) by default
        annualization_factor: float = np.sqrt(8760),  # 8,760 hourly bars per year
        min_trades: int = 30,  # below this, risk/return statistics are flagged as unreliable
    ) -> None:
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.allow_short = allow_short
        self.annualization_factor = annualization_factor
        self.min_trades = min_trades

    def run_backtest(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        model_name: str = "Model",
        horizon: int = 1,
        threshold: float | None = None,
        non_overlapping: bool = True,
    ) -> BacktestResult:
        """Executes trading simulation on out-of-sample predictions with friction and horizon awareness.

        Args:
            y_true: Realized forward returns (e.g. over horizon h).
            y_pred: Forecasted future returns from the model.
            model_name: Name of the model being evaluated.
            horizon: Holding period / target horizon in bars.
            threshold: Minimum forecasted return to justify trading over friction costs.
                       Defaults to fee_rate + slippage_rate.
            non_overlapping: If True and horizon > 1, evaluates independent non-overlapping
                             holding steps of length h to avoid compounding overlapping forward returns.

        Returns:
            BacktestResult dataclass instance.
        """
        r_true_full = np.asarray(y_true, dtype=np.float64)
        r_pred_full = np.asarray(y_pred, dtype=np.float64)

        valid = np.isfinite(r_true_full) & np.isfinite(r_pred_full)
        r_true_full = r_true_full[valid]
        r_pred_full = r_pred_full[valid]
        n_full = len(r_true_full)

        if n_full == 0:
            return BacktestResult(
                model_name=model_name,
                total_trades=0,
                win_rate=0.0,
                cumulative_return=0.0,
                benchmark_return=0.0,
                excess_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                profit_factor=0.0,
                fee_rate=self.fee_rate,
                slippage_rate=self.slippage_rate,
                min_trades=self.min_trades,
                statistically_reliable=False,
            )

        # Handle multi-step overlapping returns: subsample at non-overlapping steps
        if non_overlapping and horizon > 1 and n_full > horizon:
            step = horizon
            sample_indices = np.arange(0, n_full, step)
            r_true = r_true_full[sample_indices]
            r_pred = r_pred_full[sample_indices]
        else:
            r_true = r_true_full
            r_pred = r_pred_full

        n = len(r_true)

        # Compute accurate annualization factor based on holding period frequency
        bars_per_year = 8760.0 / max(1, horizon)
        annualization_factor = np.sqrt(bars_per_year)

        # Execution cost per round-trip/rebalance
        unit_cost = self.fee_rate + self.slippage_rate
        trade_threshold = threshold if threshold is not None else unit_cost

        # Determine target position: +1 (Long), 0 (Cash), -1 (Short if allowed)
        positions = np.zeros(n, dtype=np.float64)
        positions[r_pred > trade_threshold] = 1.0
        if self.allow_short:
            positions[r_pred < -trade_threshold] = -1.0

        # Calculate position changes (turnover)
        pos_prev = np.concatenate([[0.0], positions[:-1]])
        trade_turnover = np.abs(positions - pos_prev)
        total_trades = int(np.count_nonzero(trade_turnover))

        # Friction transaction costs
        friction = trade_turnover * unit_cost

        # Strategy net return per period
        net_returns = (positions * r_true) - friction

        # Account for closing any active position back to cash at the end of the simulation
        if positions[-1] != 0.0:
            final_exit_friction = np.abs(positions[-1]) * unit_cost
            net_returns[-1] -= final_exit_friction
            total_trades += 1

        # Cumulative wealth curve starting strictly from initial capital = 1.0
        equity_curve = np.concatenate([[1.0], np.cumprod(np.maximum(0.0, 1.0 + net_returns))])
        cum_ret = float(equity_curve[-1] - 1.0) if len(equity_curve) > 0 else 0.0

        # Benchmark (Buy and Hold) starting strictly from initial capital = 1.0
        benchmark_curve = np.concatenate([[1.0], np.cumprod(np.maximum(0.0, 1.0 + r_true))])
        bench_ret = float(benchmark_curve[-1] - 1.0) if len(benchmark_curve) > 0 else 0.0
        excess_ret = cum_ret - bench_ret

        # Win rate on active trading periods
        active_periods = (positions != 0)
        if np.any(active_periods):
            win_rate = float(np.mean(net_returns[active_periods] > 0))
        else:
            win_rate = 0.0

        # Sharpe ratio
        mean_ret = np.mean(net_returns)
        std_ret = np.std(net_returns)
        if std_ret > 1e-9:
            sharpe = float((mean_ret / std_ret) * annualization_factor)
        else:
            sharpe = 0.0

        # Sortino ratio (downside deviation)
        downside = net_returns[net_returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 0.0
        if downside_std > 1e-9:
            sortino = float((mean_ret / downside_std) * annualization_factor)
        else:
            sortino = 0.0

        # Maximum Drawdown calculated strictly from initial capital = 1.0 and running peak
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Profit factor
        gross_profits = np.sum(net_returns[net_returns > 0])
        gross_losses = np.abs(np.sum(net_returns[net_returns < 0]))
        if gross_losses > 1e-9:
            profit_factor = float(gross_profits / gross_losses)
        else:
            profit_factor = 99.0 if gross_profits > 0 else 0.0

        return BacktestResult(
            model_name=model_name,
            total_trades=total_trades,
            win_rate=win_rate,
            cumulative_return=cum_ret,
            benchmark_return=bench_ret,
            excess_return=excess_ret,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
            min_trades=self.min_trades,
            statistically_reliable=total_trades >= self.min_trades,
        )
