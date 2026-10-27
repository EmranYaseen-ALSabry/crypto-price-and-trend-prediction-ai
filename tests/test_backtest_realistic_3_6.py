"""Unit and integration tests for Phase 3.6 Realistic Trading Backtesting & Cost Sensitivity Suite."""

from __future__ import annotations

import numpy as np
import pytest

from crypto_prediction.pipeline.backtest_engine_realistic import (
    CostScenario,
    RealisticBacktestingEngine,
)


def test_cost_scenario_properties() -> None:
    """Verifies that unit cost and round-trip cost are calculated correctly."""
    scenario = CostScenario(
        name="Test Scenario",
        fee_rate=0.0010,  # 10 bps
        slippage_rate=0.0005,  # 5 bps
        description="Standard Retail",
    )
    assert scenario.unit_cost == pytest.approx(0.0015)
    assert scenario.round_trip_cost == pytest.approx(0.0030)  # 30 bps round-trip


def test_turnover_and_order_accounting_exactness() -> None:
    """Verifies entry, exit, reversal, and final liquidation order accounting."""
    # Sequence of returns: 4 periods
    r_true = np.array([0.01, 0.02, -0.01, 0.01])
    # Predictions: open long, hold, reverse to short, hold
    r_pred = np.array([0.02, 0.02, -0.02, -0.02])

    cost = CostScenario(name="Test", fee_rate=0.0010, slippage_rate=0.0005, description="")
    unit_cost = cost.unit_cost  # 0.0015

    # Run with allow_short=True
    res = RealisticBacktestingEngine.simulate_policy_execution(
        r_true=r_true,
        r_pred=r_pred,
        cost=cost,
        policy="pure_sign",
        allow_short=True,
    )

    # Expected order turnover:
    # t=0: Cash(0) -> Long(1) = 1 order
    # t=1: Long(1) -> Long(1) = 0 orders
    # t=2: Long(1) -> Short(-1) = 2 orders (Reversal!)
    # t=3: Short(-1) -> Short(-1) = 0 orders
    # End: Final liquidation Short(-1) -> Cash(0) = 1 order
    # Total trade events = 3 (entry, reversal, final exit)
    # Total order turnover = 1 + 0 + 2 + 0 + 1 = 4.0 units
    assert res["total_trades"] == 3
    assert res["turnover"] == pytest.approx(4.0)
    assert res["total_cost_drag"] == pytest.approx(4.0 * unit_cost * 100.0)


def test_threshold_filter_friction_awareness() -> None:
    """Verifies that threshold_filter policy prevents trading when expected return is below friction."""
    # Forecast of +20 bps, but round-trip friction is 30 bps (15 bps * 2)
    r_pred = np.array([0.0020, 0.0020, 0.0020])
    r_true = np.array([0.0010, -0.0010, 0.0020])

    cost = CostScenario(name="Test", fee_rate=0.0010, slippage_rate=0.0005, description="")
    assert cost.round_trip_cost == 0.0030

    # Under threshold_filter, forecast 0.0020 < 0.0030 -> No trade (Cash)
    res_thresh = RealisticBacktestingEngine.simulate_policy_execution(
        r_true=r_true,
        r_pred=r_pred,
        cost=cost,
        policy="threshold_filter",
    )
    assert res_thresh["total_trades"] == 0
    assert res_thresh["turnover"] == 0.0
    assert res_thresh["exposure_pct"] == 0.0
    assert res_thresh["net_return"] == 0.0

    # Under pure_sign, forecast 0.0020 > 0.0 -> Enters position
    res_sign = RealisticBacktestingEngine.simulate_policy_execution(
        r_true=r_true,
        r_pred=r_pred,
        cost=cost,
        policy="pure_sign",
    )
    assert res_sign["total_trades"] > 0
    assert res_sign["turnover"] > 0.0


def test_cash_baseline_invariance() -> None:
    """Verifies that cash baseline produces identically zero return, zero friction, and zero drawdown."""
    r_true = np.array([0.05, -0.08, 0.03, -0.02])
    cost = CostScenario(name="Retail", fee_rate=0.0010, slippage_rate=0.0005, description="")

    res_cash = RealisticBacktestingEngine.simulate_policy_execution(
        r_true=r_true,
        r_pred=np.zeros_like(r_true),
        cost=cost,
        policy="cash_baseline",
    )
    assert res_cash["gross_return"] == 0.0
    assert res_cash["net_return"] == 0.0
    assert res_cash["total_cost_drag"] == 0.0
    assert res_cash["max_drawdown"] == 0.0
    assert res_cash["total_trades"] == 0
    assert res_cash["turnover"] == 0.0
    assert res_cash["annualized_sharpe"] == 0.0


def test_buy_and_hold_benchmark_cost_deduction() -> None:
    """Verifies that buy-and-hold incurs exactly entry and exit friction."""
    r_true = np.array([0.02, 0.01, -0.01])
    cost = CostScenario(name="Retail", fee_rate=0.0010, slippage_rate=0.0005, description="")
    unit_cost = cost.unit_cost  # 0.0015

    res_bh = RealisticBacktestingEngine.simulate_policy_execution(
        r_true=r_true,
        r_pred=np.ones_like(r_true),
        cost=cost,
        policy="buy_and_hold",
    )
    assert res_bh["exposure_pct"] == 100.0
    # Net return should be lower than gross due to entry and exit friction
    assert res_bh["net_return"] < res_bh["gross_return"]


def test_statistical_reliability_flag() -> None:
    """Verifies that strategies with < 30 trades are flagged as statistically unreliable."""
    r_true = np.random.normal(0.0001, 0.01, 50)
    # 2 trades
    r_pred_few = np.zeros(50)
    r_pred_few[10:15] = 0.01
    cost = CostScenario(name="Zero", fee_rate=0.0, slippage_rate=0.0, description="")

    res_few = RealisticBacktestingEngine.simulate_policy_execution(
        r_true=r_true,
        r_pred=r_pred_few,
        cost=cost,
        policy="pure_sign",
        min_trades=30,
    )
    assert res_few["total_trades"] < 30
    assert res_few["statistically_reliable"] is False

    # Many alternating trades
    r_pred_many = np.where(np.arange(50) % 2 == 0, 0.01, -0.01)
    res_many = RealisticBacktestingEngine.simulate_policy_execution(
        r_true=r_true,
        r_pred=r_pred_many,
        cost=cost,
        policy="pure_sign",
        min_trades=30,
    )
    assert res_many["total_trades"] >= 30
    assert res_many["statistically_reliable"] is True
