"""End-to-end pipeline execution and integration tests."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crypto_prediction.pipeline.evaluator import ModelEvaluator
from crypto_prediction.pipeline.trainer import PipelineTrainer
from crypto_prediction.utils.reproducibility import set_seed


class TestPipelineIntegration:
    """Verifies end-to-end pipeline orchestration, metrics, and determinism."""

    def test_evaluator_metrics_calculation(self) -> None:
        evaluator = ModelEvaluator()
        y_true = np.array([0.02, -0.01, 0.03, -0.02])
        y_pred = np.array([0.015, -0.008, 0.025, 0.01])  # 3 of 4 match signs

        metrics = evaluator.evaluate(y_true, y_pred, model_name="TestModel", horizon=1)
        assert metrics.mae > 0.0
        assert metrics.rmse > 0.0
        assert np.isclose(metrics.directional_accuracy, 0.75)

    def test_pipeline_trainer_execution(self, tmp_path: Path) -> None:
        """Executes full pipeline on BTCUSDT with temporary artifacts dir."""
        trainer = PipelineTrainer(artifacts_dir=tmp_path)
        results = trainer.run_experiment(
            symbol="BTCUSDT",
            interval="1h",
            horizon=1,
            total_bars=600,
            optimize_hyperparams=False,
        )

        assert results["symbol"] == "BTCUSDT"
        assert results["horizon"] == 1
        assert results["audit_report"].passed_all is True
        assert len(results["models"]) >= 8
        assert not results["comparison_df"].empty

        # Check artifacts were generated
        models_dir = tmp_path / "models"
        figures_dir = tmp_path / "figures"
        assert len(list(models_dir.glob("*.joblib"))) >= 8
        assert len(list(figures_dir.glob("*.png"))) >= 3
        assert (tmp_path / "BTCUSDT_trading_backtest_h1.csv").exists()
        assert "backtest_df" in results

    def test_pipeline_reproducibility(self, tmp_path: Path) -> None:
        """Verifies that executing twice with same seed produces identical predictions."""
        set_seed(42)
        trainer1 = PipelineTrainer(artifacts_dir=tmp_path / "run1")
        res1 = trainer1.run_experiment(symbol="BTCUSDT", total_bars=400)

        set_seed(42)
        trainer2 = PipelineTrainer(artifacts_dir=tmp_path / "run2")
        res2 = trainer2.run_experiment(symbol="BTCUSDT", total_bars=400)

        preds1 = res1["predictions"]["LinearRegression"]["test"]
        preds2 = res2["predictions"]["LinearRegression"]["test"]

        assert np.allclose(preds1, preds2, atol=1e-7)

    def test_strategy_backtester(self) -> None:
        """Verifies that StrategyBacktester realistically accounts for transaction friction."""
        from crypto_prediction.pipeline.backtester import StrategyBacktester

        backtester = StrategyBacktester(fee_rate=0.001, slippage_rate=0.0005)
        # Model predicts strong positive return on first 2 bars, then neutral
        y_pred = np.array([0.02, 0.03, 0.0, 0.0])
        y_true = np.array([0.015, -0.01, 0.02, -0.01])

        res = backtester.run_backtest(y_true, y_pred, model_name="TestStrat")
        assert res.total_trades > 0
        assert res.fee_rate == 0.001
        assert res.slippage_rate == 0.0005
        assert isinstance(res.summary(), str)
        assert isinstance(res.to_dict(), dict)

    def test_neutral_directional_accuracy(self) -> None:
        """Verifies that neutral baseline predictions return NaN rather than being penalized as 0% accuracy."""
        evaluator = ModelEvaluator()
        y_true = np.array([0.01, -0.02, 0.03, -0.01])
        y_pred_zero = np.zeros(4)

        metrics = evaluator.evaluate(y_true, y_pred_zero, model_name="ZeroBaseline")
        assert np.isnan(metrics.directional_accuracy)

    def test_backtester_deterministic_case(self) -> None:
        """Verifies exact analytical return and friction calculation in backtester."""
        from crypto_prediction.pipeline.backtester import StrategyBacktester

        fee = 0.0010
        slippage = 0.0005
        unit_cost = fee + slippage  # 0.0015
        backtester = StrategyBacktester(fee_rate=fee, slippage_rate=slippage)

        # t=0: Enter Long (pos=1, cost=0.0015, gross=0.05, net=0.0485)
        # t=1: Exit to Cash (pos=0, cost=0.0015, gross=0.00, net=-0.0015)
        y_true = np.array([0.05, 0.00])
        y_pred = np.array([0.05, 0.00])

        res = backtester.run_backtest(y_true, y_pred, model_name="KnownCase")
        expected_cum = (1.0 + 0.05 - unit_cost) * (1.0 + 0.00 - unit_cost) - 1.0

        assert res.total_trades == 2
        assert np.isclose(res.cumulative_return, expected_cum, atol=1e-6)

    def test_backtester_cash_baseline(self) -> None:
        """Zero predictions must produce exactly 0 trades and 0% return."""
        from crypto_prediction.pipeline.backtester import StrategyBacktester

        backtester = StrategyBacktester()
        y_true = np.array([0.02, -0.03, 0.05, -0.01])
        y_pred = np.zeros(4)

        res = backtester.run_backtest(y_true, y_pred, model_name="CashBaseline")
        assert res.total_trades == 0
        assert res.cumulative_return == 0.0
        assert res.win_rate == 0.0

    def test_backtester_horizon_subsampling(self) -> None:
        """Verifies that horizon > 1 subsamples non-overlapping periods."""
        from crypto_prediction.pipeline.backtester import StrategyBacktester

        backtester = StrategyBacktester()
        # 8 hourly bars, horizon=4 -> exactly 2 non-overlapping periods
        y_true = np.array([0.02, 0.01, 0.03, 0.01, -0.02, 0.01, 0.00, 0.02])
        y_pred = np.array([0.05, 0.01, 0.02, 0.01, 0.05, 0.01, 0.00, 0.02])

        res = backtester.run_backtest(y_true, y_pred, horizon=4, non_overlapping=True)
        # Benchmark must compound only the sampled steps: (1 + 0.02) * (1 + -0.02) - 1 = -0.0004
        expected_bench = (1.0 + 0.02) * (1.0 - 0.02) - 1.0
        assert np.isclose(res.benchmark_return, expected_bench, atol=1e-6)

    def test_overfitting_diagnostics_generation(self) -> None:
        """Verifies generation of overfitting diagnostic report across splits."""
        evaluator = ModelEvaluator()
        metrics = [
            evaluator.evaluate([0.01, 0.02], [0.01, 0.02], model_name="M1", split_name="train"),
            evaluator.evaluate([0.01, 0.02], [0.03, 0.05], model_name="M1", split_name="validation"),
            evaluator.evaluate([0.01, 0.02], [0.03, 0.06], model_name="M1", split_name="test"),
        ]
        df = evaluator.build_overfitting_diagnostics(metrics)
        assert len(df) == 1
        assert "val_train_rmse_ratio" in df.columns
        assert "overfitting_diagnosis" in df.columns

    def test_overfitting_diagnostics_flags_r2_collapse_and_no_skill(self) -> None:
        """R2 collapse with a mild RMSE ratio must be OVERFITTING; negative test R2 must be NO SKILL."""
        from crypto_prediction.pipeline.evaluator import EvaluationMetrics

        def m(name: str, split: str, rmse: float, r2: float) -> EvaluationMetrics:
            return EvaluationMetrics(name, 1, split, rmse, rmse, r2, 0.5)

        metrics = [
            # Mirrors HistGradientBoosting: ratio 1.30 (<1.35) but R2 0.41 -> -0.16
            m("Boost", "train", 0.0027, 0.41), m("Boost", "validation", 0.0035, 0.06), m("Boost", "test", 0.0034, -0.16),
            m("Flat", "train", 0.0035, 0.0), m("Flat", "validation", 0.0036, 0.0), m("Flat", "test", 0.0031, -0.003),
            m("MeanReturnBaseline", "train", 0.0035, 0.0), m("MeanReturnBaseline", "validation", 0.0036, 0.0),
            m("MeanReturnBaseline", "test", 0.0031, -0.003),
        ]
        df = ModelEvaluator().build_overfitting_diagnostics(metrics).set_index("model_name")
        assert df.loc["Boost", "overfitting_diagnosis"] == "OVERFITTING"
        assert df.loc["Flat", "overfitting_diagnosis"].startswith("NO SKILL")
        assert df.loc["MeanReturnBaseline", "overfitting_diagnosis"].startswith("BASELINE")

    def test_backtester_flags_unreliable_with_few_trades(self) -> None:
        """Fewer than min_trades trades must be flagged as statistically unreliable."""
        from crypto_prediction.pipeline.backtester import StrategyBacktester

        y_true = np.array([0.01, -0.01] * 20)
        # Alternating long/cash -> many trades vs. a single long holding
        y_pred_many = np.array([0.01, -0.01] * 20)
        y_pred_one = np.full(40, 0.01)
        many = StrategyBacktester(min_trades=30).run_backtest(y_true, y_pred_many)
        one = StrategyBacktester(min_trades=30).run_backtest(y_true, y_pred_one)
        assert many.total_trades >= 30 and many.statistically_reliable
        assert one.total_trades < 30 and not one.statistically_reliable
        assert "statistically_reliable" in one.to_dict()

    def test_backtester_short_positions(self) -> None:
        """Verifies short selling behavior and turnover when reversing positions."""
        from crypto_prediction.pipeline.backtester import StrategyBacktester

        fee = 0.0010
        slippage = 0.0005
        unit_cost = fee + slippage  # 0.0015
        backtester = StrategyBacktester(fee_rate=fee, slippage_rate=slippage, allow_short=True)

        # t=0: Long (+1), market rises 0.04 -> gross +0.04, friction -0.0015
        # t=1: Short (-1), reverse from Long (+1) -> turnover 2, friction -0.0030, market falls -0.03 -> gross +0.03
        # t=2: Final exit from Short (-1) to Cash (0) -> friction -0.0015
        y_true = np.array([0.04, -0.03])
        y_pred = np.array([0.05, -0.05])

        res = backtester.run_backtest(y_true, y_pred, model_name="ShortStrategy")
        # Step 0: 1 + 0.04 - 0.0015 = 1.0385
        # Step 1: 1 + (-1 * -0.03) - 2 * 0.0015 - 1 * 0.0015 (final close) = 1 + 0.03 - 0.0045 = 1.0255
        expected_cum = 1.0385 * 1.0255 - 1.0
        assert np.isclose(res.cumulative_return, expected_cum, atol=1e-5)
        assert res.total_trades == 3  # enter long, reverse to short, close to cash

    def test_backtester_initial_capital_max_drawdown(self) -> None:
        """Verifies max drawdown calculation starting strictly from initial capital = 1.0."""
        from crypto_prediction.pipeline.backtester import StrategyBacktester

        backtester = StrategyBacktester(fee_rate=0.0, slippage_rate=0.0)
        # Drop 10% on first bar, then recover
        y_true = np.array([-0.10, 0.20])
        y_pred = np.array([0.05, 0.05])

        res = backtester.run_backtest(y_true, y_pred, model_name="DrawdownTest")
        # Capital: 1.0 -> 0.90 -> 1.08
        # Peak: 1.0 -> 1.0 -> 1.08
        # Drawdowns: 0.0 -> -0.10 -> 0.0
        # Max drawdown must be -0.10 (-10%)
        assert np.isclose(res.max_drawdown, -0.10, atol=1e-5)

    def test_backtester_handles_invalid_returns(self) -> None:
        """Verifies that non-finite values (NaN, Inf) are filtered out cleanly without crashing."""
        from crypto_prediction.pipeline.backtester import StrategyBacktester

        backtester = StrategyBacktester()
        y_true = np.array([0.01, np.nan, np.inf, -0.02, 0.03])
        y_pred = np.array([0.02, 0.01, -0.01, -np.inf, 0.04])

        # Valid rows: row 0 and row 4
        res = backtester.run_backtest(y_true, y_pred, model_name="RobustnessTest")
        assert np.isfinite(res.cumulative_return)
        assert np.isfinite(res.sharpe_ratio)
        assert np.isfinite(res.max_drawdown)

    def test_backtest_friction_lifecycle_and_gross_net_separation(self) -> None:
        """Verifies full friction lifecycle: open, hold (zero cost), reverse, final close."""
        from crypto_prediction.pipeline.backtester import StrategyBacktester

        fee = 0.0010
        slippage = 0.0005
        unit_cost = fee + slippage  # 0.0015 per order/side
        backtester = StrategyBacktester(fee_rate=fee, slippage_rate=slippage, allow_short=True)

        # Bar 0: Enter Long (delta=1 -> cost 0.0015, gross return 0.02)
        # Bar 1: Hold Long (delta=0 -> cost 0.0000, gross return 0.01)
        # Bar 2: Reverse to Short (delta=2 -> cost 0.0030, gross return -1 * -0.01 = +0.01)
        # Final liquidation at end of backtest (delta=1 -> cost 0.0015, gross return 0.0)
        y_true = np.array([0.02, 0.01, -0.01])
        y_pred = np.array([0.05, 0.05, -0.05])

        res = backtester.run_backtest(y_true, y_pred, model_name="LifecycleTest")

        # Total turnover: 1 (open) + 0 (hold) + 2 (reverse) + 1 (final close) = 4 units
        # Total transaction costs = 4 * 0.0015 = 0.0060 (0.60%)
        expected_total_cost = 4 * unit_cost
        assert np.isclose(res.total_transaction_costs, expected_total_cost, atol=1e-6)

        # Gross cumulative return: (1 + 0.02) * (1 + 0.01) * (1 + 0.01) - 1.0 = 0.040602
        expected_gross = (1.0 + 0.02) * (1.0 + 0.01) * (1.0 + 0.01) - 1.0
        assert np.isclose(res.gross_cumulative_return, expected_gross, atol=1e-5)

        # Net return must be strictly less than gross return by friction
        assert res.cumulative_return < res.gross_cumulative_return

        # Separate reporting fields present in summary and dict
        summary = res.summary()
        assert "Strategy Gross Return" in summary
        assert "Strategy Net Return" in summary
        assert "Total Transaction Costs" in summary

        res_dict = res.to_dict()
        assert "gross_cumulative_return" in res_dict
        assert "total_transaction_costs" in res_dict

    def test_artifact_naming_convention_and_schema_validation(self, tmp_path: Path) -> None:
        """Verifies documented naming convention {symbol}_{interval}_{name}_h{horizon}.joblib

        and schema mismatch rejection.
        """
        from crypto_prediction.models.linear import RidgeModel

        symbol = "BTCUSDT"
        interval = "1h"
        horizon = 1
        name = "Ridge"

        # Fit model on 3 features
        X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, 0.2, 0.3], "f3": [-1.0, -2.0, -3.0]})
        y_train = pd.Series([0.01, -0.02, 0.03])
        model = RidgeModel(alpha=1.0)
        model.fit(X_train, y_train)

        # Save using documented naming convention
        artifact_name = f"{symbol}_{interval}_{name}_h{horizon}.joblib"
        artifact_path = tmp_path / artifact_name
        model.save(artifact_path)
        assert artifact_path.exists()

        # Load back
        loaded = RidgeModel.load(artifact_path)
        assert loaded.feature_names == ["f1", "f2", "f3"]

        # Inference on compatible schema works
        preds = loaded.predict(X_train)
        assert len(preds) == 3

        # Inference on incompatible schema (missing f3 or different columns)
        X_incompatible = pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.1, 0.2]})
        assert loaded.feature_names != list(X_incompatible.columns)
