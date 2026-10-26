"""Model evaluation metrics and baseline benchmarking engine."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class EvaluationMetrics:
    """Metrics container for a single model on a specific partition."""

    model_name: str
    horizon: int
    split_name: str
    mae: float
    rmse: float
    r2: float
    directional_accuracy: float
    active_directional_bets: int = 0
    neutral_bets: int = 0
    price_mae: float | None = None
    price_rmse: float | None = None

    def to_dict(self) -> dict[str, float | str | int | None]:
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "split_name": self.split_name,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "directional_accuracy": self.directional_accuracy,
            "active_directional_bets": self.active_directional_bets,
            "neutral_bets": self.neutral_bets,
            "price_mae": self.price_mae,
            "price_rmse": self.price_rmse,
        }


class ModelEvaluator:
    """Computes time-series evaluation metrics and benchmarks models against baselines."""

    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates directional accuracy: fraction of time active directional predictions match true market direction.

        Neutral predictions (pred == 0.0, e.g. ZeroReturnBaseline) do not take directional bets and are
        not penalized as incorrect; if a model produces exclusively neutral calls, NaN is returned.
        """
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        # Signs: +1, -1, 0
        sign_true = np.sign(y_t)
        sign_pred = np.sign(y_p)

        # Active bets where model predicts a direction
        active_bets = (sign_pred != 0)
        if not np.any(active_bets):
            return float("nan")

        correct = (sign_true[active_bets] == sign_pred[active_bets])
        return float(np.mean(correct))

    def evaluate(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        model_name: str,
        horizon: int = 1,
        split_name: str = "test",
        current_close: np.ndarray | pd.Series | None = None,
        true_future_price: np.ndarray | pd.Series | None = None,
    ) -> EvaluationMetrics:
        """Computes comprehensive forecasting performance metrics.

        Args:
            y_true: Ground truth target returns.
            y_pred: Predicted target returns.
            model_name: Name of evaluated estimator.
            horizon: Forecast horizon (bars ahead).
            split_name: Partition name ('train', 'validation', 'test').
            current_close: Optional base close prices at time t.
            true_future_price: Optional actual future prices at time t+h.

        Returns:
            EvaluationMetrics object.
        """
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)

        # Filter NaNs if any
        valid = (~np.isnan(y_t)) & (~np.isnan(y_p))
        y_t = y_t[valid]
        y_p = y_p[valid]

        mae = float(mean_absolute_error(y_t, y_p))
        rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
        # R2 on returns is often close to zero or negative in efficient financial markets
        try:
            r2 = float(r2_score(y_t, y_p))
        except Exception:
            r2 = float("nan")

        dir_acc = self.calculate_directional_accuracy(y_t, y_p)
        active_count = int(np.count_nonzero(y_p != 0))
        neutral_count = int(len(y_p) - active_count)

        price_mae = None
        price_rmse = None
        if current_close is not None and true_future_price is not None:
            c_close = np.asarray(current_close)[valid]
            f_price = np.asarray(true_future_price)[valid]
            pred_price = c_close * (1.0 + y_p)
            price_mae = float(mean_absolute_error(f_price, pred_price))
            price_rmse = float(np.sqrt(mean_squared_error(f_price, pred_price)))

        return EvaluationMetrics(
            model_name=model_name,
            horizon=horizon,
            split_name=split_name,
            mae=mae,
            rmse=rmse,
            r2=r2,
            directional_accuracy=dir_acc,
            active_directional_bets=active_count,
            neutral_bets=neutral_count,
            price_mae=price_mae,
            price_rmse=price_rmse,
        )

    @staticmethod
    def build_comparison_table(metrics_list: list[EvaluationMetrics]) -> pd.DataFrame:
        """Converts a list of EvaluationMetrics into a styled comparative DataFrame."""
        rows = [m.to_dict() for m in metrics_list]
        df = pd.DataFrame(rows)
        return df

    # Thresholds for the overfitting diagnosis
    RMSE_RATIO_OVERFIT = 1.35
    RMSE_RATIO_SHIFT = 0.85
    R2_GAP_OVERFIT = 0.15  # train R2 minus held-out R2

    @staticmethod
    def build_overfitting_diagnostics(metrics_list: list[EvaluationMetrics]) -> pd.DataFrame:
        """Diagnoses overfitting and lack of skill from RMSE ratios *and* R2 gaps.

        RMSE ratio alone is misleading on returns: a model can collapse from train R2 = +0.41 to
        test R2 = -0.16 while val/train RMSE stays under 1.35. The R2 gap catches that case, and
        the NO SKILL label flags models that do not beat the mean-return baseline out of sample.
        """
        records: dict[str, dict[str, float]] = {}
        for m in metrics_list:
            rec = records.setdefault(m.model_name, {})
            rec[f"{m.split_name}_rmse"] = m.rmse
            rec[f"{m.split_name}_mae"] = m.mae
            rec[f"{m.split_name}_r2"] = m.r2

        rows = []
        for name, data in records.items():
            train_rmse = data.get("train_rmse", np.nan)
            val_rmse = data.get("validation_rmse", np.nan)
            test_rmse = data.get("test_rmse", np.nan)
            train_r2 = data.get("train_r2", np.nan)
            val_r2 = data.get("validation_r2", np.nan)
            test_r2 = data.get("test_r2", np.nan)

            ratio = (val_rmse / train_rmse) if (train_rmse and not np.isnan(train_rmse)) else np.nan

            # Worst train -> held-out R2 drop (ignoring NaNs)
            gaps = [train_r2 - r for r in (val_r2, test_r2) if not np.isnan(r) and not np.isnan(train_r2)]
            r2_gap = max(gaps) if gaps else np.nan

            # Skill is judged on the final out-of-sample split (test), falling back to validation.
            final_r2 = test_r2 if not np.isnan(test_r2) else val_r2
            no_skill = (not np.isnan(final_r2)) and final_r2 <= 0.0

            if name.endswith("Baseline"):
                status = "BASELINE (reference)"
            elif (not np.isnan(ratio) and ratio > ModelEvaluator.RMSE_RATIO_OVERFIT) or (
                not np.isnan(r2_gap) and r2_gap > ModelEvaluator.R2_GAP_OVERFIT
            ):
                status = "OVERFITTING"
            elif no_skill:
                status = "NO SKILL (does not beat mean baseline)"
            elif not np.isnan(ratio) and ratio < ModelEvaluator.RMSE_RATIO_SHIFT:
                status = "UNDERFITTING / DISTRIBUTION SHIFT"
            else:
                status = "HEALTHY"

            rows.append({
                "model_name": name,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "test_rmse": test_rmse,
                "val_train_rmse_ratio": ratio,
                "train_r2": train_r2,
                "val_r2": val_r2,
                "test_r2": test_r2,
                "train_heldout_r2_gap": r2_gap,
                "overfitting_diagnosis": status,
            })

        return pd.DataFrame(rows)
