"""Automated data leakage detection and temporal audit guard."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from crypto_prediction.features.engineer import FeatureEngineer
from crypto_prediction.utils.logger import get_logger

logger = get_logger("leakage_guard")


@dataclass
class LeakageAuditReport:
    """Detailed audit results verifying temporal causality and zero leakage."""

    passed_all: bool
    perturbation_test_passed: bool
    target_isolation_passed: bool
    monotonic_timestamps_passed: bool
    purging_gap_passed: bool
    max_perturbation_error: float
    details: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.passed_all else "FAILED"
        lines = [
            f"=== Temporal Data Leakage Audit [{status}] ===",
            f"1. Future Perturbation Invariance: {'PASSED' if self.perturbation_test_passed else 'FAILED'} (Max Error: {self.max_perturbation_error:.2e})",
            f"2. Target Isolation from Features: {'PASSED' if self.target_isolation_passed else 'FAILED'}",
            f"3. Monotonic Timestamp Sequence:   {'PASSED' if self.monotonic_timestamps_passed else 'FAILED'}",
            f"4. Purging & Horizon Separation:  {'PASSED' if self.purging_gap_passed else 'FAILED'}",
        ]
        for msg in self.details:
            lines.append(f"  * {msg}")
        return "\n".join(lines)


class LeakageGuard:
    """Verifies that features are mathematically causal and immune to future-data leakage."""

    @staticmethod
    def run_perturbation_test(
        df_raw: pd.DataFrame,
        feature_engineer: FeatureEngineer,
        cutoff_idx: int | None = None,
        perturbation_factor: float = 2.5,
    ) -> tuple[bool, float, str]:
        """Modifies future prices at index > cutoff_idx and asserts features at index <= cutoff_idx
        remain strictly invariant.

        Args:
            df_raw: Cleaned OHLCV DataFrame with DatetimeIndex.
            feature_engineer: FeatureEngineer instance.
            cutoff_idx: Optional integer row boundary. Defaults to ~70% of dataset length.
            perturbation_factor: Multiplier to heavily distort future prices.

        Returns:
            Tuple of (passed: bool, max_abs_diff: float, detail: str).
        """
        if len(df_raw) < 50:
            raise ValueError(f"DataFrame too short ({len(df_raw)}) for perturbation test.")

        actual_cutoff = cutoff_idx if cutoff_idx is not None else int(len(df_raw) * 0.70)
        if len(df_raw) <= actual_cutoff + 5:
            actual_cutoff = len(df_raw) - 10

        # 1. Compute features on unperturbed data
        X_orig = feature_engineer.compute_features(df_raw)

        # 2. Perturb future rows strictly after actual_cutoff
        df_perturbed = df_raw.copy()
        future_mask = np.arange(len(df_perturbed)) > actual_cutoff

        # Shock close, open, high, low in the future
        for col in ["open", "high", "low", "close"]:
            df_perturbed.loc[future_mask, col] = df_perturbed.loc[future_mask, col] * perturbation_factor
        df_perturbed.loc[future_mask, "volume"] = df_perturbed.loc[future_mask, "volume"] * 10.0

        # High >= Max(Open, Close), Low <= Min(Open, Close)
        df_perturbed["high"] = df_perturbed[["high", "open", "close"]].max(axis=1)
        df_perturbed["low"] = df_perturbed[["low", "open", "close"]].min(axis=1)

        # 3. Compute features on perturbed data
        X_perturbed = feature_engineer.compute_features(df_perturbed)

        # 4. Compare feature values at rows <= actual_cutoff
        # Exclude initial warm-up NaNs from both
        sub_orig = X_orig.iloc[: actual_cutoff + 1]
        sub_pert = X_perturbed.iloc[: actual_cutoff + 1]

        valid_mask = (~sub_orig.isna()) & (~sub_pert.isna())
        diffs = (sub_orig.to_numpy()[valid_mask.to_numpy()] - sub_pert.to_numpy()[valid_mask.to_numpy()])
        max_abs_diff = float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0.0

        # Tolerant only to floating point representation roundoff (< 1e-10)
        passed = max_abs_diff < 1e-10
        detail = (
            f"Perturbation test: max absolute difference at rows <= {actual_cutoff} is {max_abs_diff:.2e}. "
            f"Status: {'Invariance confirmed (No lookahead)' if passed else 'LEAKAGE DETECTED!'}"
        )
        return passed, max_abs_diff, detail

    @staticmethod
    def verify_target_isolation(feature_columns: list[str]) -> tuple[bool, str]:
        """Verifies that no target, future, or forward-looking columns leak into input features."""
        forbidden_substrings = ["target", "future", "ahead", "shift_-", "lead"]
        leaked = [
            col
            for col in feature_columns
            if any(sub in col.lower() for sub in forbidden_substrings)
        ]
        if leaked:
            return False, f"Found target-like or future-derived columns in model features: {leaked}"
        return True, "Feature columns are cleanly isolated from all target variables."

    @staticmethod
    def audit_splits(
        train_index: pd.DatetimeIndex,
        val_index: pd.DatetimeIndex,
        test_index: pd.DatetimeIndex,
        required_purge_bars: int = 1,
        full_index: pd.DatetimeIndex | None = None,
    ) -> tuple[bool, bool, list[str]]:
        """Verifies chronological monotonic separation and purging between splits.

        Args:
            train_index: DatetimeIndex of the training split.
            val_index: DatetimeIndex of the validation split.
            test_index: DatetimeIndex of the test split.
            required_purge_bars: Minimum number of bars that must separate splits.
            full_index: Optional master DatetimeIndex of the complete series to compute exact bar counts.

        Returns:
            Tuple of (monotonic_passed: bool, purging_passed: bool, issues: list[str]).
        """
        issues: list[str] = []
        monotonic_passed = True
        purging_passed = True

        # 1. Monotonic increasing within splits
        if not train_index.is_monotonic_increasing:
            issues.append("Train index is not monotonically increasing.")
            monotonic_passed = False
        if not val_index.is_monotonic_increasing:
            issues.append("Validation index is not monotonically increasing.")
            monotonic_passed = False
        if not test_index.is_monotonic_increasing:
            issues.append("Test index is not monotonically increasing.")
            monotonic_passed = False

        # 2. Strict chronological order across splits
        if len(train_index) > 0 and len(val_index) > 0:
            if train_index[-1] >= val_index[0]:
                issues.append(
                    f"Train maximum timestamp ({train_index[-1]}) overlaps or exceeds Validation start ({val_index[0]})."
                )
                monotonic_passed = False

        if len(val_index) > 0 and len(test_index) > 0:
            if val_index[-1] >= test_index[0]:
                issues.append(
                    f"Validation maximum timestamp ({val_index[-1]}) overlaps or exceeds Test start ({test_index[0]})."
                )
                monotonic_passed = False

        # 3. Purging gap verification
        if required_purge_bars > 0:
            if len(train_index) > 0 and len(val_index) > 0:
                if full_index is not None:
                    train_end_loc = full_index.get_loc(train_index[-1])
                    val_start_loc = full_index.get_loc(val_index[0])
                    gap_train_val = val_start_loc - train_end_loc - 1
                else:
                    # Estimate based on median interval in train_index
                    if len(train_index) > 1:
                        freq_est = pd.Series(train_index).diff().median()
                        gap_train_val = int((val_index[0] - train_index[-1]) / freq_est) - 1
                    else:
                        gap_train_val = 0

                if gap_train_val < required_purge_bars:
                    purging_passed = False
                    issues.append(
                        f"Purging gap between Train and Validation is {gap_train_val} bars, "
                        f"which is less than required {required_purge_bars} bars."
                    )

            if len(val_index) > 0 and len(test_index) > 0:
                if full_index is not None:
                    val_end_loc = full_index.get_loc(val_index[-1])
                    test_start_loc = full_index.get_loc(test_index[0])
                    gap_val_test = test_start_loc - val_end_loc - 1
                else:
                    if len(val_index) > 1:
                        freq_est = pd.Series(val_index).diff().median()
                        gap_val_test = int((test_index[0] - val_index[-1]) / freq_est) - 1
                    else:
                        gap_val_test = 0

                if gap_val_test < required_purge_bars:
                    purging_passed = False
                    issues.append(
                        f"Purging gap between Validation and Test is {gap_val_test} bars, "
                        f"which is less than required {required_purge_bars} bars."
                    )

        return monotonic_passed, purging_passed, issues

    def run_full_audit(
        self,
        df_raw: pd.DataFrame,
        feature_engineer: FeatureEngineer,
        train_index: pd.DatetimeIndex,
        val_index: pd.DatetimeIndex,
        test_index: pd.DatetimeIndex,
        required_purge_bars: int = 1,
        full_index: pd.DatetimeIndex | None = None,
    ) -> LeakageAuditReport:
        """Runs the entire leakage prevention and causality audit suite."""
        pert_passed, max_diff, pert_msg = self.run_perturbation_test(df_raw, feature_engineer)
        features = feature_engineer.compute_features(df_raw)
        target_passed, target_msg = self.verify_target_isolation(list(features.columns))
        monotonic_passed, purging_passed, split_issues = self.audit_splits(
            train_index,
            val_index,
            test_index,
            required_purge_bars=required_purge_bars,
            full_index=full_index if full_index is not None else df_raw.index if isinstance(df_raw.index, pd.DatetimeIndex) else None,
        )

        details = [pert_msg, target_msg] + split_issues

        passed_all = pert_passed and target_passed and monotonic_passed and purging_passed

        return LeakageAuditReport(
            passed_all=passed_all,
            perturbation_test_passed=pert_passed,
            target_isolation_passed=target_passed,
            monotonic_timestamps_passed=monotonic_passed,
            purging_gap_passed=purging_passed,
            max_perturbation_error=max_diff,
            details=details,
        )
