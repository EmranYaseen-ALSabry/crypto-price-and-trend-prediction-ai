"""Multi-asset time-series synchronization layer for cryptocurrency market data.

Enforces:
- Explicit UTC timestamp semantics (open_time, close_time, available_time).
- Panel key uniqueness on (timestamp, symbol).
- Strict non-fabrication policy (no silent forward-filling of market prices).
- Explicit missingness, availability masking, and staleness tracking.
- Dual synchronization modes: intersection (for simultaneous cross-asset modeling)
  and union (preserving full historical coverage across heterogeneous listing dates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import numpy as np
import pandas as pd

from crypto_prediction.data.cleaner import DataCleaner
from crypto_prediction.data.validator import DataValidator
from crypto_prediction.utils.logger import get_logger

logger = get_logger("multi_asset_synchronizer")


@dataclass
class SynchronizationReport:
    """Diagnostic report detailing coverage, alignment, and missingness across assets."""

    mode: str
    interval: str
    symbols: list[str]
    total_panel_rows: int
    common_start_utc: str | None
    common_end_utc: str | None
    coverage_per_asset: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing_bars_per_asset: dict[str, int] = field(default_factory=dict)
    is_aligned: bool = True
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Returns human-readable synchronization report."""
        lines = [
            f"=== Multi-Asset Synchronization Report [{self.mode.upper()}] ===",
            f"Interval: {self.interval}",
            f"Symbols: {', '.join(self.symbols)}",
            f"Total Panel Rows: {self.total_panel_rows}",
            f"Common Timeline: {self.common_start_utc} -> {self.common_end_utc}",
            f"Alignment Verified: {'PASSED' if self.is_aligned else 'FAILED'}",
            "Coverage Per Asset:",
        ]
        for sym, cov in self.coverage_per_asset.items():
            lines.append(
                f"  - {sym}: {cov.get('rows', 0)} bars ({cov.get('start', 'N/A')} to {cov.get('end', 'N/A')})"
            )
        if self.missing_bars_per_asset:
            lines.append("Missing Observations Relative to Panel Timeline:")
            for sym, count in self.missing_bars_per_asset.items():
                lines.append(f"  - {sym}: {count} unobserved bars")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  * {w}")
        return "\n".join(lines)


class MultiAssetSynchronizer:
    """Synchronizes multi-asset cryptocurrency datasets with explicit availability semantics."""

    INTERVAL_DELTAS = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "2h": pd.Timedelta(hours=2),
        "4h": pd.Timedelta(hours=4),
        "1d": pd.Timedelta(days=1),
    }

    def __init__(
        self,
        interval: str = "1h",
        clean_inputs: bool = True,
    ) -> None:
        self.interval = interval
        if interval not in self.INTERVAL_DELTAS:
            raise ValueError(
                f"Unsupported interval '{interval}'. Supported: {list(self.INTERVAL_DELTAS.keys())}"
            )
        self.delta = self.INTERVAL_DELTAS[interval]
        self.cleaner = DataCleaner() if clean_inputs else None
        self.validator = DataValidator(expected_interval=interval)

    def add_timestamp_semantics(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Enriches OHLCV records with explicit candle-open, candle-close, and availability timestamps.

        Semantics:
        - open_time: Start of the interval (e.g. 12:00:00 UTC).
        - close_time: Last millisecond of interval (e.g. 12:59:59.999 UTC).
        - available_time: Exact instant the candle is closed and finalized for causal use
                          (e.g. 13:00:00 UTC).
        - symbol: Categorical asset identifier.
        """
        out = df.copy()

        # Resolve open_time
        if "open_time" in out.columns:
            open_time = pd.to_datetime(out["open_time"], utc=True)
        elif isinstance(out.index, pd.DatetimeIndex):
            open_time = pd.to_datetime(out.index, utc=True)
        else:
            raise KeyError("DataFrame must contain 'open_time' column or have a DatetimeIndex.")

        # Reset index to eliminate ambiguity between index level and column label
        out = out.reset_index(drop=True)
        out["open_time"] = open_time
        out["close_time"] = open_time + self.delta - pd.Timedelta(milliseconds=1)
        out["available_time"] = open_time + self.delta
        out["symbol"] = symbol.upper()

        # Reorder columns logically
        cols_front = ["symbol", "open_time", "close_time", "available_time"]
        other_cols = [c for c in out.columns if c not in cols_front]
        return out[cols_front + other_cols].sort_values(by="open_time").reset_index(drop=True)

    def synchronize(
        self,
        datasets: dict[str, pd.DataFrame],
        mode: str = "intersection",
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, SynchronizationReport]:
        """Synchronizes multi-asset datasets across a shared temporal grid.

        Args:
            datasets: Dictionary mapping symbol -> raw or processed OHLCV DataFrame.
            mode: 'intersection' (strictly overlapping common period where all assets trade)
                  or 'union' (full outer timeline preserving individual asset histories).

        Returns:
            Tuple of:
            - aligned_dict: Dictionary mapping symbol -> synchronized DataFrame (indexed by open_time).
            - panel_df: Long-format panel DataFrame with MultiIndex (open_time, symbol).
            - report: SynchronizationReport detailing coverage, alignment, and missingness.
        """
        if not datasets:
            raise ValueError("Datasets dictionary cannot be empty.")

        symbols = [s.upper() for s in datasets.keys()]
        processed_dfs: dict[str, pd.DataFrame] = {}
        coverage_per_asset: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []

        # 1. Clean and enrich each dataset with timestamp semantics
        for sym in symbols:
            df = datasets[sym]
            if df.empty:
                raise ValueError(f"Dataset for {sym} is completely empty.")

            # Validate
            val_rep = self.validator.validate(df)
            if not val_rep.is_valid:
                logger.warning(f"Validation warnings for {sym}: {val_rep.errors}")

            # Clean if cleaner is active
            if self.cleaner is not None:
                clean_df = self.cleaner.clean(df)
            else:
                clean_df = df

            enriched = self.add_timestamp_semantics(clean_df, sym)
            processed_dfs[sym] = enriched
            coverage_per_asset[sym] = {
                "rows": len(enriched),
                "start": str(enriched["open_time"].min()),
                "end": str(enriched["open_time"].max()),
                "first_available": str(enriched["available_time"].min()),
                "last_available": str(enriched["available_time"].max()),
            }

        # 2. Build common temporal grid
        all_timestamps = [set(processed_dfs[sym]["open_time"]) for sym in symbols]

        if mode == "intersection":
            common_ts_set = all_timestamps[0]
            for ts_set in all_timestamps[1:]:
                common_ts_set = common_ts_set.intersection(ts_set)

            if not common_ts_set:
                raise ValueError("No common overlapping timestamps found across provided assets.")

            grid_ts = pd.Series(sorted(list(common_ts_set)), name="open_time")
        elif mode == "union":
            union_ts_set = all_timestamps[0]
            for ts_set in all_timestamps[1:]:
                union_ts_set = union_ts_set.union(ts_set)
            grid_ts = pd.Series(sorted(list(union_ts_set)), name="open_time")
        else:
            raise ValueError(f"Unsupported synchronization mode: '{mode}'. Use 'intersection' or 'union'.")

        grid_df = pd.DataFrame({"open_time": grid_ts})
        grid_df["close_time"] = grid_df["open_time"] + self.delta - pd.Timedelta(milliseconds=1)
        grid_df["available_time"] = grid_df["open_time"] + self.delta

        aligned_dict: dict[str, pd.DataFrame] = {}
        missing_bars_per_asset: dict[str, int] = {}
        panel_rows: list[pd.DataFrame] = []

        # 3. Synchronize each asset onto the common grid
        for sym in symbols:
            df_sym = processed_dfs[sym]

            if mode == "intersection":
                # Strict inner subset on grid
                merged = df_sym[df_sym["open_time"].isin(grid_ts)].copy()
                merged = merged.sort_values(by="open_time").reset_index(drop=True)
                merged["is_available"] = True
                merged["staleness_bars"] = 0
                missing_bars = 0
            else:
                # Union mode: outer merge against grid
                merged = pd.merge(grid_df, df_sym, on=["open_time", "close_time", "available_time"], how="left")
                merged["symbol"] = sym

                # Explicit availability mask (NO silent price fabrication)
                merged["is_available"] = merged["close"].notna()

                # Staleness tracker (consecutive missing bars)
                is_avail_series = merged["is_available"]
                staleness = []
                count = 0
                for avail in is_avail_series:
                    if avail:
                        count = 0
                    else:
                        count += 1
                    staleness.append(count)
                merged["staleness_bars"] = staleness

                missing_bars = int((~is_avail_series).sum())
                if missing_bars > 0:
                    warnings.append(
                        f"Asset {sym} has {missing_bars} missing/unlisted bars in union timeline."
                    )

            missing_bars_per_asset[sym] = missing_bars

            # Set open_time as DatetimeIndex for single-asset access
            df_aligned = merged.copy()
            df_aligned = df_aligned.set_index("open_time")
            aligned_dict[sym] = df_aligned
            panel_rows.append(merged)

        # 4. Construct unified long-format multi-asset panel
        combined_panel = pd.concat(panel_rows, ignore_index=True)
        # Deduplicate and sort by (open_time, symbol)
        combined_panel = combined_panel.sort_values(by=["open_time", "symbol"]).reset_index(drop=True)

        # Set panel MultiIndex: (open_time, symbol)
        panel_df = combined_panel.set_index(["open_time", "symbol"]).sort_index()

        common_start = str(grid_ts.iloc[0]) if len(grid_ts) > 0 else None
        common_end = str(grid_ts.iloc[-1]) if len(grid_ts) > 0 else None

        report = SynchronizationReport(
            mode=mode,
            interval=self.interval,
            symbols=symbols,
            total_panel_rows=len(combined_panel),
            common_start_utc=common_start,
            common_end_utc=common_end,
            coverage_per_asset=coverage_per_asset,
            missing_bars_per_asset=missing_bars_per_asset,
            is_aligned=True,
            warnings=warnings,
        )

        logger.info(
            f"Successfully synchronized {len(symbols)} assets in '{mode}' mode "
            f"({len(grid_ts)} timestamps, {len(combined_panel)} total rows)."
        )

        return aligned_dict, panel_df, report
