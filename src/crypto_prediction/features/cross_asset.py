"""Cross-Asset Causal Feature Engineering Pipeline for Cryptocurrency Time-Series.

Constructs:
- Asset-Specific Causal Features (returns, volatility, momentum, volume, morphology, technicals).
- Cross-Asset Features (lead-lag returns, relative volatility ratios, rolling correlations, momentum spreads).
- Volatility-Aware & Regime Normalization (vol-standardized returns, term structure ratios, Parkinson volatility).
- Market-Wide Aggregate Indicators (market equal-weighted return, return dispersion, aggregate volatility).

Strict Anti-Leakage Rules:
- All rolling statistics strictly trailing (center=False).
- No future price, return, or target information in input features.
- All operations vectorized without global or fold-transgressing standardizations.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from crypto_prediction.features.engineer import FeatureEngineer
from crypto_prediction.features.registry import FeatureRegistry
from crypto_prediction.utils.logger import get_logger

logger = get_logger("cross_asset_features")


class CrossAssetFeatureEngineer:
    """Computes strictly causal asset-specific, cross-asset, and market-wide features."""

    def __init__(
        self,
        anchor_symbols: list[str] | None = None,
        return_lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        forecast_horizons: list[int] | None = None,
        primary_horizon: int = 1,
        include_technicals: bool = True,
        include_patterns: bool = False,
        include_market_aggregates: bool = True,
    ) -> None:
        self.anchor_symbols = [s.upper() for s in (anchor_symbols or ["BTCUSDT", "ETHUSDT"])]
        self.return_lags = return_lags or [1, 2, 3, 6, 12, 24, 48]
        self.rolling_windows = rolling_windows or [6, 12, 24, 48, 168]
        self.forecast_horizons = forecast_horizons or [1, 4, 24]
        self.primary_horizon = primary_horizon
        self.include_technicals = include_technicals
        self.include_patterns = include_patterns
        self.include_market_aggregates = include_market_aggregates

        self.base_engineer = FeatureEngineer(
            return_lags=self.return_lags,
            rolling_windows=self.rolling_windows,
            forecast_horizons=self.forecast_horizons,
            primary_horizon=self.primary_horizon,
            include_patterns=self.include_patterns,
        )
        self.registry = FeatureRegistry()
        self.feature_names_: list[str] = []

    def compute_asset_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes all causal asset-specific statistical and morphology features.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Input DataFrame must be indexed by a DatetimeIndex.")

        feat = pd.DataFrame(index=df.index)
        close = df["close"]
        open_p = df["open"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # 1. Base return
        ret_1h = close.pct_change(1)
        feat["return_1h"] = ret_1h

        # 2. Return lags
        for lag in self.return_lags:
            feat[f"return_lag_{lag}"] = close.pct_change(lag)

        # 3. Rolling return volatility and mean (strictly trailing: center=False)
        for w in self.rolling_windows:
            if w <= len(df):
                feat[f"volatility_{w}h"] = ret_1h.rolling(window=w, center=False).std()
                if w in [12, 24, 48]:
                    feat[f"return_mean_{w}h"] = ret_1h.rolling(window=w, center=False).mean()

        # 4. Parkinson (1980) High-Low Volatility Estimator (trailing 24h)
        # sigma_P = sqrt( 1 / (4 * ln(2) * N) * sum( (ln(H/L))^2 ) )
        if len(df) >= 24:
            safe_hl = (high / (low + 1e-9)).clip(lower=1.0)
            log_hl_sq = np.log(safe_hl) ** 2
            parkinson_denom = 4.0 * np.log(2.0)
            feat["volatility_parkinson_24h"] = np.sqrt(
                log_hl_sq.rolling(window=24, center=False).mean() / parkinson_denom
            )

        # 5. Volatility-Aware Ratios
        if "volatility_24h" in feat.columns:
            feat["return_vol_standardized_24h"] = ret_1h / (feat["volatility_24h"] + 1e-9)
        if "volatility_24h" in feat.columns and "volatility_168h" in feat.columns:
            feat["vol_ratio_24h_168h"] = feat["volatility_24h"] / (feat["volatility_168h"] + 1e-9)

        # 6. Momentum (Multi-period cumulative returns)
        for m in [6, 12, 24, 168]:
            if m <= len(df):
                feat[f"momentum_{m}h"] = close.pct_change(m)

        # 7. Volume Dynamics
        feat["volume_change_1h"] = volume.pct_change(1).clip(lower=-1.0, upper=10.0)
        for vw in [12, 24]:
            vol_sma = volume.rolling(window=vw, center=False).mean()
            feat[f"volume_sma_ratio_{vw}"] = (volume / (vol_sma + 1e-9)) - 1.0

        # 8. Candle Morphology (Scale-invariant intra-bar geometry)
        safe_open = open_p.replace(0, np.nan)
        feat["candle_hl_spread"] = (high - low) / (safe_open + 1e-9)
        feat["candle_co_spread"] = (close - open_p) / (safe_open + 1e-9)
        feat["candle_upper_shadow"] = (high - np.maximum(open_p, close)) / (safe_open + 1e-9)
        feat["candle_lower_shadow"] = (np.minimum(open_p, close) - low) / (safe_open + 1e-9)

        # 9. Existing Technical Indicators
        if self.include_technicals:
            # RSI (Wilder's exponential smoothing)
            diff = close.diff()
            gain = diff.clip(lower=0)
            loss = -diff.clip(upper=0)
            for period in [7, 14, 28]:
                avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
                rs = avg_gain / (avg_loss + 1e-9)
                feat[f"rsi_{period}"] = 100.0 - (100.0 / (1.0 + rs))

            # MACD normalized by close price
            ema_fast = close.ewm(span=12, adjust=False).mean()
            ema_slow = close.ewm(span=26, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - macd_signal
            feat["macd_line"] = macd_line / close
            feat["macd_signal"] = macd_signal / close
            feat["macd_hist"] = macd_hist / close

            # Bollinger Bands
            bb_sma = close.rolling(window=20, center=False).mean()
            bb_std = close.rolling(window=20, center=False).std()
            upper_bb = bb_sma + (2.0 * bb_std)
            lower_bb = bb_sma - (2.0 * bb_std)
            feat["bb_percent_b"] = (close - lower_bb) / (upper_bb - lower_bb + 1e-9)
            feat["bb_bandwidth"] = (upper_bb - lower_bb) / (bb_sma + 1e-9)

            # Average True Range (ATR) normalized by close
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.ewm(alpha=1.0 / 14, adjust=False).mean()
            feat["atr_normalized_14"] = atr / close

        # 10. Cyclical Calendar
        hours = df.index.hour
        dows = df.index.dayofweek
        feat["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        feat["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        feat["day_sin"] = np.sin(2 * np.pi * dows / 7.0)
        feat["day_cos"] = np.cos(2 * np.pi * dows / 7.0)

        # 11. Chart Patterns if enabled
        if self.include_patterns and self.base_engineer.pattern_recognizer is not None:
            pattern_df = self.base_engineer.pattern_recognizer.extract_pattern_features(df)
            for p_col in pattern_df.columns:
                feat[p_col] = pattern_df[p_col]

        return feat

    def compute_cross_asset_panel(
        self,
        aligned_datasets: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Computes both asset-specific and cross-asset features across a synchronized panel.

        Args:
            aligned_datasets: Dictionary mapping symbol -> DataFrame with identical DatetimeIndex.

        Returns:
            Dictionary mapping symbol -> complete engineered feature matrix.
        """
        symbols = [s.upper() for s in aligned_datasets.keys()]
        first_idx = aligned_datasets[symbols[0]].index

        # Verify exact alignment
        for sym in symbols[1:]:
            if not aligned_datasets[sym].index.equals(first_idx):
                raise ValueError(
                    f"Datasets must have identical DatetimeIndex for cross-asset feature engineering. "
                    f"Mismatch between {symbols[0]} and {sym}."
                )

        # 1. Compute asset-specific feature matrices
        asset_feats: dict[str, pd.DataFrame] = {}
        returns_wide = pd.DataFrame(index=first_idx)
        vols_24h_wide = pd.DataFrame(index=first_idx)

        for sym in symbols:
            df_sym = aligned_datasets[sym]
            feats_sym = self.compute_asset_specific_features(df_sym)
            asset_feats[sym] = feats_sym
            returns_wide[sym] = feats_sym["return_1h"]
            if "volatility_24h" in feats_sym.columns:
                vols_24h_wide[sym] = feats_sym["volatility_24h"]

        # 2. Market-wide aggregate indicators
        market_feats = pd.DataFrame(index=first_idx)
        if self.include_market_aggregates and len(symbols) > 1:
            market_feats["market_mean_return_1h"] = returns_wide.mean(axis=1)
            market_feats["market_return_dispersion_1h"] = returns_wide.std(axis=1)
            if not vols_24h_wide.empty:
                market_feats["market_mean_volatility_24h"] = vols_24h_wide.mean(axis=1)

        # 3. Compute cross-asset pairwise features
        final_matrices: dict[str, pd.DataFrame] = {}

        for target_sym in symbols:
            feat_out = asset_feats[target_sym].copy()
            target_ret = returns_wide[target_sym]
            target_vol_24h = asset_feats[target_sym].get("volatility_24h")
            target_vol_168h = asset_feats[target_sym].get("volatility_168h")

            # A. Peer Returns and Lags
            for ref_sym in symbols:
                ref_prefix = ref_sym.lower().replace("usdt", "")
                ref_ret = returns_wide[ref_sym]

                # Lagged returns of reference asset (strictly trailing)
                feat_out[f"{ref_prefix}_return_lag_1"] = ref_ret.shift(0)  # already r_{t-1} to r_t
                feat_out[f"{ref_prefix}_return_lag_2"] = ref_ret.shift(1)
                feat_out[f"{ref_prefix}_return_lag_3"] = ref_ret.shift(2)

                # Pairwise interactions between target and reference asset
                if ref_sym != target_sym:
                    # B. Rolling Pearson correlation (24h and 168h)
                    feat_out[f"cross_corr_{ref_prefix}_24h"] = target_ret.rolling(
                        window=24, center=False
                    ).corr(ref_ret)
                    if len(first_idx) >= 168:
                        feat_out[f"cross_corr_{ref_prefix}_168h"] = target_ret.rolling(
                            window=168, center=False
                        ).corr(ref_ret)

                    # C. Relative Volatility Ratio
                    ref_vol_24h = asset_feats[ref_sym].get("volatility_24h")
                    if target_vol_24h is not None and ref_vol_24h is not None:
                        feat_out[f"cross_vol_ratio_{ref_prefix}_24h"] = target_vol_24h / (
                            ref_vol_24h + 1e-9
                        )
                    ref_vol_168h = asset_feats[ref_sym].get("volatility_168h")
                    if target_vol_168h is not None and ref_vol_168h is not None:
                        feat_out[f"cross_vol_ratio_{ref_prefix}_168h"] = target_vol_168h / (
                            ref_vol_168h + 1e-9
                        )

                    # D. Momentum Spreads (Target return minus reference return)
                    feat_out[f"cross_mom_spread_{ref_prefix}_1h"] = target_ret - ref_ret
                    target_mom_24h = asset_feats[target_sym].get("momentum_24h")
                    ref_mom_24h = asset_feats[ref_sym].get("momentum_24h")
                    if target_mom_24h is not None and ref_mom_24h is not None:
                        feat_out[f"cross_mom_spread_{ref_prefix}_24h"] = target_mom_24h - ref_mom_24h

            # E. Attach Market-Wide Indicators
            if not market_feats.empty:
                for col in market_feats.columns:
                    feat_out[col] = market_feats[col]

            final_matrices[target_sym] = feat_out

        # Record feature names from first matrix
        self.feature_names_ = list(final_matrices[symbols[0]].columns)
        logger.info(
            f"Engineered {len(self.feature_names_)} cross-asset features across {len(symbols)} assets."
        )
        return final_matrices

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constructs forward return and price targets for configured horizons."""
        return self.base_engineer.create_targets(df)

    def prepare_dataset(
        self,
        aligned_datasets: dict[str, pd.DataFrame],
        target_horizon: int | None = None,
        drop_warmup: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Prepares complete feature matrices, targets, and metadata for each asset.

        Args:
            aligned_datasets: Synchronized OHLCV datasets across assets.
            target_horizon: Prediction horizon (defaults to self.primary_horizon).
            drop_warmup: Drops warmup and boundary target NaNs.

        Returns:
            Dictionary mapping symbol -> {'X': X, 'y': y, 'meta': metadata}.
        """
        h = target_horizon or self.primary_horizon
        primary_target_col = f"target_return_{h}"

        feature_matrices = self.compute_cross_asset_panel(aligned_datasets)
        output: dict[str, dict[str, Any]] = {}

        # Construct targets and align per asset
        raw_usable_indices = None
        for sym, df_sym in aligned_datasets.items():
            feats = feature_matrices[sym]
            targets = self.create_targets(df_sym)
            combined = pd.concat(
                [feats, targets, df_sym[["open", "high", "low", "close", "volume"]]],
                axis=1,
            )

            if drop_warmup:
                # Find valid rows without NaNs
                valid_mask = combined[feats.columns].notna().all(axis=1) & combined[
                    primary_target_col
                ].notna()
                valid_idx = combined.index[valid_mask]
                if raw_usable_indices is None:
                    raw_usable_indices = set(valid_idx)
                else:
                    raw_usable_indices = raw_usable_indices.intersection(set(valid_idx))

        common_usable_index = sorted(list(raw_usable_indices)) if raw_usable_indices else []

        for sym, df_sym in aligned_datasets.items():
            feats = feature_matrices[sym]
            targets = self.create_targets(df_sym)
            meta_cols = ["close"] + list(targets.columns)

            if drop_warmup:
                X = feats.loc[common_usable_index].copy()
                y = targets.loc[common_usable_index, primary_target_col].copy()
                meta = df_sym.loc[common_usable_index, ["open", "high", "low", "close", "volume"]].copy()
                for c in targets.columns:
                    meta[c] = targets.loc[common_usable_index, c]
            else:
                X = feats.copy()
                y = targets[primary_target_col].copy()
                meta = df_sym[["open", "high", "low", "close", "volume"]].copy()
                for c in targets.columns:
                    meta[c] = targets[c]

            output[sym] = {"X": X, "y": y, "meta": meta}

        logger.info(
            f"Prepared synchronized multi-asset dataset: {len(common_usable_index)} usable observations "
            f"per asset across {len(self.feature_names_)} features. Horizon={h}."
        )
        return output
