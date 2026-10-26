"""Causal feature engineering pipeline for cryptocurrency time-series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from crypto_prediction.features.patterns import PatternRecognizer
from crypto_prediction.features.registry import FeatureRegistry
from crypto_prediction.utils.logger import get_logger

logger = get_logger("feature_engineer")


class FeatureEngineer:
    """Computes strictly causal statistical, technical, calendar, and chart pattern features

    without lookahead bias or future-data leakage. Supports dynamic arbitrary forecasting horizons.
    """

    def __init__(
        self,
        return_lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        rsi_periods: list[int] | None = None,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_window: int = 20,
        bb_std: float = 2.0,
        atr_window: int = 14,
        volume_sma_windows: list[int] | None = None,
        include_cyclical_time: bool = True,
        include_patterns: bool = True,
        forecast_horizons: list[int] | None = None,
        primary_horizon: int = 1,
    ) -> None:
        self.return_lags = return_lags or [1, 2, 3, 6, 12, 24, 48]
        self.rolling_windows = rolling_windows or [6, 12, 24, 48, 168]
        self.rsi_periods = rsi_periods or [7, 14, 28]
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.atr_window = atr_window
        self.volume_sma_windows = volume_sma_windows or [12, 24]
        self.include_cyclical_time = include_cyclical_time
        self.include_patterns = include_patterns
        self.forecast_horizons = forecast_horizons or [1, 4, 24]
        self.primary_horizon = primary_horizon

        # Ensure primary_horizon is tracked
        if self.primary_horizon not in self.forecast_horizons:
            self.forecast_horizons.append(self.primary_horizon)

        self.registry = FeatureRegistry()
        self.pattern_recognizer = PatternRecognizer() if self.include_patterns else None
        self.feature_names_: list[str] = []

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes all causal features for the given OHLCV DataFrame.

        Every feature computed at row t depends solely on observations <= t.

        Args:
            df: Cleaned OHLCV DataFrame with DatetimeIndex.

        Returns:
            DataFrame containing engineered feature columns indexed by time.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Input DataFrame must be indexed by a DatetimeIndex.")

        feat = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # 1. Base 1-period return
        ret_1h = close.pct_change(1)
        feat["return_1h"] = ret_1h

        # 2. Return lags (t-k)
        for lag in self.return_lags:
            col_name = f"return_lag_{lag}"
            # Multi-bar return over past lag periods: (P_t - P_{t-lag}) / P_{t-lag}
            feat[col_name] = close.pct_change(lag)

        # 3. Rolling return statistics (strictly trailing: center=False)
        for w in self.rolling_windows:
            if w <= len(df):
                feat[f"volatility_{w}h"] = ret_1h.rolling(window=w, center=False).std()
                if w in [12, 24, 48]:
                    feat[f"return_mean_{w}h"] = ret_1h.rolling(window=w, center=False).mean()

        # 4. Exponential Moving Average Ratios: Close / EMA - 1
        for period in [12, 26, 50]:
            ema = close.ewm(span=period, adjust=False).mean()
            feat[f"ema_ratio_{period}"] = (close / (ema + 1e-9)) - 1.0

        # 5. Relative Strength Index (RSI) using Wilder's Exponential Smoothing
        diff = close.diff()
        gain = diff.clip(lower=0)
        loss = -diff.clip(upper=0)
        for period in self.rsi_periods:
            avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            feat[f"rsi_{period}"] = 100.0 - (100.0 / (1.0 + rs))

        # 6. MACD (Moving Average Convergence Divergence)
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        # Normalize MACD by close price for scale invariance across different price regimes
        feat["macd_line"] = macd_line / close
        feat["macd_signal"] = macd_signal / close
        feat["macd_hist"] = macd_hist / close

        # 7. Bollinger Bands
        bb_sma = close.rolling(window=self.bb_window, center=False).mean()
        bb_std = close.rolling(window=self.bb_window, center=False).std()
        upper_bb = bb_sma + (self.bb_std * bb_std)
        lower_bb = bb_sma - (self.bb_std * bb_std)

        # %B and Bandwidth
        feat["bb_percent_b"] = (close - lower_bb) / (upper_bb - lower_bb + 1e-9)
        feat["bb_bandwidth"] = (upper_bb - lower_bb) / (bb_sma + 1e-9)

        # 8. Average True Range (ATR)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1.0 / self.atr_window, adjust=False).mean()
        feat[f"atr_normalized_{self.atr_window}"] = atr / close

        # 9. Volume Features
        feat["volume_change_1h"] = volume.pct_change(1).clip(lower=-1.0, upper=10.0)
        for vw in self.volume_sma_windows:
            vol_sma = volume.rolling(window=vw, center=False).mean()
            feat[f"volume_sma_ratio_{vw}"] = (volume / (vol_sma + 1e-9)) - 1.0

        # 10. Cyclical Calendar Features
        if self.include_cyclical_time:
            hours = df.index.hour
            dows = df.index.dayofweek
            feat["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            feat["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
            feat["day_sin"] = np.sin(2 * np.pi * dows / 7.0)
            feat["day_cos"] = np.cos(2 * np.pi * dows / 7.0)

        # 11. Dynamic Horizon-Tailored Features (when horizon is custom/arbitrary, e.g. 17h)
        h = self.primary_horizon
        if h > 4 and h not in self.return_lags:
            feat[f"return_lag_{h}"] = close.pct_change(h)
            feat[f"volatility_{h}h"] = ret_1h.rolling(window=max(6, h), center=False).std()
            feat[f"volatility_scaled_{h}h"] = ret_1h.rolling(window=24, center=False).std() * np.sqrt(h)
            ema_h = close.ewm(span=max(12, h * 2), adjust=False).mean()
            feat[f"ema_ratio_{h}"] = (close / (ema_h + 1e-9)) - 1.0

        # 12. Technical Chart and Candlestick Pattern Features
        if self.include_patterns and self.pattern_recognizer is not None:
            pattern_df = self.pattern_recognizer.extract_pattern_features(df)
            for p_col in pattern_df.columns:
                feat[p_col] = pattern_df[p_col]

        self.feature_names_ = list(feat.columns)
        return feat

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constructs forward targets for all configured forecasting horizons.

        Target at time t is strictly the forward return or forward price over (t, t+h]:
        - target_return_{h} = (Close_{t+h} - Close_t) / Close_t
        - target_price_{h}  = Close_{t+h}

        Args:
            df: Cleaned OHLCV DataFrame.

        Returns:
            DataFrame with target columns.
        """
        targets = pd.DataFrame(index=df.index)
        close = df["close"]

        for h in self.forecast_horizons:
            # Shift backwards into the future
            future_close = close.shift(-h)
            targets[f"target_return_{h}"] = (future_close - close) / close
            targets[f"target_price_{h}"] = future_close

        return targets

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        target_horizon: int | None = None,
        drop_warmup: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Engineers features and targets, aligns them, and drops warm-up / future NaNs.

        Args:
            df: Cleaned OHLCV DataFrame with DatetimeIndex.
            target_horizon: Horizon to use as primary target (defaults to self.primary_horizon).
            drop_warmup: If True, drops early rows where rolling features are NaN.

        Returns:
            Tuple of:
            - X: DataFrame of engineered input features (no targets, strictly <= t).
            - y: Series of the selected primary target variable.
            - metadata: DataFrame containing 'close' and other horizon targets for backtesting.
        """
        h = target_horizon or self.primary_horizon

        features = self.compute_features(df)
        targets = self.create_targets(df)

        primary_target_col = f"target_return_{h}"

        # Combine for unified alignment
        combined = pd.concat([features, targets, df[["open", "high", "low", "close", "volume"]]], axis=1)

        if drop_warmup:
            # Drop rows with NaN in features (warmup lookback) or target (final h rows)
            combined = combined.dropna(subset=self.feature_names_ + [primary_target_col])

        X = combined[self.feature_names_].copy()
        y = combined[primary_target_col].copy()

        # Metadata retains prices and all targets for downstream simulation
        meta_cols = ["close"] + [col for col in targets.columns]
        metadata = combined[meta_cols].copy()

        logger.info(
            f"Prepared dataset: {len(X)} rows, {len(X.columns)} features. Target: {primary_target_col} (horizon={h})."
        )
        return X, y, metadata
