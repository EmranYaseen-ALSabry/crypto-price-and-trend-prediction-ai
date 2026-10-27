"""Feature registry documenting metadata, mathematical formula, lookback, and causality proofs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureMetadata:
    """Detailed specifications and causality audit for a engineered feature."""

    name: str
    category: str
    formula: str
    lookback_window: int
    description: str
    causality_proof: str


class FeatureRegistry:
    """Registry maintaining metadata and documentation for all engineered features."""

    def __init__(self) -> None:
        self._registry: dict[str, FeatureMetadata] = {}
        self._register_default_features()

    def register(self, metadata: FeatureMetadata) -> None:
        """Registers a new feature definition."""
        self._registry[metadata.name] = metadata

    def get(self, name: str) -> FeatureMetadata | None:
        """Retrieves metadata for a named feature."""
        return self._registry.get(name)

    def all_features(self) -> list[FeatureMetadata]:
        """Returns all registered features."""
        return list(self._registry.values())

    def _register_default_features(self) -> None:
        """Populates the registry with standard technical and statistical features."""
        # Returns and lags
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            self.register(
                FeatureMetadata(
                    name=f"return_lag_{lag}",
                    category="Momentum / Returns",
                    formula=f"(Close_t - Close_{{t-{lag}}}) / Close_{{t-{lag}}}",
                    lookback_window=lag,
                    description=f"Past fractional return over {lag} hourly bars.",
                    causality_proof=f"Computed strictly using Close prices from index t and t-{lag}. No future data accessed.",
                )
            )

        # Rolling Volatility
        for window in [6, 12, 24, 48, 168]:
            self.register(
                FeatureMetadata(
                    name=f"volatility_{window}h",
                    category="Volatility",
                    formula=f"std(returns[t-{window}+1 : t])",
                    lookback_window=window,
                    description=f"Sample standard deviation of 1-hour returns over trailing {window} bars.",
                    causality_proof=f"Trailing rolling window with center=False. Uses observations from [t-{window}+1, t].",
                )
            )

        # Rolling Return Means
        for window in [12, 24, 48]:
            self.register(
                FeatureMetadata(
                    name=f"return_mean_{window}h",
                    category="Trend",
                    formula=f"mean(returns[t-{window}+1 : t])",
                    lookback_window=window,
                    description=f"Trailing mean hourly return over past {window} bars.",
                    causality_proof=f"Trailing window strictly ending at time t.",
                )
            )

        # EMA Ratios
        for period in [12, 26, 50]:
            self.register(
                FeatureMetadata(
                    name=f"ema_ratio_{period}",
                    category="Trend / Moving Average",
                    formula=f"(Close_t / EMA_{period}(Close)[t]) - 1.0",
                    lookback_window=period * 2,
                    description=f"Relative deviation of current close from trailing exponential moving average ({period}).",
                    causality_proof=f"Recursive EMA filter computed forward in time; value at t depends solely on Close_{{<=t}}.",
                )
            )

        # RSI
        for period in [7, 14, 28]:
            self.register(
                FeatureMetadata(
                    name=f"rsi_{period}",
                    category="Oscillator",
                    formula=f"100 - (100 / (1 + RS_{period})) using Wilder smoothing",
                    lookback_window=period * 2,
                    description=f"Relative Strength Index over past {period} periods.",
                    causality_proof=f"Wilder smoothing filter applied chronologically over past price changes up to t.",
                )
            )

        # MACD
        self.register(
            FeatureMetadata(
                name="macd_line",
                category="Momentum",
                formula="EMA_12(Close) - EMA_26(Close)",
                lookback_window=26,
                description="Moving Average Convergence Divergence line.",
                causality_proof="Difference between two backward-looking causal EMAs.",
            )
        )
        self.register(
            FeatureMetadata(
                name="macd_signal",
                category="Momentum",
                formula="EMA_9(macd_line)",
                lookback_window=35,
                description="MACD 9-period signal line.",
                causality_proof="EMA of macd_line, strictly using past values.",
            )
        )
        self.register(
            FeatureMetadata(
                name="macd_hist",
                category="Momentum",
                formula="macd_line - macd_signal",
                lookback_window=35,
                description="MACD histogram oscillator.",
                causality_proof="Linear combination of past causal signals.",
            )
        )

        # Bollinger Bands
        self.register(
            FeatureMetadata(
                name="bb_percent_b",
                category="Volatility / Range",
                formula="(Close_t - LowerBB_t) / (UpperBB_t - LowerBB_t)",
                lookback_window=20,
                description="Bollinger Bands %B position within 20-period 2-std band.",
                causality_proof="Trailing 20-bar SMA and trailing std ending at t.",
            )
        )
        self.register(
            FeatureMetadata(
                name="bb_bandwidth",
                category="Volatility / Range",
                formula="(UpperBB_t - LowerBB_t) / SMA_20(Close)",
                lookback_window=20,
                description="Bollinger Bands normalized width.",
                causality_proof="Computed from trailing 20 bars strictly.",
            )
        )

        # ATR
        self.register(
            FeatureMetadata(
                name="atr_normalized_14",
                category="Volatility",
                formula="ATR_14 / Close_t",
                lookback_window=14,
                description="Average True Range (14 periods) divided by close price.",
                causality_proof="True range uses High_t, Low_t, Close_{t-1}, smoothed recursively.",
            )
        )

        # Volume
        self.register(
            FeatureMetadata(
                name="volume_change_1h",
                category="Volume",
                formula="(Volume_t - Volume_{t-1}) / (Volume_{t-1} + eps)",
                lookback_window=2,
                description="1-hour percentage change in trading volume.",
                causality_proof="Uses only Volume at t and Volume at t-1.",
            )
        )
        self.register(
            FeatureMetadata(
                name="volume_sma_ratio_24",
                category="Volume",
                formula="(Volume_t / SMA_24(Volume)[t]) - 1.0",
                lookback_window=24,
                description="Volume relative to its trailing 24-hour simple moving average.",
                causality_proof="SMA window covers [t-23, t].",
            )
        )

        # Candlestick Morphology (Scale-Invariant)
        self.register(
            FeatureMetadata(
                name="candle_hl_spread",
                category="Morphology",
                formula="(High_t - Low_t) / Open_t",
                lookback_window=1,
                description="Normalized intra-bar high-low range.",
                causality_proof="Bar t is fully closed. Uses Open, High, Low from bar t.",
            )
        )
        self.register(
            FeatureMetadata(
                name="candle_co_spread",
                category="Morphology",
                formula="(Close_t - Open_t) / Open_t",
                lookback_window=1,
                description="Normalized bar body (Close minus Open).",
                causality_proof="Uses Open and Close from finalized bar t.",
            )
        )
        self.register(
            FeatureMetadata(
                name="candle_upper_shadow",
                category="Morphology",
                formula="(High_t - max(Open_t, Close_t)) / Open_t",
                lookback_window=1,
                description="Normalized upper shadow length.",
                causality_proof="Uses High, Open, Close from finalized bar t.",
            )
        )
        self.register(
            FeatureMetadata(
                name="candle_lower_shadow",
                category="Morphology",
                formula="(min(Open_t, Close_t) - Low_t) / Open_t",
                lookback_window=1,
                description="Normalized lower shadow length.",
                causality_proof="Uses Low, Open, Close from finalized bar t.",
            )
        )

        # Volatility-Aware & Regime Normalization
        self.register(
            FeatureMetadata(
                name="volatility_parkinson_24h",
                category="Volatility",
                formula="sqrt(1/(4*ln(2)*24) * sum(ln(High/Low)^2))",
                lookback_window=24,
                description="Parkinson extreme-value volatility estimator over past 24 bars.",
                causality_proof="Uses trailing 24 High and Low bars ending at t.",
            )
        )
        self.register(
            FeatureMetadata(
                name="return_vol_standardized_24h",
                category="Volatility / Normalization",
                formula="return_1h / (volatility_24h + eps)",
                lookback_window=24,
                description="Hourly return standardized by its trailing 24h rolling volatility.",
                causality_proof="Trailing return at t divided by trailing 24h volatility at t.",
            )
        )
        self.register(
            FeatureMetadata(
                name="vol_ratio_24h_168h",
                category="Volatility / Term Structure",
                formula="volatility_24h / (volatility_168h + eps)",
                lookback_window=168,
                description="Ratio of short-term (24h) to long-term (168h) realized volatility.",
                causality_proof="Trailing 24h and 168h windows ending at observation time t.",
            )
        )

        # Cross-Asset Lead-Lag & Relative Relationships
        for lag in [1, 2, 3]:
            self.register(
                FeatureMetadata(
                    name=f"btc_return_lag_{lag}",
                    category="Cross-Asset / Returns",
                    formula=f"return_BTC_{{t-{lag}+1}}",
                    lookback_window=lag,
                    description=f"Bitcoin hourly return lagged by {lag} period(s).",
                    causality_proof=f"Uses Bitcoin closed candle returns strictly from index t-{lag}+1 and earlier.",
                )
            )

        self.register(
            FeatureMetadata(
                name="cross_vol_ratio_btc_24h",
                category="Cross-Asset / Volatility",
                formula="volatility_asset_24h / (volatility_BTC_24h + eps)",
                lookback_window=24,
                description="Asset 24h volatility divided by Bitcoin 24h volatility.",
                causality_proof="Both trailing 24h volatilities end at observation time t.",
            )
        )
        self.register(
            FeatureMetadata(
                name="cross_corr_btc_24h",
                category="Cross-Asset / Correlation",
                formula="corr(returns_asset[t-23:t], returns_BTC[t-23:t])",
                lookback_window=24,
                description="Trailing 24h Pearson correlation of hourly returns between asset and BTC.",
                causality_proof="Trailing window [t-23, t] ending at observation time t.",
            )
        )
        self.register(
            FeatureMetadata(
                name="cross_mom_spread_btc_24h",
                category="Cross-Asset / Momentum",
                formula="momentum_asset_24h - momentum_BTC_24h",
                lookback_window=24,
                description="Asset 24h return minus Bitcoin 24h return.",
                causality_proof="Trailing 24h returns ending at observation time t.",
            )
        )

        # Market-Wide Aggregate Indicators
        self.register(
            FeatureMetadata(
                name="market_mean_return_1h",
                category="Market / Aggregates",
                formula="mean_k(returns_k_1h)",
                lookback_window=1,
                description="Equal-weighted average hourly return across all active crypto assets.",
                causality_proof="Cross-sectional mean of available returns for bar ending at t.",
            )
        )
        self.register(
            FeatureMetadata(
                name="market_return_dispersion_1h",
                category="Market / Aggregates",
                formula="std_k(returns_k_1h)",
                lookback_window=1,
                description="Cross-sectional standard deviation of hourly returns across assets.",
                causality_proof="Cross-sectional standard deviation of returns for bar ending at t.",
            )
        )
        self.register(
            FeatureMetadata(
                name="market_mean_volatility_24h",
                category="Market / Aggregates",
                formula="mean_k(volatility_k_24h)",
                lookback_window=24,
                description="Equal-weighted average trailing 24h volatility across assets.",
                causality_proof="Cross-sectional mean of trailing 24h volatilities at time t.",
            )
        )

        # Cyclical Time
        self.register(
            FeatureMetadata(
                name="hour_sin",
                category="Calendar",
                formula="sin(2 * pi * hour / 24)",
                lookback_window=0,
                description="Sine encoding of hour of day.",
                causality_proof="Derived from bar timestamp known at bar closing time t.",
            )
        )
        self.register(
            FeatureMetadata(
                name="hour_cos",
                category="Calendar",
                formula="cos(2 * pi * hour / 24)",
                lookback_window=0,
                description="Cosine encoding of hour of day.",
                causality_proof="Derived from bar timestamp known at bar closing time t.",
            )
        )
        self.register(
            FeatureMetadata(
                name="day_sin",
                category="Calendar",
                formula="sin(2 * pi * day_of_week / 7)",
                lookback_window=0,
                description="Sine encoding of day of week.",
                causality_proof="Derived from bar timestamp known at bar closing time t.",
            )
        )
        self.register(
            FeatureMetadata(
                name="day_cos",
                category="Calendar",
                formula="cos(2 * pi * day_of_week / 7)",
                lookback_window=0,
                description="Cosine encoding of day of week.",
                causality_proof="Derived from bar timestamp known at bar closing time t.",
            )
        )
