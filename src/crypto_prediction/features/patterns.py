"""Causal Technical Chart Patterns and Candlestick Formation Recognition Engine.

All detections are strictly backward-looking (computed at time t using solely observations <= t).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class DetectedPattern:
    """Represents an identified technical or chart pattern."""

    name: str
    pattern_type: str  # 'candlestick', 'reversal', 'continuation', 'support_resistance'
    signal: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    target_price_delta: float
    description: str


@dataclass
class MarketRegime:
    """Represents current detected market regime, trend direction, and structure."""

    regime: str  # 'STRONG_BULLISH' | 'BULLISH' | 'STRONG_BEARISH' | 'BEARISH' | 'RANGING_SIDEWAYS'
    structure: str  # 'HH_HL (Bullish Expansion)' | 'LH_LL (Bearish Decline)' | 'CHOPPY_CONSOLIDATION'
    ema_alignment: str  # 'BULLISH (20 > 50 > 200)' | 'BEARISH (20 < 50 < 200)' | 'MIXED'
    adx: float
    adx_level: str  # 'STRONG_TREND' | 'MODERATE_TREND' | 'WEAK_SIDEWAYS'
    is_trending: bool
    trend_bias: str  # 'BUY_DIPS_ONLY' | 'SELL_RALLIES_ONLY' | 'RANGE_BOUND'
    ema20_last: float
    ema50_last: float
    ema200_last: float
    swing_points: list[dict[str, Any]]
    description: str


@dataclass
class ConfirmationSignal:
    """Tactical trade execution setup with candlestick confirmation metrics."""

    status: str  # 'CONFIRMED_SELL' | 'CONFIRMED_BUY' | 'PENDING_CONFIRMATION' | 'INVALIDATED_BREAKOUT' | 'INVALIDATED_BREAKDOWN' | 'NEUTRAL'
    setup_type: str  # 'SHORT' | 'LONG' | 'WAIT'
    entry_range: tuple[float, float]
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    trigger_candle_name: str
    rationale: str
    invalidation_price: float


class PatternRecognizer:
    """Identifies classical candlestick and geometric chart patterns causally."""

    def __init__(self, tolerance: float = 0.02) -> None:
        self.tolerance = tolerance

    # =========================================================================
    # 1. Single & Multi-Bar Candlestick Patterns
    # =========================================================================

    @staticmethod
    def detect_pinbars(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> np.ndarray:
        """Detects Pinbar / Hammer (Bullish +1) and Shooting Star (Bearish -1).

        Hammer: Long lower shadow (>= 2x body), tiny upper shadow (<= 0.3x body), close near high.
        Shooting Star: Long upper shadow (>= 2x body), tiny lower shadow (<= 0.3x body), close near low.
        """
        n = len(closes)
        signals = np.zeros(n, dtype=np.float64)

        body = np.abs(closes - opens)
        total_range = highs - lows + 1e-9

        upper_shadow = highs - np.maximum(opens, closes)
        lower_shadow = np.minimum(opens, closes) - lows

        # Hammer / Bullish Pinbar
        is_hammer = (lower_shadow >= 2.0 * body) & (upper_shadow <= 0.3 * body) & (body / total_range < 0.4)
        signals[is_hammer] = 1.0

        # Shooting Star / Bearish Pinbar
        is_star = (upper_shadow >= 2.0 * body) & (lower_shadow <= 0.3 * body) & (body / total_range < 0.4)
        signals[is_star] = -1.0

        return signals

    @staticmethod
    def detect_engulfing(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> np.ndarray:
        """Detects Bullish Engulfing (+1) and Bearish Engulfing (-1).

        Bullish: Bar t-1 is bearish, Bar t is bullish and completely engulfs Bar t-1 body.
        Bearish: Bar t-1 is bullish, Bar t is bearish and completely engulfs Bar t-1 body.
        """
        n = len(closes)
        signals = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            prev_body = closes[i - 1] - opens[i - 1]
            curr_body = closes[i] - opens[i]

            # Bullish Engulfing
            if (
                prev_body < 0
                and curr_body > 0
                and opens[i] <= closes[i - 1]
                and closes[i] >= opens[i - 1]
                and abs(curr_body) > abs(prev_body) * 1.05
            ):
                signals[i] = 1.0

            # Bearish Engulfing
            elif (
                prev_body > 0
                and curr_body < 0
                and opens[i] >= closes[i - 1]
                and closes[i] <= opens[i - 1]
                and abs(curr_body) > abs(prev_body) * 1.05
            ):
                signals[i] = -1.0

        return signals

    @staticmethod
    def detect_doji(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> np.ndarray:
        """Detects Doji indecision candles: Body <= 10% of total range."""
        body = np.abs(closes - opens)
        total_range = highs - lows + 1e-9
        is_doji = (body / total_range) <= 0.10
        return is_doji.astype(np.float64)

    # =========================================================================
    # 2. Dynamic Support & Resistance Zones
    # =========================================================================

    @staticmethod
    def compute_dynamic_support_resistance(
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        window: int = 48,
    ) -> tuple[pd.Series, pd.Series]:
        """Calculates trailing dynamic support and resistance levels.

        Support = Rolling minimum of trailing swing lows.
        Resistance = Rolling maximum of trailing swing highs.
        Strictly trailing (center=False).
        """
        support = lows.rolling(window=window, min_periods=window // 2, center=False).min()
        resistance = highs.rolling(window=window, min_periods=window // 2, center=False).max()
        return support, resistance

    # =========================================================================
    # 3. Geometric Chart Patterns (Double Bottom & Double Top)
    # =========================================================================

    def detect_double_patterns(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        lookback: int = 40,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detects Double Bottoms (+1) and Double Tops (-1) across trailing windows.

        Double Bottom: Two distinct local troughs at similar price level (within tolerance)
                       separated by an intermediate peak (neckline).
        Double Top:    Two distinct local peaks at similar price level separated by a trough.
        """
        n = len(closes)
        double_bottoms = np.zeros(n, dtype=np.float64)
        double_tops = np.zeros(n, dtype=np.float64)

        for i in range(lookback, n):
            window_lows = lows[i - lookback : i + 1]
            window_highs = highs[i - lookback : i + 1]
            window_close = closes[i - lookback : i + 1]

            # Find local extrema within the trailing window
            # Troughs
            trough_candidates = []
            for j in range(2, len(window_lows) - 2):
                if (
                    window_lows[j] <= window_lows[j - 1]
                    and window_lows[j] <= window_lows[j - 2]
                    and window_lows[j] <= window_lows[j + 1]
                    and window_lows[j] <= window_lows[j + 2]
                ):
                    trough_candidates.append((j, window_lows[j]))

            # If at least 2 distinct troughs exist separated by at least 5 bars
            if len(trough_candidates) >= 2:
                for idx1 in range(len(trough_candidates) - 1):
                    for idx2 in range(idx1 + 1, len(trough_candidates)):
                        pos1, val1 = trough_candidates[idx1]
                        pos2, val2 = trough_candidates[idx2]
                        if pos2 - pos1 >= 6 and abs(val1 - val2) / val1 <= self.tolerance:
                            # Verify peak in between (neckline)
                            intermediate_peak = np.max(window_highs[pos1:pos2])
                            if intermediate_peak > val1 * (1.0 + self.tolerance):
                                # If current price is near or breaking the neckline
                                if window_close[-1] >= intermediate_peak * 0.98:
                                    double_bottoms[i] = 1.0

            # Peaks (Double Top)
            peak_candidates = []
            for j in range(2, len(window_highs) - 2):
                if (
                    window_highs[j] >= window_highs[j - 1]
                    and window_highs[j] >= window_highs[j - 2]
                    and window_highs[j] >= window_highs[j + 1]
                    and window_highs[j] >= window_highs[j + 2]
                ):
                    peak_candidates.append((j, window_highs[j]))

            if len(peak_candidates) >= 2:
                for idx1 in range(len(peak_candidates) - 1):
                    for idx2 in range(idx1 + 1, len(peak_candidates)):
                        pos1, val1 = peak_candidates[idx1]
                        pos2, val2 = peak_candidates[idx2]
                        if pos2 - pos1 >= 6 and abs(val1 - val2) / val1 <= self.tolerance:
                            intermediate_trough = np.min(window_lows[pos1:pos2])
                            if intermediate_trough < val1 * (1.0 - self.tolerance):
                                if window_close[-1] <= intermediate_trough * 1.02:
                                    double_tops[i] = 1.0

        return double_bottoms, double_tops

    # =========================================================================
    # 4. Feature Extraction Pipeline
    # =========================================================================

    def extract_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts all causal pattern and price action features from OHLCV data.

        Returns:
            DataFrame containing technical pattern features aligned with df.index.
        """
        opens = df["open"].to_numpy(dtype=np.float64)
        highs = df["high"].to_numpy(dtype=np.float64)
        lows = df["low"].to_numpy(dtype=np.float64)
        closes = df["close"].to_numpy(dtype=np.float64)

        feat = pd.DataFrame(index=df.index)

        # 1. Candlestick patterns
        feat["pattern_pinbar"] = self.detect_pinbars(opens, highs, lows, closes)
        feat["pattern_engulfing"] = self.detect_engulfing(opens, highs, lows, closes)
        feat["pattern_doji"] = self.detect_doji(opens, highs, lows, closes)

        # 2. Dynamic Support & Resistance
        support_48h, resistance_48h = self.compute_dynamic_support_resistance(
            df["high"], df["low"], df["close"], window=48
        )
        support_168h, resistance_168h = self.compute_dynamic_support_resistance(
            df["high"], df["low"], df["close"], window=168
        )

        close_series = df["close"]
        feat["dist_to_support_48h"] = (close_series - support_48h) / (close_series + 1e-9)
        feat["dist_to_resistance_48h"] = (resistance_48h - close_series) / (close_series + 1e-9)
        feat["support_resistance_ratio_48h"] = (close_series - support_48h) / (
            resistance_48h - support_48h + 1e-9
        )

        feat["dist_to_support_168h"] = (close_series - support_168h) / (close_series + 1e-9)
        feat["dist_to_resistance_168h"] = (resistance_168h - close_series) / (close_series + 1e-9)

        # 3. Double Bottom & Double Top Formations
        db, dt = self.detect_double_patterns(closes, highs, lows, lookback=48)
        feat["pattern_double_bottom"] = db
        feat["pattern_double_top"] = dt

        # 4. Pattern Summary Score
        # Bullish patterns (+1) vs Bearish patterns (-1)
        feat["pattern_composite_signal"] = (
            feat["pattern_pinbar"]
            + feat["pattern_engulfing"]
            + feat["pattern_double_bottom"]
            - feat["pattern_double_top"]
        ).clip(lower=-3.0, upper=3.0)

        return feat

    def identify_latest_patterns(self, df: pd.DataFrame, recent_bars: int = 10) -> list[DetectedPattern]:
        """Summarizes human-readable detected patterns in the most recent trading bars."""
        features = self.extract_pattern_features(df)
        recent_feat = features.iloc[-recent_bars:]
        recent_df = df.iloc[-recent_bars:]

        patterns: list[DetectedPattern] = []

        # Check for Pinbars
        pinbar_active = recent_feat[recent_feat["pattern_pinbar"] != 0]
        if not pinbar_active.empty:
            last_pin = pinbar_active.iloc[-1]
            if last_pin["pattern_pinbar"] > 0:
                patterns.append(
                    DetectedPattern(
                        name="Hammer / Bullish Pinbar",
                        pattern_type="candlestick",
                        signal="bullish",
                        confidence=0.75,
                        target_price_delta=+0.015,
                        description="Long lower wick rejection indicating strong buyer absorption at local lows.",
                    )
                )
            else:
                patterns.append(
                    DetectedPattern(
                        name="Shooting Star / Bearish Pinbar",
                        pattern_type="candlestick",
                        signal="bearish",
                        confidence=0.75,
                        target_price_delta=-0.015,
                        description="Long upper wick rejection indicating seller overhead supply.",
                    )
                )

        # Check for Engulfing
        engulfing_active = recent_feat[recent_feat["pattern_engulfing"] != 0]
        if not engulfing_active.empty:
            last_eng = engulfing_active.iloc[-1]
            if last_eng["pattern_engulfing"] > 0:
                patterns.append(
                    DetectedPattern(
                        name="Bullish Engulfing",
                        pattern_type="candlestick",
                        signal="bullish",
                        confidence=0.80,
                        target_price_delta=+0.020,
                        description="Green candle body completely engulfed the preceding red candle body.",
                    )
                )
            else:
                patterns.append(
                    DetectedPattern(
                        name="Bearish Engulfing",
                        pattern_type="candlestick",
                        signal="bearish",
                        confidence=0.80,
                        target_price_delta=-0.020,
                        description="Red candle body completely engulfed the preceding green candle body.",
                    )
                )

        # Check for Double Bottom
        db_active = recent_feat[recent_feat["pattern_double_bottom"] > 0]
        if not db_active.empty:
            patterns.append(
                DetectedPattern(
                    name="Double Bottom (W-Formation)",
                    pattern_type="reversal",
                    signal="bullish",
                    confidence=0.85,
                    target_price_delta=+0.035,
                    description="Two distinct swing troughs retested and defended with neckline breakout momentum.",
                )
            )

        # Check for Double Top
        dt_active = recent_feat[recent_feat["pattern_double_top"] > 0]
        if not dt_active.empty:
            patterns.append(
                DetectedPattern(
                    name="Double Top (M-Formation)",
                    pattern_type="reversal",
                    signal="bearish",
                    confidence=0.85,
                    target_price_delta=-0.035,
                    description="Two distinct swing peaks failed to break resistance, breaking support neckline.",
                )
            )

        # Support / Resistance Proximity
        last_row = features.iloc[-1]
        dist_supp = last_row.get("dist_to_support_48h", 0.0)
        dist_res = last_row.get("dist_to_resistance_48h", 0.0)

        if dist_supp < 0.008:
            patterns.append(
                DetectedPattern(
                    name="Support Zone Confluence",
                    pattern_type="support_resistance",
                    signal="bullish",
                    confidence=0.70,
                    target_price_delta=+0.010,
                    description=f"Price is within {dist_supp*100:.2f}% of key 48h support zone.",
                )
            )
        elif dist_res < 0.008:
            patterns.append(
                DetectedPattern(
                    name="Resistance Zone Ceiling",
                    pattern_type="support_resistance",
                    signal="bearish",
                    confidence=0.70,
                    target_price_delta=-0.010,
                    description=f"Price is within {dist_res*100:.2f}% of key 48h resistance ceiling.",
                )
            )

        return patterns

    # =========================================================================
    # 5. Market Regime & Trend Structure Recognition
    # =========================================================================

    @staticmethod
    def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Computes Wilder's Average Directional Index (ADX) causally."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        alpha = 1.0 / period
        tr_smooth = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm, index=high.index).ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=high.index).ewm(alpha=alpha, adjust=False).mean()

        plus_di = 100.0 * (plus_dm_smooth / (tr_smooth + 1e-9))
        minus_di = 100.0 * (minus_dm_smooth / (tr_smooth + 1e-9))

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        return adx

    @staticmethod
    def detect_market_structure(high: pd.Series, low: pd.Series, window: int = 5) -> tuple[str, list[dict[str, Any]]]:
        """Identifies swing highs/lows and structural flow (HH/HL vs LH/LL)."""
        n = len(high)
        if n < window * 3:
            return "CHOPPY_CONSOLIDATION", []

        swing_points: list[dict[str, Any]] = []
        for i in range(window, n - window):
            h_val = float(high.iloc[i])
            l_val = float(low.iloc[i])
            is_high = h_val == float(high.iloc[i - window : i + window + 1].max())
            is_low = l_val == float(low.iloc[i - window : i + window + 1].min())

            if is_high:
                swing_points.append({"index": high.index[i], "type": "HIGH", "price": h_val, "label": "HIGH"})
            elif is_low:
                swing_points.append({"index": low.index[i], "type": "LOW", "price": l_val, "label": "LOW"})

        highs = [p for p in swing_points if p["type"] == "HIGH"]
        lows = [p for p in swing_points if p["type"] == "LOW"]

        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1]["price"] > highs[-2]["price"]
            hl = lows[-1]["price"] > lows[-2]["price"]
            lh = highs[-1]["price"] < highs[-2]["price"]
            ll = lows[-1]["price"] < lows[-2]["price"]

            highs[-1]["label"] = "HH" if hh else "LH"
            lows[-1]["label"] = "HL" if hl else "LL"

            if hh and hl:
                return "HH_HL (Bullish Expansion)", swing_points
            elif lh and ll:
                return "LH_LL (Bearish Decline)", swing_points
            else:
                return "CHOPPY_CONSOLIDATION", swing_points

        return "CHOPPY_CONSOLIDATION", swing_points

    def detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Determines market regime, trend direction, momentum, and structural flow."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        n = len(close)

        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=min(200, n), adjust=False).mean()

        last_close = float(close.iloc[-1])
        last_e20 = float(ema20.iloc[-1])
        last_e50 = float(ema50.iloc[-1])
        last_e200 = float(ema200.iloc[-1])

        adx_series = self.compute_adx(high, low, close, period=14)
        last_adx = float(adx_series.dropna().iloc[-1]) if not adx_series.dropna().empty else 15.0

        struct_name, swings = self.detect_market_structure(high, low, window=min(5, max(2, n // 30)))

        is_trending = last_adx >= 25.0
        adx_level = "STRONG_TREND" if last_adx >= 25.0 else ("MODERATE_TREND" if last_adx >= 18.0 else "WEAK_SIDEWAYS")

        # Alignment
        if last_e20 > last_e50 > last_e200:
            alignment = "BULLISH (20 > 50 > 200)"
        elif last_e20 < last_e50 < last_e200:
            alignment = "BEARISH (20 < 50 < 200)"
        else:
            alignment = "MIXED"

        # Determine Regime
        if last_e20 > last_e50 and last_close > last_e50 and "Bullish" in struct_name:
            regime = "STRONG_BULLISH" if is_trending else "BULLISH"
            bias = "BUY_DIPS_ONLY"
            desc = f"Bullish structure ({struct_name}) with price above EMA50; favored strategy is buying pullbacks."
        elif last_e20 < last_e50 and last_close < last_e50 and "Bearish" in struct_name:
            regime = "STRONG_BEARISH" if is_trending else "BEARISH"
            bias = "SELL_RALLIES_ONLY"
            desc = f"Bearish structure ({struct_name}) with price below EMA50; favored strategy is shorting rallies."
        elif last_e20 > last_e50 and last_close > last_e50:
            regime = "BULLISH"
            bias = "BUY_DIPS_ONLY"
            desc = "Upward momentum with EMA20 > EMA50; bullish bias."
        elif last_e20 < last_e50 and last_close < last_e50:
            regime = "BEARISH"
            bias = "SELL_RALLIES_ONLY"
            desc = "Downward momentum with EMA20 < EMA50; bearish bias."
        else:
            regime = "RANGING_SIDEWAYS"
            bias = "RANGE_BOUND"
            desc = f"Indecisive market regime with mixed EMA alignment ({alignment}) and ADX={last_adx:.1f}."

        return MarketRegime(
            regime=regime,
            structure=struct_name,
            ema_alignment=alignment,
            adx=round(last_adx, 2),
            adx_level=adx_level,
            is_trending=is_trending,
            trend_bias=bias,
            ema20_last=round(last_e20, 2),
            ema50_last=round(last_e50, 2),
            ema200_last=round(last_e200, 2),
            swing_points=swings,
            description=desc,
        )

    # =========================================================================
    # 6. Conditional Candlestick Confirmation & Tactical Execution
    # =========================================================================

    def evaluate_confirmation_signal(
        self,
        df: pd.DataFrame,
        horizon: int,
        predicted_price: float,
        current_price: float,
        support: float | None = None,
        resistance: float | None = None,
        market_regime: MarketRegime | None = None,
    ) -> ConfirmationSignal:
        """Evaluates conditional candlestick confirmation signals at dynamic support/resistance corridors.

        Args:
            df: OHLCV DataFrame (at least trailing bars).
            horizon: Forward prediction horizon in hours.
            predicted_price: Model forecasted price at t+horizon.
            current_price: Latest market price at time t.
            support: Trailing support level (computed dynamically if None).
            resistance: Trailing resistance level (computed dynamically if None).
            market_regime: Optional market regime object. If None and df is long enough, inferred automatically.

        Returns:
            ConfirmationSignal with status, entry range, SL, TP, R:R, and trigger candle details.
        """
        if support is None or resistance is None:
            supp_s, res_s = self.compute_dynamic_support_resistance(
                df["high"], df["low"], df["close"], window=min(48, len(df))
            )
            if support is None:
                support = float(supp_s.dropna().iloc[-1]) if not supp_s.dropna().empty else current_price * 0.98
            if resistance is None:
                resistance = float(res_s.dropna().iloc[-1]) if not res_s.dropna().empty else current_price * 1.02

        # Automatically infer market regime if not supplied and sufficient history exists
        if market_regime is None and len(df) >= 30:
            try:
                market_regime = self.detect_market_regime(df)
            except Exception:
                market_regime = None

        # ---------------------------------------------------------------------
        # Trend-Filtering & Professional Risk/Reward Reconciliation
        # ---------------------------------------------------------------------
        # Case A: Strong Bearish / Bearish Regime with SELL_RALLIES_ONLY bias
        if market_regime is not None and market_regime.trend_bias == "SELL_RALLIES_ONLY":
            # If the model projects an upward bounce or noise (predicted >= current),
            # buying dips against a macro downtrend is prohibited.
            # The high-probability setup is selling the retest of supply / resistance.
            if predicted_price >= current_price:
                entry_min = float(min(current_price * 1.008, resistance * 0.992))
                entry_max = float(resistance * 1.002)
                entry_range = (entry_min, entry_max)
                stop_loss = round(float(resistance * 1.008), 2)
                take_profit = round(float(support), 2)
                risk = max(1e-6, stop_loss - entry_min)
                reward = max(0.0, entry_min - take_profit)
                rr = round(reward / risk, 2) if reward > 0 else 1.5

                return ConfirmationSignal(
                    status="PENDING_CONFIRMATION",
                    setup_type="SHORT",
                    entry_range=entry_range,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward_ratio=max(1.5, rr),
                    trigger_candle_name="Pullback to Supply Zone (Awaiting Reversal Candle)",
                    rationale=(
                        f"Market regime is {market_regime.regime} (Bias: SELL_RALLIES_ONLY). "
                        f"Counter-trend long setups filtered out to protect capital. "
                        f"High-probability tactical setup: Wait for pullback rally into supply zone "
                        f"[${entry_min:,.2f} - ${entry_max:,.2f}] to enter SHORT targeting support "
                        f"(${take_profit:,.2f}) with favorable R:R."
                    ),
                    invalidation_price=stop_loss,
                )

        # Case B: Strong Bullish / Bullish Regime with BUY_DIPS_ONLY bias
        if market_regime is not None and market_regime.trend_bias == "BUY_DIPS_ONLY":
            # If the model projects a downward dip or noise (predicted <= current),
            # shorting into a macro bull run is prohibited.
            # The high-probability setup is buying the dip at key support / demand.
            if predicted_price <= current_price:
                entry_min = float(support * 0.998)
                entry_max = float(max(current_price * 0.992, support * 1.008))
                entry_range = (entry_min, entry_max)
                stop_loss = round(float(support * 0.992), 2)
                take_profit = round(float(resistance), 2)
                risk = max(1e-6, entry_max - stop_loss)
                reward = max(0.0, take_profit - entry_max)
                rr = round(reward / risk, 2) if reward > 0 else 1.5

                return ConfirmationSignal(
                    status="PENDING_CONFIRMATION",
                    setup_type="LONG",
                    entry_range=entry_range,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward_ratio=max(1.5, rr),
                    trigger_candle_name="Dip to Demand Zone (Awaiting Reversal Candle)",
                    rationale=(
                        f"Market regime is {market_regime.regime} (Bias: BUY_DIPS_ONLY). "
                        f"Counter-trend short setups filtered out to protect capital. "
                        f"High-probability tactical setup: Wait for dip into demand zone "
                        f"[${entry_min:,.2f} - ${entry_max:,.2f}] to enter LONG targeting resistance "
                        f"(${take_profit:,.2f}) with favorable R:R."
                    ),
                    invalidation_price=stop_loss,
                )

        last_candle = df.iloc[-1]
        open_p = float(last_candle["open"])
        close_p = float(last_candle["close"])
        high_p = float(last_candle["high"])
        low_p = float(last_candle["low"])

        body = abs(close_p - open_p)
        total_range = high_p - low_p + 1e-9
        upper_wick = high_p - max(open_p, close_p)
        lower_wick = min(open_p, close_p) - low_p

        # Candlestick pattern checks on latest closed bar
        is_star = (upper_wick >= 2.0 * body) and (lower_wick <= 0.3 * body) and (body / total_range < 0.4)
        is_hammer = (lower_wick >= 2.0 * body) and (upper_wick <= 0.3 * body) and (body / total_range < 0.4)

        is_bearish_engulfing = False
        is_bullish_engulfing = False
        if len(df) >= 2:
            prev_candle = df.iloc[-2]
            prev_open = float(prev_candle["open"])
            prev_close = float(prev_candle["close"])
            prev_body = prev_close - prev_open
            curr_body = close_p - open_p

            if prev_body > 0 and curr_body < 0:
                if open_p >= prev_close and close_p <= prev_open and abs(curr_body) > abs(prev_body) * 1.05:
                    is_bearish_engulfing = True
            elif prev_body < 0 and curr_body > 0:
                if open_p <= prev_close and close_p >= prev_open and abs(curr_body) > abs(prev_body) * 1.05:
                    is_bullish_engulfing = True

        # Case 1: Bearish setup (predicted_price < current_price)
        if predicted_price < current_price:
            # Resistance band: [resistance * 0.995, resistance * 1.015]
            breakout_threshold = resistance * 1.015

            if close_p > breakout_threshold:
                # Strong bullish breakout above resistance
                status = "INVALIDATED_BREAKOUT"
                setup_type = "LONG"
                entry_range = (float(resistance), float(current_price))
                stop_loss = round(float(resistance * 0.995), 2)
                take_profit = round(float(current_price + (current_price - resistance)), 2)
                risk = max(1e-6, current_price - stop_loss)
                reward = max(0.0, take_profit - current_price)
                rr = round(reward / risk, 2)
                trigger_name = "Bullish Breakout Candle"
                inval = round(float(breakout_threshold), 2)
                rationale = (
                    f"Strong bullish body closed above resistance ceiling (${breakout_threshold:,.2f}); "
                    f"bearish mean-reversion invalidated. Shift to breakout momentum long."
                )
            elif is_star or is_bearish_engulfing or (upper_wick >= 2.0 * body and upper_wick > 0):
                status = "CONFIRMED_SELL"
                setup_type = "SHORT"
                entry_range = (float(current_price), float(resistance))
                high_ref = max(high_p, resistance)
                stop_loss = round(float(high_ref + 0.003 * high_ref), 2)
                take_profit = round(float(predicted_price), 2)
                risk = max(1e-6, stop_loss - current_price)
                reward = max(0.0, current_price - take_profit)
                rr = round(reward / risk, 2)
                if is_star:
                    trigger_name = "Shooting Star (Bearish Pinbar)"
                elif is_bearish_engulfing:
                    trigger_name = "Bearish Engulfing"
                else:
                    trigger_name = "Upper Wick Supply Rejection (>= 2x Body)"
                inval = stop_loss
                rationale = (
                    f"Bearish reversal trigger ({trigger_name}) confirmed at resistance band "
                    f"[${resistance * 0.995:,.2f} - ${resistance * 1.015:,.2f}]; short entry validated toward target."
                )
            else:
                status = "PENDING_CONFIRMATION"
                setup_type = "WAIT"
                entry_range = (float(current_price), float(resistance))
                stop_loss = round(float(resistance * 1.015), 2)
                take_profit = round(float(predicted_price), 2)
                risk = max(1e-6, stop_loss - current_price)
                reward = max(0.0, current_price - take_profit)
                rr = round(reward / risk, 2)
                trigger_name = "None (Consolidation/Testing)"
                inval = round(float(resistance * 1.015), 2)
                rationale = (
                    f"Price testing resistance corridor [${resistance * 0.995:,.2f} - ${resistance * 1.015:,.2f}]; "
                    f"wait for reversal candlestick close confirmation before entering short."
                )

        # Case 2: Bullish setup (predicted_price > current_price)
        elif predicted_price > current_price:
            # Support band: [support * 0.985, support * 1.005]
            breakdown_threshold = support * 0.985

            if close_p < breakdown_threshold:
                # Strong bearish breakdown below support
                status = "INVALIDATED_BREAKDOWN"
                setup_type = "SHORT"
                entry_range = (float(current_price), float(support))
                stop_loss = round(float(support * 1.005), 2)
                take_profit = round(float(current_price - (support - current_price)), 2)
                risk = max(1e-6, stop_loss - current_price)
                reward = max(0.0, current_price - take_profit)
                rr = round(reward / risk, 2)
                trigger_name = "Bearish Breakdown Candle"
                inval = round(float(breakdown_threshold), 2)
                rationale = (
                    f"Strong bearish body closed below support corridor (${breakdown_threshold:,.2f}); "
                    f"bullish bounce invalidated. Shift to breakdown momentum short."
                )
            elif is_hammer or is_bullish_engulfing or (lower_wick >= 2.0 * body and lower_wick > 0):
                status = "CONFIRMED_BUY"
                setup_type = "LONG"
                entry_range = (float(support), float(current_price))
                low_ref = min(low_p, support)
                stop_loss = round(float(low_ref - 0.003 * low_ref), 2)
                take_profit = round(float(predicted_price), 2)
                risk = max(1e-6, current_price - stop_loss)
                reward = max(0.0, take_profit - current_price)
                rr = round(reward / risk, 2)
                if is_hammer:
                    trigger_name = "Hammer (Bullish Pinbar)"
                elif is_bullish_engulfing:
                    trigger_name = "Bullish Engulfing"
                else:
                    trigger_name = "Lower Wick Demand Absorption (>= 2x Body)"
                inval = stop_loss
                rationale = (
                    f"Bullish reversal trigger ({trigger_name}) confirmed at support band "
                    f"[${support * 0.985:,.2f} - ${support * 1.005:,.2f}]; long dip entry validated."
                )
            else:
                status = "PENDING_CONFIRMATION"
                setup_type = "WAIT"
                entry_range = (float(support), float(current_price))
                stop_loss = round(float(support * 0.985), 2)
                take_profit = round(float(predicted_price), 2)
                risk = max(1e-6, current_price - stop_loss)
                reward = max(0.0, take_profit - current_price)
                rr = round(reward / risk, 2)
                trigger_name = "None (Consolidation/Testing)"
                inval = round(float(support * 0.985), 2)
                rationale = (
                    f"Price testing support corridor [${support * 0.985:,.2f} - ${support * 1.005:,.2f}]; "
                    f"wait for reversal candlestick close confirmation before entering long."
                )

        # Case 3: Neutral setup (predicted_price == current_price)
        else:
            status = "NEUTRAL"
            setup_type = "WAIT"
            entry_range = (float(current_price), float(current_price))
            stop_loss = round(float(current_price * 0.99), 2)
            take_profit = round(float(current_price), 2)
            rr = 1.0
            trigger_name = "None"
            inval = float(current_price)
            rationale = "Model price target is flat / neutral; no tactical execution trigger."

        return ConfirmationSignal(
            status=status,
            setup_type=setup_type,
            entry_range=entry_range,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr,
            trigger_candle_name=trigger_name,
            rationale=rationale,
            invalidation_price=inval,
        )


def evaluate_confirmation_signal(
    df: pd.DataFrame,
    horizon: int,
    predicted_price: float,
    current_price: float,
    support: float | None = None,
    resistance: float | None = None,
    market_regime: MarketRegime | None = None,
) -> ConfirmationSignal:
    """Module-level helper to evaluate confirmation signal using default PatternRecognizer."""
    return PatternRecognizer().evaluate_confirmation_signal(
        df=df,
        horizon=horizon,
        predicted_price=predicted_price,
        current_price=current_price,
        support=support,
        resistance=resistance,
        market_regime=market_regime,
    )


def detect_market_regime(df: pd.DataFrame) -> MarketRegime:
    """Module-level helper to detect market regime and trend structure."""
    return PatternRecognizer().detect_market_regime(df)
