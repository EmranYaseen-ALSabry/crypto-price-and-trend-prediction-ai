"""Unit tests for technical chart and candlestick pattern recognition engine."""

import numpy as np
import pandas as pd
import pytest

from crypto_prediction.data.collector import DataCollector
from crypto_prediction.features.patterns import ConfirmationSignal, PatternRecognizer


@pytest.fixture
def ohlcv_pattern_df() -> pd.DataFrame:
    collector = DataCollector()
    return collector.generate_synthetic_ohlcv(symbol="BTCUSDT", n_bars=300, seed=42)


class TestPatternRecognition:
    """Verifies that candlestick and chart patterns are detected causally and accurately."""

    def test_pinbar_detection(self) -> None:
        recognizer = PatternRecognizer()
        # Synthetic Hammer: Open 100, High 100.1, Low 85, Close 99.8 -> Lower shadow = 14.8, Body = 0.2, Upper shadow = 0.1
        opens = np.array([100.0])
        highs = np.array([100.2])
        lows = np.array([85.0])
        closes = np.array([99.0])

        signals = recognizer.detect_pinbars(opens, highs, lows, closes)
        # Lower shadow = 14.0, Body = 1.0, Upper shadow = 0.2 <= 0.3 * 1.0
        assert signals[0] == 1.0  # Bullish Hammer

        # Synthetic Shooting Star: Open 90, High 105, Low 89.8, Close 91 -> Upper shadow = 14.0, Body = 1.0, Lower shadow = 0.2
        opens_star = np.array([90.0])
        highs_star = np.array([105.0])
        lows_star = np.array([89.8])
        closes_star = np.array([91.0])

        signals_star = recognizer.detect_pinbars(opens_star, highs_star, lows_star, closes_star)
        assert signals_star[0] == -1.0  # Bearish Shooting Star

    def test_engulfing_detection(self) -> None:
        recognizer = PatternRecognizer()
        # Bar 0: Red candle (Open 100, Close 95)
        # Bar 1: Green candle (Open 94, Close 102) -> Bullish engulfing
        opens = np.array([100.0, 94.0])
        highs = np.array([101.0, 103.0])
        lows = np.array([94.0, 93.0])
        closes = np.array([95.0, 102.0])

        signals = recognizer.detect_engulfing(opens, highs, lows, closes)
        assert signals[1] == 1.0  # Bullish Engulfing

    def test_support_resistance_causality(self, ohlcv_pattern_df: pd.DataFrame) -> None:
        recognizer = PatternRecognizer()
        support, resistance = recognizer.compute_dynamic_support_resistance(
            ohlcv_pattern_df["high"], ohlcv_pattern_df["low"], ohlcv_pattern_df["close"], window=24
        )

        assert len(support) == len(ohlcv_pattern_df)
        assert len(resistance) == len(ohlcv_pattern_df)

        valid_mask = (~support.isna()) & (~resistance.isna())
        # Resistance must be >= Support
        assert (resistance[valid_mask] >= support[valid_mask]).all()

    def test_pattern_feature_extraction(self, ohlcv_pattern_df: pd.DataFrame) -> None:
        recognizer = PatternRecognizer()
        feat = recognizer.extract_pattern_features(ohlcv_pattern_df)

        expected_cols = [
            "pattern_pinbar",
            "pattern_engulfing",
            "pattern_doji",
            "dist_to_support_48h",
            "dist_to_resistance_48h",
            "pattern_double_bottom",
            "pattern_double_top",
            "pattern_composite_signal",
        ]

        for col in expected_cols:
            assert col in feat.columns
            assert not np.isinf(feat[col].dropna()).any()

    def test_confirmation_signal_evaluation(self) -> None:
        recognizer = PatternRecognizer()
        resistance = 82000.0
        support = 80000.0

        # 1. Synthetic shooting star near resistance produces CONFIRMED_SELL with positive R:R
        # Upper wick >= 2x body, close near open, inside resistance corridor
        df_shooting_star = pd.DataFrame([
            {"open": 81500.0, "high": 81700.0, "low": 81400.0, "close": 81600.0, "volume": 100.0},
            {"open": 81800.0, "high": 82500.0, "low": 81750.0, "close": 81820.0, "volume": 150.0},
        ])
        signal_star = recognizer.evaluate_confirmation_signal(
            df=df_shooting_star,
            horizon=17,
            predicted_price=79000.0,
            current_price=81820.0,
            support=support,
            resistance=resistance,
        )
        assert isinstance(signal_star, ConfirmationSignal)
        assert signal_star.status == "CONFIRMED_SELL"
        assert signal_star.setup_type == "SHORT"
        assert signal_star.risk_reward_ratio > 0
        assert signal_star.take_profit == 79000.0
        assert signal_star.stop_loss > 81820.0

        # 2. Synthetic normal candle near resistance produces PENDING_CONFIRMATION
        df_normal = pd.DataFrame([
            {"open": 81500.0, "high": 81700.0, "low": 81400.0, "close": 81600.0, "volume": 100.0},
            {"open": 81750.0, "high": 81900.0, "low": 81700.0, "close": 81850.0, "volume": 100.0},
        ])
        signal_normal = recognizer.evaluate_confirmation_signal(
            df=df_normal,
            horizon=17,
            predicted_price=79000.0,
            current_price=81850.0,
            support=support,
            resistance=resistance,
        )
        assert signal_normal.status == "PENDING_CONFIRMATION"
        assert signal_normal.setup_type == "WAIT"

        # 3. Breakout candle produces INVALIDATED_BREAKOUT
        # Candle closes above resistance * 1.015 (82000 * 1.015 = 83230)
        df_breakout = pd.DataFrame([
            {"open": 82000.0, "high": 82200.0, "low": 81900.0, "close": 82100.0, "volume": 100.0},
            {"open": 82100.0, "high": 83600.0, "low": 82050.0, "close": 83500.0, "volume": 300.0},
        ])
        signal_breakout = recognizer.evaluate_confirmation_signal(
            df=df_breakout,
            horizon=17,
            predicted_price=79000.0,
            current_price=83500.0,
            support=support,
            resistance=resistance,
        )
        assert signal_breakout.status == "INVALIDATED_BREAKOUT"
        assert signal_breakout.setup_type == "LONG"

    def test_market_regime_detection(self) -> None:
        recognizer = PatternRecognizer()

        # 1. Strong Bullish Trend: monotonic upward prices
        dates = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
        prices_up = np.linspace(50000, 70000, 100)
        df_bullish = pd.DataFrame({
            "open": prices_up - 20,
            "high": prices_up + 50,
            "low": prices_up - 50,
            "close": prices_up,
            "volume": 1000.0,
        }, index=dates)

        regime_bull = recognizer.detect_market_regime(df_bullish)
        assert "BULLISH" in regime_bull.regime
        assert regime_bull.trend_bias == "BUY_DIPS_ONLY"
        assert regime_bull.ema20_last > regime_bull.ema50_last
        assert regime_bull.adx > 0

        # 2. Strong Bearish Trend: monotonic downward prices
        prices_down = np.linspace(70000, 50000, 100)
        df_bearish = pd.DataFrame({
            "open": prices_down + 20,
            "high": prices_down + 50,
            "low": prices_down - 50,
            "close": prices_down,
            "volume": 1000.0,
        }, index=dates)

        regime_bear = recognizer.detect_market_regime(df_bearish)
        assert "BEARISH" in regime_bear.regime
        assert regime_bear.trend_bias == "SELL_RALLIES_ONLY"
        assert regime_bear.ema20_last < regime_bear.ema50_last
