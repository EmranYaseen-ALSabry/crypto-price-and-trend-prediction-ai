"""CLI script for dynamic live cryptocurrency price, return, and chart pattern forecasting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src to sys.path
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from crypto_prediction.data.cleaner import DataCleaner
from crypto_prediction.data.collector import DataCollector
from crypto_prediction.features.engineer import FeatureEngineer
from crypto_prediction.features.patterns import ConfirmationSignal, PatternRecognizer
from crypto_prediction.models.boosters import CatBoostModel, LightGBMModel, XGBoostModel
from crypto_prediction.models.linear import LinearRegressionModel, RidgeModel
from crypto_prediction.models.trees import RandomForestModel
from crypto_prediction.utils.chart_plotter import AdvancedChartVisualizer


def run_live_prediction(
    symbol: str = "BTCUSDT",
    horizon: int = 1,
    interval: str = "1h",
    auto_train_if_missing: bool = False,
    render_charts: bool = True,
) -> None:
    """Fetches live market data, identifies technical patterns, and outputs dynamic forecasts."""
    sym = symbol.upper()
    valid_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
    if interval not in valid_intervals:
        print(f"Warning: interval '{interval}' not in standard list {valid_intervals}, attempting anyway.")

    # Determine human-friendly unit
    is_minute = interval.endswith("m")
    is_hour = interval.endswith("h")
    int_val = int(interval[:-1]) if interval[:-1].isdigit() else 1
    total_time_str = f"{horizon * int_val} Minute(s)" if is_minute else (f"{horizon * int_val} Hour(s)" if is_hour else f"{horizon} Bar(s) of {interval}")

    print("=" * 72)
    print(f"  Crypto Price Prediction AI - Dynamic Horizon & Pattern Forecasting")
    print(f"  Asset: {sym} | Interval: {interval} | Target Horizon: {horizon} Bar(s) Ahead (~{total_time_str})")
    print("=" * 72)

    collector = DataCollector()
    raw_df = collector.fetch_historical_bars(symbol=sym, interval=interval, total_bars=10000)

    cleaner = DataCleaner()
    clean_df = cleaner.clean(raw_df)

    last_time = clean_df.index[-1]
    current_price = float(clean_df["close"].iloc[-1])

    # 1. Technical Chart & Candlestick Pattern Recognition
    print(f"\n[+] Last Recorded Bar: {last_time} UTC")
    print(f"[+] Current Market Price: ${current_price:,.2f}")

    recognizer = PatternRecognizer()

    # Market Regime & Trend Structure Recognition
    market_regime = recognizer.detect_market_regime(clean_df)
    reg_icon = "🟢" if "BULLISH" in market_regime.regime else ("🔴" if "BEARISH" in market_regime.regime else "🟡")
    print(f"\n[◈] Market Trend & Regime Analysis:")
    print("-" * 72)
    print(f"  Regime Classification: {market_regime.regime} {reg_icon} ({market_regime.adx_level}, ADX: {market_regime.adx})")
    print(f"  Market Structure:      {market_regime.structure}")
    print(f"  EMA Alignment:         {market_regime.ema_alignment}")
    print(f"  Key Moving Averages:   EMA20: ${market_regime.ema20_last:,.2f} | EMA50: ${market_regime.ema50_last:,.2f} | EMA200: ${market_regime.ema200_last:,.2f}")
    print(f"  Operational Bias:      {market_regime.trend_bias}")
    print(f"  Analysis Note:         {market_regime.description}")

    detected_patterns = recognizer.identify_latest_patterns(clean_df, recent_bars=15)

    print("\n[★] Detected Technical & Chart Patterns:")
    print("-" * 72)
    if detected_patterns:
        for p in detected_patterns:
            icon = "▲ [BULLISH]" if p.signal == "bullish" else "▼ [BEARISH]" if p.signal == "bearish" else "■ [NEUTRAL]"
            print(f"  {icon:<14} {p.name:<26} (Confidence: {p.confidence*100:.0f}%)")
            print(f"    -> Description: {p.description}")
            print(f"    -> Pattern Target Bias: {p.target_price_delta*100:+.2f}%")
    else:
        print("  No high-conviction reversal pattern formed in the trailing 15 bars.")

    # 2. Dynamic Feature Engineering
    engineer = FeatureEngineer(primary_horizon=horizon, include_patterns=True)
    X, y, meta = engineer.prepare_dataset(clean_df, target_horizon=horizon, drop_warmup=True)

    latest_features = engineer.compute_features(clean_df).iloc[-1:]

    # 3. Model Predictions (Load pre-trained or dynamic train on-the-fly)
    models_info = [
        ("CatBoost", CatBoostModel),
        ("Ridge", RidgeModel),
        ("LightGBM", LightGBMModel),
        ("XGBoost", XGBoostModel),
        ("RandomForest", RandomForestModel),
        ("LinearRegression", LinearRegressionModel),
    ]

    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[-] Model Forecasts for {horizon}-Bar Horizon ({interval}):")
    print("-" * 72)
    print(f"{'Model Name':<18} | {'Return Forecast':<16} | {'Price Target':<14} | {'Direction'}")
    print("-" * 72)

    predicted_returns: list[float] = []
    expected_features = list(X.columns)

    for name, model_cls in models_info:
        # Check standard convention first, then fallback to legacy convention
        model_file = models_dir / f"{sym}_{interval}_{name}_h{horizon}.joblib"
        legacy_file = models_dir / f"{sym}_{name}_h{horizon}.joblib"
        target_load_file = model_file if model_file.exists() else (legacy_file if legacy_file.exists() else model_file)

        model = None
        if target_load_file.exists():
            try:
                candidate = model_cls.load(target_load_file)
                # Verify feature schema compatibility
                schema_valid = False
                if hasattr(candidate, "feature_names") and candidate.feature_names is not None:
                    if candidate.feature_names == expected_features:
                        schema_valid = True
                    else:
                        print(f"  [!] Schema mismatch for {name}: expected {len(expected_features)} features, artifact has {len(candidate.feature_names)}.")
                elif hasattr(candidate, "pipeline") and hasattr(candidate.pipeline.named_steps.get("scaler"), "n_features_in_"):
                    n_in = candidate.pipeline.named_steps["scaler"].n_features_in_
                    if n_in == len(expected_features):
                        schema_valid = True
                    else:
                        print(f"  [!] Dimension mismatch for {name}: expected {len(expected_features)}, artifact expects {n_in}.")
                else:
                    inner_model = getattr(candidate, "model", None)
                    if hasattr(inner_model, "n_features_in_"):
                        if inner_model.n_features_in_ == len(expected_features):
                            schema_valid = True
                        else:
                            print(f"  [!] Dimension mismatch for {name}: expected {len(expected_features)}, artifact expects {inner_model.n_features_in_}.")
                    else:
                        schema_valid = True

                if schema_valid:
                    model = candidate
            except Exception as e:
                print(f"  [!] Failed loading {target_load_file.name}: {e}")
                model = None

        if model is None:
            if auto_train_if_missing:
                # Explicitly allowed dynamic training
                print(f"  [i] Training {name} on-the-fly (auto_train_if_missing=True)...")
                if name == "CatBoost":
                    model = CatBoostModel(iterations=100, depth=4, verbose=0)
                elif name == "Ridge":
                    model = RidgeModel(alpha=1.0)
                elif name == "LightGBM":
                    model = LightGBMModel(n_estimators=100, max_depth=4, verbose=-1)
                elif name == "XGBoost":
                    model = XGBoostModel(n_estimators=100, max_depth=4)
                elif name == "RandomForest":
                    model = RandomForestModel(n_estimators=100, max_depth=5)
                elif name == "LinearRegression":
                    model = LinearRegressionModel()

                if model is not None:
                    model.fit(X, y)
                    model.save(model_file)
            else:
                print(f"  [-] {name:<18} | Skipped (artifact missing or incompatible; pass --train-if-missing)")

        if model is not None:
            ret = float(model.predict(latest_features)[0])
            predicted_returns.append(ret)
            target_p = current_price * (1.0 + ret)
            direction = "▲ UP (Bullish)" if ret > 0 else "▼ DOWN (Bearish)"
            print(f"{name:<18} | {ret * 100:>+12.3f}%   | ${target_p:>11,.2f} | {direction}")

    if predicted_returns:
        avg_ret = float(np.mean(predicted_returns))
        avg_price = current_price * (1.0 + avg_ret)
        avg_dir = "▲ UP (Bullish)" if avg_ret > 0 else "▼ DOWN (Bearish)"

        # Calculate prediction interval based on horizon and volatility
        std_ret = float(np.std(predicted_returns)) if len(predicted_returns) > 1 else 0.005
        # ATR / volatility scaled to horizon
        hourly_vol = float(clean_df["close"].pct_change().rolling(24).std().iloc[-1])
        horizon_vol = hourly_vol * np.sqrt(horizon)
        margin_error = max(1.96 * horizon_vol, std_ret * 1.96)
        lower_price = current_price * (1.0 + avg_ret - margin_error)
        upper_price = current_price * (1.0 + avg_ret + margin_error)

        print("-" * 72)
        print(f"{'ENSEMBLE AVERAGE':<18} | {avg_ret * 100:>+12.3f}%   | ${avg_price:>11,.2f} | {avg_dir}")
        print(f"{'PREDICTION CONE (95%)':<18} | ${lower_price:>11,.2f}  <--->  ${upper_price:>11,.2f}")
        print("=" * 72)

        # 4. Conditional Scenario Trading & Candlestick Confirmation Evaluation
        is_minute = interval.endswith("m")
        corridor_window = min(10 if is_minute else 24, len(clean_df) // 2)
        support_s, resist_s = recognizer.compute_dynamic_support_resistance(
            clean_df["high"], clean_df["low"], clean_df["close"], window=corridor_window
        )
        last_support = float(support_s.dropna().iloc[-1]) if not support_s.dropna().empty else current_price * 0.98
        last_resistance = float(resist_s.dropna().iloc[-1]) if not resist_s.dropna().empty else current_price * 1.02

        confirmation_signal = recognizer.evaluate_confirmation_signal(
            df=clean_df,
            horizon=horizon,
            predicted_price=avg_price,
            current_price=current_price,
            support=last_support,
            resistance=last_resistance,
            market_regime=market_regime,
            interval=interval,
        )

        badge_map = {
            "CONFIRMED_SELL": "[CONFIRMED REVERSAL TRIGGER - ENTER SHORT]",
            "CONFIRMED_BUY": "[CONFIRMED REVERSAL TRIGGER - ENTER LONG]",
            "PENDING_CONFIRMATION": "[PENDING CONFIRMATION - WAIT FOR CANDLE CLOSE]",
            "INVALIDATED_BREAKOUT": "[INVALIDATED BREAKOUT - MOMENTUM LONG SHIFT]",
            "INVALIDATED_BREAKDOWN": "[INVALIDATED BREAKDOWN - MOMENTUM SHORT SHIFT]",
            "NEUTRAL": "[NEUTRAL - NO SETUP]",
        }
        status_badge = badge_map.get(confirmation_signal.status, f"[{confirmation_signal.status}]")

        action_map = {
            "SHORT": "SHORT PULLBACK (Target Lower Prediction Boundary)",
            "LONG": "LONG DIP (Target Upper Prediction Boundary)",
            "WAIT": "WAIT / STANDBY (Corridor Retest in Progress)",
        }
        trade_action = action_map.get(confirmation_signal.setup_type, confirmation_signal.setup_type)

        entry_min = min(confirmation_signal.entry_range)
        entry_max = max(confirmation_signal.entry_range)

        print("\n" + "=" * 72)
        print("  TACTICAL EXECUTION SETUP & CANDLESTICK CONFIRMATION")
        print("=" * 72)
        print(f"  Status Badge:       {status_badge}")
        print(f"  Trade Action:       {trade_action}")
        print(f"  Entry Corridor:     ${entry_min:>11,.2f}  <--->  ${entry_max:>11,.2f}")
        print(f"  Stop Loss:          ${confirmation_signal.stop_loss:>11,.2f}")
        print(f"  Take Profit:        ${confirmation_signal.take_profit:>11,.2f}")
        print(f"  Risk / Reward (R:R): {confirmation_signal.risk_reward_ratio:>10.2f} : 1")
        print(f"  Trigger Candle:     {confirmation_signal.trigger_candle_name}")
        print(f"  Invalidation Level: ${confirmation_signal.invalidation_price:>11,.2f}")
        print(f"  Reversal Condition: {confirmation_signal.rationale}")
        print("=" * 72 + "\n")

        # 5. Render and Save Visual Charts
        if render_charts:
            figures_dir = Path("artifacts/figures")
            figures_dir.mkdir(parents=True, exist_ok=True)
            html_path = figures_dir / f"{sym}_{interval}_forecast_projection_{horizon}.html"
            png_path = figures_dir / f"{sym}_{interval}_forecast_projection_{horizon}.png"

            visualizer = AdvancedChartVisualizer()
            visualizer.create_interactive_projection_chart(
                df=clean_df,
                symbol=sym,
                horizon=horizon,
                interval=interval,
                predicted_price=avg_price,
                lower_bound=lower_price,
                upper_bound=upper_price,
                patterns=detected_patterns,
                save_html_path=html_path,
                confirmation_signal=confirmation_signal,
            )
            visualizer.render_matplotlib_projection(
                df=clean_df,
                symbol=sym,
                horizon=horizon,
                interval=interval,
                predicted_price=avg_price,
                lower_bound=lower_price,
                upper_bound=upper_price,
                save_path=png_path,
                confirmation_signal=confirmation_signal,
            )
            if html_path.exists():
                print(f"[✓] Interactive Chart Generated: {html_path}")
            if png_path.exists():
                print(f"[✓] Static Projection Chart:     {png_path}\n")
    else:
        print("\n[!] No valid model predictions generated.")
        print("    If model artifacts are missing, run training or pass --train-if-missing.")

    print("Disclaimer: For research & quantitative education only. Not financial advice.\n")


def _parse_horizon_arg(val: str | int) -> int:
    """Parses horizon argument allowing either pure integers (e.g. 1, 4) or timeframe-suffixed strings (e.g. '1h', '4h')."""
    if isinstance(val, int):
        return val
    s = str(val).strip().lower().rstrip("mhd")
    return int(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Crypto Price & Pattern Prediction CLI")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Crypto symbol (BTCUSDT, ETHUSDT, SOLUSDT)")
    parser.add_argument("--interval", type=str, default="1h", help="Timeframe interval (e.g. 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h)")
    parser.add_argument("--horizon", type=_parse_horizon_arg, default=1, help="Forecast horizon in bars/periods (e.g. 1 or '1h')")
    parser.add_argument("--train-if-missing", action="store_true", help="Allow training model artifacts on-the-fly if missing or incompatible")
    parser.add_argument("--no-charts", action="store_true", help="Disable chart rendering")
    args = parser.parse_args()

    run_live_prediction(
        symbol=args.symbol,
        horizon=args.horizon,
        interval=args.interval,
        auto_train_if_missing=args.train_if_missing,
        render_charts=not args.no_charts,
    )
