"""Interactive Candlestick and Chart Pattern Visualizer with Future Price Projection Cone."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from crypto_prediction.features.patterns import ConfirmationSignal, DetectedPattern, MarketRegime, PatternRecognizer


def _parse_time_delta(interval: str, horizon: int) -> pd.Timedelta:
    """Parses interval string (e.g. '1m', '5m', '15m', '1h', '4h', '1d') and scales by horizon."""
    import re
    m = re.match(r"^(\d+)([mhd])$", interval.strip().lower())
    if not m:
        return pd.Timedelta(hours=horizon)
    val, unit = int(m.group(1)), m.group(2)
    deltas = {
        "m": pd.Timedelta(minutes=val * horizon),
        "h": pd.Timedelta(hours=val * horizon),
        "d": pd.Timedelta(days=val * horizon),
    }
    return deltas.get(unit, pd.Timedelta(hours=horizon))


class AdvancedChartVisualizer:
    """Renders interactive candlestick charts annotated with detected patterns and future price projections."""

    def __init__(self, theme: str = "plotly_dark") -> None:
        self.theme = theme
        self.recognizer = PatternRecognizer()

    def create_interactive_projection_chart(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        horizon: int = 1,
        interval: str = "1h",
        predicted_price: float | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        recent_bars: int = 120,
        patterns: list[DetectedPattern] | None = None,
        save_html_path: str | Path | None = None,
        save_png_path: str | Path | None = None,
        confirmation_signal: ConfirmationSignal | None = None,
    ) -> go.Figure:
        """Constructs an interactive Plotly candlestick chart with patterns and future forecast cone.

        Args:
            df: Cleaned OHLCV DataFrame.
            symbol: Cryptocurrency symbol.
            horizon: Forward prediction horizon in intervals/bars.
            interval: Bar interval (e.g. '1m', '5m', '15m', '1h', '4h').
            predicted_price: Model forecasted price at t+h.
            lower_bound: Lower prediction interval boundary.
            upper_bound: Upper prediction interval boundary.
            recent_bars: Number of trailing bars to display for clarity.
            patterns: Optional list of DetectedPattern objects.
            save_html_path: File path to save interactive HTML.
            save_png_path: File path to save static image.
            confirmation_signal: Optional ConfirmationSignal for tactical execution overlay.

        Returns:
            Plotly Figure object.
        """
        # Dynamic recent bars displayed: 300+ bars as requested for comprehensive technical context
        is_minute = interval.endswith("m")
        bars_to_show = max(300, recent_bars)
        display_df = df.iloc[-bars_to_show:].copy()
        if not isinstance(display_df.index, pd.DatetimeIndex):
            if "open_time" in display_df.columns:
                display_df["open_time"] = pd.to_datetime(display_df["open_time"], utc=True)
                display_df = display_df.set_index("open_time")
            else:
                try:
                    display_df.index = pd.to_datetime(display_df.index, utc=True)
                except Exception:
                    display_df.index = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(display_df), freq=interval)

        # 1. Base Subplots (Candlesticks on row 1, Volume on row 2)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25],
            subplot_titles=[
                "",
                "Trading Volume",
            ],
        )

        # 2. Candlestick Trace
        fig.add_trace(
            go.Candlestick(
                x=display_df.index,
                open=display_df["open"],
                high=display_df["high"],
                low=display_df["low"],
                close=display_df["close"],
                name=f"{symbol} OHLC",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=1,
            col=1,
        )

        # 3. EMA Trend Lines (EMA 20, 50, 200)
        ema20 = display_df["close"].ewm(span=20, adjust=False).mean()
        ema50 = display_df["close"].ewm(span=50, adjust=False).mean()
        ema200 = df["close"].ewm(span=min(200, len(df)), adjust=False).mean().iloc[-len(display_df):]

        fig.add_trace(
            go.Scatter(
                x=display_df.index,
                y=ema20,
                mode="lines",
                name="EMA 20",
                line=dict(color="#00e5ff", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=display_df.index,
                y=ema50,
                mode="lines",
                name="EMA 50",
                line=dict(color="#ffb300", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=display_df.index,
                y=ema200,
                mode="lines",
                name="EMA 200 (Macro)",
                line=dict(color="#e040fb", width=1.8, dash="dot"),
            ),
            row=1,
            col=1,
        )

        # 4. Market Regime & Trend Structure Detection
        market_regime = self.recognizer.detect_market_regime(df)

        disp_start = display_df.index[0]
        for pt in market_regime.swing_points:
            pt_idx = pt.get("index")
            pt_label = pt.get("label", "")
            if pt_idx >= disp_start and pt_label in ("HH", "HL", "LH", "LL"):
                is_high = pt["type"] == "HIGH"
                y_pos = pt["price"]
                # Color code market structure based on trend nature
                if pt_label == "HH":
                    badge_col = "#00e676"  # Bullish Higher High (Green)
                elif pt_label == "HL":
                    badge_col = "#00e5ff"  # Bullish Higher Low (Cyan)
                elif pt_label == "LH":
                    badge_col = "#ef5350"  # Bearish Lower High (Red)
                elif pt_label == "LL":
                    badge_col = "#ff9800"  # Bearish Lower Low (Amber/Orange breakdown low)
                else:
                    badge_col = "#78909c"

                fig.add_annotation(
                    x=pt_idx,
                    y=y_pos,
                    text=f"<b>{pt_label}</b>",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=0.8,
                    arrowcolor=badge_col,
                    ay=-22 if is_high else 22,
                    font=dict(color="#ffffff", size=9),
                    bgcolor=badge_col,
                    row=1,
                    col=1,
                )

        # Prominent Market Regime Header Banner (Includes symbol, regime, bias, and detected patterns)
        detected = patterns or self.recognizer.identify_latest_patterns(df)
        pat_suffix = ""
        if detected:
            pat_badges = " | ".join([f"[{p.signal.upper()}] {p.name}" for p in detected[:3]])
            pat_suffix = f"<br><span style='font-size:10px;font-weight:normal;opacity:0.95;'>Active Patterns: {pat_badges}</span>"

        # Correctly format horizon display string
        int_val = int(interval[:-1]) if interval[:-1].isdigit() else 1
        time_unit_str = f"{horizon * int_val}m" if interval.endswith("m") else (f"{horizon * int_val}h" if interval.endswith("h") else f"{horizon} bars")
        header_title = f"{symbol} ({time_unit_str}) | [REGIME: {market_regime.regime}]"

        regime_col = "#00e676" if "BULLISH" in market_regime.regime else ("#ff1744" if "BEARISH" in market_regime.regime else "#ffab00")
        fig.add_annotation(
            text=f"<b>{header_title}</b> | ADX: {market_regime.adx} ({market_regime.adx_level}) | Structure: {market_regime.structure} | Bias: {market_regime.trend_bias}{pat_suffix}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.12,
            showarrow=False,
            font=dict(size=11, color="#ffffff"),
            bgcolor=regime_col,
            bordercolor="#ffffff",
            borderwidth=1,
            xanchor="center",
        )

        # 5. Volume Bar Trace
        colors = np.where(display_df["close"] >= display_df["open"], "#26a69a", "#ef5350")
        fig.add_trace(
            go.Bar(
                x=display_df.index,
                y=display_df["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

        # 6. Support and Resistance Levels
        corridor_window = min(10 if is_minute else 24, len(display_df) // 2)
        support, resistance = self.recognizer.compute_dynamic_support_resistance(
            display_df["high"], display_df["low"], display_df["close"], window=corridor_window
        )
        last_support = float(support.dropna().iloc[-1]) if not support.dropna().empty else float(display_df["low"].min())
        last_resistance = float(resistance.dropna().iloc[-1]) if not resistance.dropna().empty else float(display_df["high"].max())
        last_price = float(display_df["close"].iloc[-1])

        # Evaluate confirmation signal if not explicitly provided
        if confirmation_signal is None and predicted_price is not None:
            confirmation_signal = self.recognizer.evaluate_confirmation_signal(
                df=df,
                horizon=horizon,
                predicted_price=predicted_price,
                current_price=last_price,
                support=last_support,
                resistance=last_resistance,
                market_regime=market_regime,
                interval=interval,
            )

        # Draw shaded translucent rectangles for Supply & Demand zones (Clean zones without text clutter)
        supply_y0 = last_resistance * 0.995
        supply_y1 = max(last_resistance * 1.015, upper_bound if upper_bound is not None else last_resistance * 1.015)
        fig.add_hrect(
            y0=supply_y0,
            y1=supply_y1,
            fillcolor="rgba(239, 83, 80, 0.12)",
            line_width=1,
            line_dash="dot",
            line_color="rgba(239, 83, 80, 0.4)",
            row=1,
            col=1,
        )

        demand_y0 = min(last_support * 0.985, lower_bound if lower_bound is not None else last_support * 0.985)
        demand_y1 = last_support * 1.005
        fig.add_hrect(
            y0=demand_y0,
            y1=demand_y1,
            fillcolor="rgba(0, 230, 118, 0.12)",
            line_width=1,
            line_dash="dot",
            line_color="rgba(0, 230, 118, 0.4)",
            row=1,
            col=1,
        )

        # Key Price Levels - Clean, non-overlapping single labels on the right margin
        fig.add_hline(
            y=last_resistance,
            line_dash="dot",
            line_color="#ef5350",
            line_width=1.5,
            annotation_text=f"Resistance: ${last_resistance:,.1f}",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#ef5350"),
            annotation_yshift=6,
            row=1,
            col=1,
        )

        tp_at_support = (confirmation_signal is not None and abs(confirmation_signal.take_profit - last_support) / last_support < 0.005)
        if not tp_at_support:
            fig.add_hline(
                y=last_support,
                line_dash="dot",
                line_color="#00e676",
                line_width=1.5,
                annotation_text=f"Support: ${last_support:,.1f}",
                annotation_position="bottom right",
                annotation_font=dict(size=10, color="#00e676"),
                annotation_yshift=-6,
                row=1,
                col=1,
            )

        # 6. Future Price Forecast Trajectory and Uncertainty Cone
        if predicted_price is not None:
            last_time = display_df.index[-1]
            delta_forward = _parse_time_delta(interval, horizon)
            target_time = last_time + delta_forward

            # Trajectory Line
            traj_x = [last_time, target_time]
            traj_y = [last_price, predicted_price]
            dir_color = "#00e676" if predicted_price >= last_price else "#ff1744"

            fig.add_trace(
                go.Scatter(
                    x=traj_x,
                    y=traj_y,
                    mode="lines+markers",
                    name=f"Forecast ({horizon}x{interval}): ${predicted_price:,.2f}",
                    line=dict(color=dir_color, width=2.5, dash="dash"),
                    marker=dict(size=7, color=dir_color),
                ),
                row=1,
                col=1,
            )

            # Shaded Uncertainty Cone (Upper & Lower Bounds)
            if lower_bound is not None and upper_bound is not None:
                cone_x = [last_time, target_time, target_time, last_time]
                cone_y = [last_price, upper_bound, lower_bound, last_price]

                fig.add_trace(
                    go.Scatter(
                        x=cone_x,
                        y=cone_y,
                        fill="toself",
                        fillcolor="rgba(33, 150, 243, 0.22)",
                        line=dict(color="rgba(33, 150, 243, 0.6)", width=1.5, dash="dot"),
                        name="Uncertainty Corridor (±1.96σ)",
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )

            # Horizontal dashed lines & Scenario Annotations from confirmation_signal
            if confirmation_signal is not None:
                sl_price = confirmation_signal.stop_loss
                tp_price = confirmation_signal.take_profit

                # Take Profit Target Horizontal Dashed Line (consolidated with support if coincident)
                tp_label = f"TP Target (Support): ${tp_price:,.1f}" if tp_at_support else f"TP Target: ${tp_price:,.1f}"
                fig.add_hline(
                    y=tp_price,
                    line_dash="dash",
                    line_color="#00e5ff",
                    line_width=2,
                    annotation_text=tp_label,
                    annotation_position="bottom right",
                    annotation_font=dict(size=10, color="#00e5ff"),
                    annotation_yshift=-8,
                    row=1,
                    col=1,
                )

                # Stop Loss Level Horizontal Dashed Line
                fig.add_hline(
                    y=sl_price,
                    line_dash="dash",
                    line_color="#ff9100",
                    line_width=2,
                    annotation_text=f"Stop Loss: ${sl_price:,.1f}",
                    annotation_position="top right",
                    annotation_font=dict(size=10, color="#ff9100"),
                    annotation_yshift=8,
                    row=1,
                    col=1,
                )

                # Directional Arrow aligned with Setup Type
                if confirmation_signal.setup_type == "SHORT":
                    # An arrow pointing from current price/entry toward target
                    entry_start_y = last_price
                    fig.add_annotation(
                        x=target_time,
                        y=tp_price,
                        ax=last_time,
                        ay=entry_start_y,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        text=f"Short Setup -> TP: ${tp_price:,.0f}",
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1.2,
                        arrowwidth=2,
                        arrowcolor="#ef5350",
                        font=dict(color="#ffffff", size=9),
                        bgcolor="rgba(239, 83, 80, 0.85)",
                        bordercolor="#ef5350",
                        row=1,
                        col=1,
                    )
                elif confirmation_signal.setup_type == "LONG":
                    # For long setup, originate arrow from current price toward the target profit level
                    entry_start_y = last_price
                    fig.add_annotation(
                        x=target_time,
                        y=tp_price,
                        ax=last_time,
                        ay=entry_start_y,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        text=f"Long Setup -> TP: ${tp_price:,.0f}",
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1.2,
                        arrowwidth=2,
                        arrowcolor="#00e676",
                        font=dict(color="#ffffff", size=9),
                        bgcolor="rgba(0, 230, 118, 0.85)",
                        bordercolor="#00e676",
                        row=1,
                        col=1,
                    )
                else:
                    # Tactical WAIT retest: position above prediction and right-aligned to prevent overlapping TP/candles
                    fig.add_annotation(
                        x=target_time,
                        y=predicted_price,
                        ax=last_time,
                        ay=last_price,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        text=f"Tactical [WAIT] -> Retest ${predicted_price:,.1f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.0,
                        arrowwidth=1.5,
                        arrowcolor="#ffb300",
                        font=dict(color="#ffffff", size=9),
                        bgcolor="rgba(15, 23, 42, 0.88)",
                        bordercolor="#ffb300",
                        yshift=18,
                        xshift=10,
                        row=1,
                        col=1,
                    )

        # 7. Layout and Styling with dedicated top clearance (t=85) and wide right margin (r=140)
        fig.update_layout(
            template=self.theme,
            xaxis_rangeslider_visible=False,
            height=720,
            margin=dict(l=50, r=140, t=85, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="center",
                x=0.5,
                font=dict(size=10),
            ),
        )

        if save_html_path:
            p = Path(save_html_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(p))

        return fig

    def render_matplotlib_projection(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        horizon: int = 1,
        interval: str = "1h",
        predicted_price: float | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        save_path: str | Path | None = None,
        confirmation_signal: ConfirmationSignal | None = None,
    ) -> plt.Figure:
        """Renders publication-grade static Matplotlib candlestick chart with forecast cone."""
        is_minute = interval.endswith("m")
        bars_to_show = min(300, len(df))
        display_df = df.iloc[-bars_to_show:].copy()
        if not isinstance(display_df.index, pd.DatetimeIndex):
            if "open_time" in display_df.columns:
                display_df["open_time"] = pd.to_datetime(display_df["open_time"], utc=True)
                display_df = display_df.set_index("open_time")
            else:
                try:
                    display_df.index = pd.to_datetime(display_df.index, utc=True)
                except Exception:
                    display_df.index = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(display_df), freq=interval)

        fig, (ax1, ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(14, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        # Plot close line & shaded price action
        ax1.plot(display_df.index, display_df["close"], color="#2980b9", lw=1.5, label="Close Price")
        ax1.fill_between(
            display_df.index,
            display_df["low"],
            display_df["high"],
            color="#3498db",
            alpha=0.2,
            label="High-Low Range",
        )

        # EMA Trend Lines (EMA 20, 50, 200)
        ema20 = display_df["close"].ewm(span=20, adjust=False).mean()
        ema50 = display_df["close"].ewm(span=50, adjust=False).mean()
        ema200 = df["close"].ewm(span=min(200, len(df)), adjust=False).mean().iloc[-len(display_df):]

        ax1.plot(display_df.index, ema20, color="#00e5ff", lw=1.2, label="EMA 20")
        ax1.plot(display_df.index, ema50, color="#ffb300", lw=1.2, label="EMA 50")
        ax1.plot(display_df.index, ema200, color="#e040fb", lw=1.5, ls="--", label="EMA 200 (Macro)")

        # Market Regime Badge
        market_regime = self.recognizer.detect_market_regime(df)
        regime_fc = "#27ae60" if "BULLISH" in market_regime.regime else ("#c0392b" if "BEARISH" in market_regime.regime else "#d35400")
        int_val = int(interval[:-1]) if interval[:-1].isdigit() else 1
        time_unit_str = f"{horizon * int_val}m" if interval.endswith("m") else (f"{horizon * int_val}h" if interval.endswith("h") else f"{horizon} bars")
        ax1.text(
            0.02, 0.95,
            f"{symbol} ({time_unit_str}) | REGIME: {market_regime.regime} | ADX: {market_regime.adx:.1f} ({market_regime.adx_level})\nStructure: {market_regime.structure} | Bias: {market_regime.trend_bias}",
            transform=ax1.transAxes,
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.35", fc=regime_fc, ec="white", lw=0.8, alpha=0.9),
            verticalalignment="top",
        )

        # Support & Resistance
        corridor_window = min(10 if is_minute else 24, len(display_df) // 2)
        support, resistance = self.recognizer.compute_dynamic_support_resistance(
            display_df["high"], display_df["low"], display_df["close"], window=corridor_window
        )
        last_supp = float(support.dropna().iloc[-1]) if not support.dropna().empty else float(display_df["low"].min())
        last_res = float(resistance.dropna().iloc[-1]) if not resistance.dropna().empty else float(display_df["high"].max())
        last_price = float(display_df["close"].iloc[-1])

        # Evaluate confirmation signal if not provided
        if confirmation_signal is None and predicted_price is not None:
            confirmation_signal = self.recognizer.evaluate_confirmation_signal(
                df=df,
                horizon=horizon,
                predicted_price=predicted_price,
                current_price=last_price,
                support=last_supp,
                resistance=last_res,
                market_regime=market_regime,
                interval=interval,
            )

        # Shaded translucent rectangles for Supply & Demand zones
        supply_y0 = last_res * 0.995
        supply_y1 = max(last_res * 1.015, upper_bound if upper_bound is not None else last_res * 1.015)
        ax1.axhspan(supply_y0, supply_y1, color="#ef5350", alpha=0.15)

        demand_y0 = min(last_supp * 0.985, lower_bound if lower_bound is not None else last_supp * 0.985)
        demand_y1 = last_supp * 1.005
        ax1.axhspan(demand_y0, demand_y1, color="#26a69a", alpha=0.15)

        ax1.axhline(last_res, color="#c0392b", ls=":", alpha=0.8, label=f"Resistance: ${last_res:,.1f}")
        tp_at_supp = confirmation_signal is not None and abs(confirmation_signal.take_profit - last_supp) / last_supp < 0.005
        if not tp_at_supp:
            ax1.axhline(last_supp, color="#27ae60", ls=":", alpha=0.8, label=f"Support: ${last_supp:,.1f}")

        # Forecast Trajectory & Uncertainty Cone
        if predicted_price is not None:
            last_time = display_df.index[-1]
            delta_forward = _parse_time_delta(interval, horizon)
            target_time = last_time + delta_forward
            color = "#27ae60" if predicted_price >= last_price else "#e74c3c"

            ax1.plot([last_time, target_time], [last_price, predicted_price], color=color, ls="--", lw=2.2, marker="o", label=f"Forecast ({horizon}x{interval})")

            if lower_bound is not None and upper_bound is not None:
                ax1.fill(
                    [last_time, target_time, target_time, last_time],
                    [last_price, upper_bound, lower_bound, last_price],
                    color="#2980b9",
                    alpha=0.15,
                    label="Prediction Corridor (±1.96σ)",
                )

            # Clear horizontal dashed lines & Scenario Annotations
            if confirmation_signal is not None:
                sl_price = confirmation_signal.stop_loss
                tp_price = confirmation_signal.take_profit

                ax1.axhline(
                    sl_price,
                    color="#ff9100",
                    ls="--",
                    lw=1.8,
                    label=f"Stop Loss: ${sl_price:,.1f}",
                )
                tp_lbl = f"TP Target (Support): ${tp_price:,.1f}" if tp_at_supp else f"Take Profit: ${tp_price:,.1f}"
                ax1.axhline(
                    tp_price,
                    color="#00e5ff",
                    ls="--",
                    lw=1.8,
                    label=tp_lbl,
                )

                # Directional Tactical Arrow aligned with setup type
                arrow_col = "#e74c3c" if confirmation_signal.setup_type == "SHORT" else "#27ae60"
                start_y = last_price
                ax1.annotate(
                    f"Setup [{confirmation_signal.setup_type}]\nTP: ${tp_price:,.0f}",
                    xy=(target_time, tp_price),
                    xytext=(last_time, start_y),
                    arrowprops=dict(arrowstyle="->", color=arrow_col, lw=1.8, ls="--"),
                    bbox=dict(boxstyle="round,pad=0.25", fc="#0f172a", ec=arrow_col, alpha=0.85),
                    color=arrow_col,
                    fontsize=8,
                    fontweight="bold",
                )

        ax1.set_title(f"{symbol} Price Action & {time_unit_str} Target Projection", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Price (USDT)")
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.4)

        # Volume
        vol_colors = np.where(display_df["close"] >= display_df["open"], "#27ae60", "#e74c3c")
        ax2.bar(display_df.index, display_df["volume"], color=vol_colors, alpha=0.7, width=0.03)
        ax2.set_ylabel("Volume")
        ax2.set_xlabel("Time (UTC)")
        ax2.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches="tight")
        return fig
