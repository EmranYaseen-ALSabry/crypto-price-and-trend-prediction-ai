"""Publication-grade visualization utilities using Matplotlib and Plotly."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_price_and_volume(
    df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots historical close price and trading volume time series."""
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(df.index, df["close"], color="#1f77b4", linewidth=1.5, label=f"{symbol} Close Price")
    ax1.set_title(f"{symbol} Historical Price & Volume (OHLCV)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price (USDT)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    colors = np.where(df["close"] >= df["open"], "#2ca02c", "#d62728")
    ax2.bar(df.index, df["volume"], color=colors, width=0.03, alpha=0.8, label="Volume")
    ax2.set_ylabel("Volume", fontsize=11)
    ax2.set_xlabel("Timestamp", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_actual_vs_predicted(
    timestamps: pd.DatetimeIndex,
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    model_name: str,
    target_type: str = "Return",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots actual values versus out-of-sample model predictions."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(timestamps, actual, label="Actual Ground Truth", color="#2c3e50", linewidth=1.5, alpha=0.85)
    ax.plot(
        timestamps,
        predicted,
        label=f"Predicted ({model_name})",
        color="#e74c3c",
        linewidth=1.5,
        linestyle="--",
        alpha=0.9,
    )

    ax.set_title(f"Out-of-Sample Evaluation: Actual vs {model_name} ({target_type})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel(target_type, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_residuals_distribution(
    residuals: np.ndarray | pd.Series,
    model_name: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots residual distribution and diagnostic histogram."""
    residuals = np.asarray(residuals)
    residuals_clean = residuals[~np.isnan(residuals)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram & KDE
    ax1.hist(residuals_clean, bins=50, color="#3498db", edgecolor="black", alpha=0.7, density=True)
    ax1.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Bias")
    ax1.set_title(f"{model_name} Residual Distribution", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Error (Actual - Predicted)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Residuals over time / sample index
    ax2.scatter(range(len(residuals_clean)), residuals_clean, alpha=0.4, color="#9b59b6", s=10)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.set_title(f"{model_name} Residuals Across Samples", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Sample Index", fontsize=11)
    ax2.set_ylabel("Residual", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    model_name: str,
    top_n: int = 20,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots top N feature importances for tree-based models."""
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices][::-1]
    top_scores = importances[indices][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, int(top_n * 0.3))))
    bars = ax.barh(range(len(top_scores)), top_scores, color="#2ecc71", edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(top_scores)))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances - {model_name}", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    metric: str = "rmse",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots comparative bar chart across all evaluated models."""
    sorted_df = comparison_df.sort_values(by=metric, ascending=(metric in ["rmse", "mae"]))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        sorted_df["model_name"],
        sorted_df[metric],
        color="#34495e",
        edgecolor="#1abc9c",
        linewidth=1.2,
        alpha=0.85,
    )

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.5f}" if abs(height) < 1 else f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(f"Model Comparison - Out-of-Sample {metric.upper()}", fontsize=14, fontweight="bold")
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_xticks(range(len(sorted_df)))
    ax.set_xticklabels(sorted_df["model_name"], rotation=30, ha="right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
