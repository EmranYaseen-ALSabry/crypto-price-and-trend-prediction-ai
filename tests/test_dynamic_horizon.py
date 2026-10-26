"""Unit tests for arbitrary dynamic horizon forecasting and visual target projections."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crypto_prediction.data.cleaner import DataCleaner
from crypto_prediction.data.collector import DataCollector
from crypto_prediction.features.engineer import FeatureEngineer
from crypto_prediction.utils.chart_plotter import AdvancedChartVisualizer
from crypto_prediction.validation.splitter import ChronologicalSplitter


@pytest.fixture
def ohlcv_dynamic_df() -> pd.DataFrame:
    collector = DataCollector()
    raw = collector.generate_synthetic_ohlcv(symbol="BTCUSDT", n_bars=600, seed=42)
    return DataCleaner().clean(raw)


class TestDynamicHorizonForecasting:
    """Verifies that arbitrary horizons (e.g. 17h, 36h) align targets and features causally."""

    def test_arbitrary_horizon_target_alignment(self, ohlcv_dynamic_df: pd.DataFrame) -> None:
        custom_horizon = 17
        engineer = FeatureEngineer(primary_horizon=custom_horizon)
        X, y, meta = engineer.prepare_dataset(ohlcv_dynamic_df, target_horizon=custom_horizon)

        assert f"target_return_{custom_horizon}" in meta.columns
        assert f"target_price_{custom_horizon}" in meta.columns
        assert len(X) == len(y)
        assert not X.isna().any().any()
        assert not y.isna().any()

        # Check random indices match exact formula
        close = ohlcv_dynamic_df["close"]
        for idx in [50, 100, 150]:
            expected_ret = (close.iloc[idx + custom_horizon] - close.iloc[idx]) / close.iloc[idx]
            timestamp = ohlcv_dynamic_df.index[idx]
            if timestamp in y.index:
                actual_ret = y.loc[timestamp]
                assert np.isclose(expected_ret, actual_ret, atol=1e-7)

    def test_dynamic_purging_for_long_horizon(self, ohlcv_dynamic_df: pd.DataFrame) -> None:
        custom_horizon = 24
        engineer = FeatureEngineer(primary_horizon=custom_horizon)
        X, y, meta = engineer.prepare_dataset(ohlcv_dynamic_df, target_horizon=custom_horizon)

        splitter = ChronologicalSplitter(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=custom_horizon,
        )
        splits = splitter.split(X, y, metadata=meta)

        # Gap between train and val must be > custom_horizon
        train_end_loc = X.index.get_loc(splits.X_train.index[-1])
        val_start_loc = X.index.get_loc(splits.X_val.index[0])
        assert (val_start_loc - train_end_loc) > custom_horizon

    def test_chart_visualizer_rendering(self, ohlcv_dynamic_df: pd.DataFrame, tmp_path: Path) -> None:
        visualizer = AdvancedChartVisualizer()
        html_file = tmp_path / "test_projection.html"
        png_file = tmp_path / "test_projection.png"

        fig_plotly = visualizer.create_interactive_projection_chart(
            df=ohlcv_dynamic_df,
            symbol="BTCUSDT",
            horizon=17,
            predicted_price=80500.0,
            lower_bound=79000.0,
            upper_bound=82000.0,
            save_html_path=html_file,
        )
        assert fig_plotly is not None
        assert html_file.exists()

        fig_mpl = visualizer.render_matplotlib_projection(
            df=ohlcv_dynamic_df,
            symbol="BTCUSDT",
            horizon=17,
            predicted_price=80500.0,
            lower_bound=79000.0,
            upper_bound=82000.0,
            save_path=png_file,
        )
        assert fig_mpl is not None
        assert png_file.exists()
