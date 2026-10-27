"""Unit and causality tests for CrossAssetFeatureEngineer:
- Feature computation and presence
- Scale invariance and mathematical bounds
- Target isolation
- Mandatory Future Perturbation Causality Test
"""

import numpy as np
import pandas as pd
import pytest

from crypto_prediction.data.cleaner import DataCleaner
from crypto_prediction.data.collector import DataCollector
from crypto_prediction.features.cross_asset import CrossAssetFeatureEngineer
from crypto_prediction.validation.leakage_guard import LeakageGuard


@pytest.fixture
def synchronized_3asset_panel() -> dict[str, pd.DataFrame]:
    """Loads 400 synchronized hourly bars for BTC, ETH, and SOL."""
    collector = DataCollector()
    cleaner = DataCleaner()
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    datasets = {}

    for sym in symbols:
        sample_path = Path("data/sample") / f"{sym}_1h.csv"
        if sample_path.exists():
            df = collector.load_from_csv(sample_path).iloc[:400].copy()
        else:
            df = collector.generate_synthetic_ohlcv(symbol=sym, n_bars=400)
        datasets[sym] = cleaner.clean(df)

    # Align common index
    common_idx = datasets["BTCUSDT"].index
    for sym in symbols[1:]:
        common_idx = common_idx.intersection(datasets[sym].index)
    common_idx = common_idx.sort_values()

    return {sym: datasets[sym].loc[common_idx].copy() for sym in symbols}


from pathlib import Path


class TestCrossAssetFeatureEngineer:
    """Verifies cross-asset feature generation, bounds, scale-invariance, and causality."""

    def test_cross_asset_feature_generation_and_names(self, synchronized_3asset_panel):
        engineer = CrossAssetFeatureEngineer(
            anchor_symbols=["BTCUSDT", "ETHUSDT"],
            include_technicals=True,
            include_patterns=False,
            include_market_aggregates=True,
        )

        matrices = engineer.compute_cross_asset_panel(synchronized_3asset_panel)
        assert len(matrices) == 3

        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            X = matrices[sym]
            assert not X.empty
            assert len(X) == len(synchronized_3asset_panel[sym])

            # 1. Asset-specific features exist
            assert "return_1h" in X.columns
            assert "volatility_24h" in X.columns
            assert "volatility_parkinson_24h" in X.columns
            assert "candle_hl_spread" in X.columns
            assert "candle_upper_shadow" in X.columns
            assert "rsi_14" in X.columns
            assert "macd_hist" in X.columns

            # 2. Peer lagged returns exist
            assert "btc_return_lag_1" in X.columns
            assert "btc_return_lag_2" in X.columns
            assert "eth_return_lag_1" in X.columns
            assert "sol_return_lag_1" in X.columns

            # 3. Market aggregates exist
            assert "market_mean_return_1h" in X.columns
            assert "market_return_dispersion_1h" in X.columns
            assert "market_mean_volatility_24h" in X.columns

            # 4. Cross-asset pairwise metrics
            if sym != "BTCUSDT":
                assert "cross_corr_btc_24h" in X.columns
                assert "cross_vol_ratio_btc_24h" in X.columns
                assert "cross_mom_spread_btc_24h" in X.columns

            # 5. Scale invariance: raw unnormalized prices must NEVER be feature columns
            for raw_col in ["open", "high", "low", "close", "volume"]:
                assert raw_col not in X.columns

    def test_mathematical_bounds_and_validity(self, synchronized_3asset_panel):
        engineer = CrossAssetFeatureEngineer(include_technicals=True)
        matrices = engineer.compute_cross_asset_panel(synchronized_3asset_panel)

        for sym in ["ETHUSDT", "SOLUSDT"]:
            X = matrices[sym]

            # Correlation must be in [-1.0, 1.0]
            corr = X["cross_corr_btc_24h"].dropna()
            assert (corr >= -1.0 - 1e-6).all()
            assert (corr <= 1.0 + 1e-6).all()

            # Volatility ratio must be strictly non-negative
            vol_ratio = X["cross_vol_ratio_btc_24h"].dropna()
            assert (vol_ratio >= 0.0).all()

            # Volatility Parkinson must be positive
            parkinson = X["volatility_parkinson_24h"].dropna()
            assert (parkinson >= 0.0).all()

            # RSI bounded [0, 100]
            rsi = X["rsi_14"].dropna()
            assert (rsi >= 0.0).all()
            assert (rsi <= 100.0).all()

    def test_target_isolation_cleanliness(self, synchronized_3asset_panel):
        engineer = CrossAssetFeatureEngineer(forecast_horizons=[1, 4, 24], primary_horizon=1)
        prepared = engineer.prepare_dataset(synchronized_3asset_panel, target_horizon=1, drop_warmup=True)

        for sym, data in prepared.items():
            X = data["X"]
            y = data["y"]

            passed, msg = LeakageGuard.verify_target_isolation(list(X.columns))
            assert passed is True, f"Target isolation failure in {sym}: {msg}"

            # y must not be NaN
            assert not y.isna().any()
            # X must not be NaN after warmup drop
            assert not X.isna().any().any()
            assert len(X) == len(y)

    def test_mandatory_future_perturbation_causality(self, synchronized_3asset_panel):
        """MANDATORY CAUSALITY TEST:
        Shocks future observations of BTC, ETH, and SOL at rows > cutoff_idx.
        Asserts that every feature for EVERY asset at rows <= cutoff_idx remains IDENTICAL (< 1e-10 diff).
        """
        engineer = CrossAssetFeatureEngineer(
            anchor_symbols=["BTCUSDT", "ETHUSDT"],
            include_technicals=True,
            include_market_aggregates=True,
        )

        cutoff_idx = 250
        n_bars = len(synchronized_3asset_panel["BTCUSDT"])
        assert n_bars > cutoff_idx + 20

        # 1. Compute unperturbed features
        orig_matrices = engineer.compute_cross_asset_panel(synchronized_3asset_panel)

        # 2. Perturb future raw prices for ALL assets at rows > cutoff_idx
        perturbed_panel = {}
        future_mask = np.arange(n_bars) > cutoff_idx

        for sym, df in synchronized_3asset_panel.items():
            df_pert = df.copy()
            # Heavily shock future prices and volume
            for col in ["open", "high", "low", "close"]:
                df_pert.loc[future_mask, col] = df_pert.loc[future_mask, col] * 3.5
            df_pert.loc[future_mask, "volume"] = df_pert.loc[future_mask, "volume"] * 10.0

            df_pert["high"] = df_pert[["high", "open", "close"]].max(axis=1)
            df_pert["low"] = df_pert[["low", "open", "close"]].min(axis=1)
            perturbed_panel[sym] = df_pert

        # 3. Compute features on perturbed multi-asset panel
        pert_matrices = engineer.compute_cross_asset_panel(perturbed_panel)

        # 4. Assert strict invariance across all assets and all features up to cutoff
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            orig_X = orig_matrices[sym].iloc[: cutoff_idx + 1]
            pert_X = pert_matrices[sym].iloc[: cutoff_idx + 1]

            # Compare non-NaN elements
            valid_mask = (~orig_X.isna()) & (~pert_X.isna())
            diffs = (orig_X.to_numpy()[valid_mask.to_numpy()] - pert_X.to_numpy()[valid_mask.to_numpy()])
            max_abs_diff = float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0.0

            assert max_abs_diff < 1e-10, (
                f"FUTURE DATA LEAKAGE DETECTED in {sym}! "
                f"Max absolute difference at <= {cutoff_idx} is {max_abs_diff:.2e}"
            )
