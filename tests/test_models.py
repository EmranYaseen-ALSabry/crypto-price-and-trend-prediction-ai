"""Unit tests for baseline, linear, tree, booster, and sequence forecasting models."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crypto_prediction.data.cleaner import DataCleaner
from crypto_prediction.features.engineer import FeatureEngineer
from crypto_prediction.models.baselines import (
    LastValueBaseline,
    MeanReturnBaseline,
    ZeroReturnBaseline,
)
from crypto_prediction.models.boosters import CatBoostModel, LightGBMModel, XGBoostModel
from crypto_prediction.models.linear import LinearRegressionModel, RidgeModel
from crypto_prediction.models.lstm import LSTMModel
from crypto_prediction.models.trees import HistGradientBoostingModel, RandomForestModel


@pytest.fixture
def train_test_data(sample_ohlcv_df: pd.DataFrame):
    cleaner = DataCleaner()
    clean_df = cleaner.clean(sample_ohlcv_df)
    engineer = FeatureEngineer(return_lags=[1, 2], rolling_windows=[6, 12], forecast_horizons=[1])
    X, y, _ = engineer.prepare_dataset(clean_df)

    n_train = int(len(X) * 0.75)
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test, y_test = X.iloc[n_train:], y.iloc[n_train:]
    return X_train, y_train, X_test, y_test


class TestForecastingModels:
    """Verifies estimator fitting, prediction shape, and persistence."""

    def test_baselines(self, train_test_data) -> None:
        X_train, y_train, X_test, y_test = train_test_data

        zero_base = ZeroReturnBaseline()
        zero_base.fit(X_train, y_train)
        preds_zero = zero_base.predict(X_test)
        assert len(preds_zero) == len(X_test)
        assert np.all(preds_zero == 0.0)

        last_base = LastValueBaseline()
        last_base.fit(X_train, y_train)
        preds_last = last_base.predict(X_test)
        assert len(preds_last) == len(X_test)
        assert np.all(preds_last == 0.0)

        mean_base = MeanReturnBaseline()
        mean_base.fit(X_train, y_train)
        preds_mean = mean_base.predict(X_test)
        assert len(preds_mean) == len(X_test)
        assert np.isclose(preds_mean[0], y_train.mean())

    def test_linear_and_ridge(self, train_test_data) -> None:
        X_train, y_train, X_test, y_test = train_test_data

        lin = LinearRegressionModel()
        lin.fit(X_train, y_train)
        preds_lin = lin.predict(X_test)
        assert len(preds_lin) == len(X_test)
        assert not np.isnan(preds_lin).any()

        ridge = RidgeModel(alpha=1.0)
        ridge.fit(X_train, y_train)
        preds_ridge = ridge.predict(X_test)
        assert len(preds_ridge) == len(X_test)
        assert not np.isnan(preds_ridge).any()

    def test_tree_ensembles(self, train_test_data) -> None:
        X_train, y_train, X_test, y_test = train_test_data

        rf = RandomForestModel(n_estimators=10, max_depth=3)
        rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_test)
        assert len(preds_rf) == len(X_test)
        assert not np.isnan(preds_rf).any()
        assert rf.get_feature_importances() is not None

        hgb = HistGradientBoostingModel(max_iter=15, max_depth=3)
        hgb.fit(X_train, y_train)
        preds_hgb = hgb.predict(X_test)
        assert len(preds_hgb) == len(X_test)
        assert not np.isnan(preds_hgb).any()

    def test_gradient_boosters(self, train_test_data) -> None:
        X_train, y_train, X_test, y_test = train_test_data

        # XGBoost
        xgb_model = XGBoostModel(n_estimators=15, max_depth=3)
        xgb_model.fit(X_train, y_train)
        preds_xgb = xgb_model.predict(X_test)
        assert len(preds_xgb) == len(X_test)
        assert not np.isnan(preds_xgb).any()

        # LightGBM
        lgb_model = LightGBMModel(n_estimators=15, max_depth=3)
        lgb_model.fit(X_train, y_train)
        preds_lgb = lgb_model.predict(X_test)
        assert len(preds_lgb) == len(X_test)
        assert not np.isnan(preds_lgb).any()

        # CatBoost
        cb_model = CatBoostModel(iterations=15, depth=3, verbose=0)
        cb_model.fit(X_train, y_train)
        preds_cb = cb_model.predict(X_test)
        assert len(preds_cb) == len(X_test)
        assert not np.isnan(preds_cb).any()

    def test_booster_split_diagnostics(self, train_test_data) -> None:
        X_train, y_train, X_test, y_test = train_test_data

        xgb_model = XGBoostModel(n_estimators=10, max_depth=3)
        xgb_model.fit(X_train, y_train)
        diag_xgb = xgb_model.get_split_diagnostics(X_test)
        assert diag_xgb["total_trees"] > 0
        assert diag_xgb["total_splits"] > 0
        assert not diag_xgb["is_degenerate_constant"]
        assert diag_xgb["prediction_variance"] > 0.0

        lgb_model = LightGBMModel(n_estimators=10, max_depth=3)
        lgb_model.fit(X_train, y_train)
        diag_lgb = lgb_model.get_split_diagnostics(X_test)
        assert diag_lgb["total_trees"] > 0
        assert diag_lgb["total_splits"] > 0
        assert not diag_lgb["is_degenerate_constant"]
        assert diag_lgb["prediction_variance"] > 0.0

    def test_booster_target_scaling_consistency(self, train_test_data) -> None:
        X_train, y_train, X_test, y_test = train_test_data

        # Fit with standardization enabled
        xgb = XGBoostModel(n_estimators=10, max_depth=3, standardize_target=True)
        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)

        # Output predictions should be in the scale of returns (mean ~ 0, range typically < 0.2)
        assert np.abs(np.mean(preds)) < 0.05
        assert np.all(np.abs(preds) < 0.5)

    def test_lstm_model_identity_and_fallback(self, train_test_data) -> None:
        X_train, y_train, X_test, y_test = train_test_data
        from crypto_prediction.models.lstm import TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            # When PyTorch is missing, default instantiation MUST fail loudly
            with pytest.raises(ImportError, match="PyTorch .* is not installed"):
                LSTMModel(sequence_length=6, hidden_dim=16, epochs=5, allow_mlp_fallback=False)

            # Explicit fallback allowed
            fallback_lstm = LSTMModel(sequence_length=6, hidden_dim=16, epochs=5, allow_mlp_fallback=True)
            assert fallback_lstm.is_genuine_lstm is False
            assert fallback_lstm.model_family == "mlp_fallback"
            assert "Fallback" in fallback_lstm.name

            fallback_lstm.fit(X_train, y_train)
            preds = fallback_lstm.predict(X_test)
            assert len(preds) == len(X_test)
            assert not np.isnan(preds).any()
        else:
            # Genuine PyTorch LSTM
            lstm = LSTMModel(sequence_length=6, hidden_dim=16, epochs=5)
            assert lstm.is_genuine_lstm is True
            assert lstm.model_family == "recurrent_lstm"
            lstm.fit(X_train, y_train)
            preds_lstm = lstm.predict(X_test)
            assert len(preds_lstm) == len(X_test)
            assert not np.isnan(preds_lstm).any()

            preds_warmup = lstm.predict(X_test, warmup_history=X_train)
            assert len(preds_warmup) == len(X_test)

    def test_lstm_legacy_artifact_detection(self, train_test_data, tmp_path: Path) -> None:
        X_train, y_train, X_test, y_test = train_test_data

        # Train with explicit fallback
        model = LSTMModel(sequence_length=4, epochs=2, allow_mlp_fallback=True)
        model.fit(X_train, y_train)

        # Save to disk
        save_file = tmp_path / "legacy_lstm.joblib"
        model.save(save_file)

        # Load through LSTMModel.load
        loaded = LSTMModel.load(save_file)
        assert loaded.is_genuine_lstm is False
        assert loaded.model_family == "mlp_fallback"

    def test_model_persistence(self, train_test_data, tmp_path: Path) -> None:
        X_train, y_train, X_test, y_test = train_test_data

        ridge = RidgeModel(alpha=2.5)
        ridge.fit(X_train, y_train)
        orig_preds = ridge.predict(X_test)

        save_file = tmp_path / "ridge_test.joblib"
        ridge.save(save_file)
        assert save_file.exists()

        loaded_ridge = RidgeModel.load(save_file)
        loaded_preds = loaded_ridge.predict(X_test)

        assert np.allclose(orig_preds, loaded_preds)
