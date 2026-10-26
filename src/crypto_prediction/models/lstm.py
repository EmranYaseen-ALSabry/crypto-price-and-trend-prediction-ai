"""Deep learning sequence model: LSTM (with PyTorch or graceful MLP fallback)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from crypto_prediction.models.base import BaseModel
from crypto_prediction.utils.logger import get_logger

logger = get_logger("lstm_model")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class _PyTorchLSTM(nn.Module):
        """Internal PyTorch recurrent architecture."""

        def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            # Use last time step output
            last_step = out[:, -1, :]
            return self.fc(last_step).squeeze(-1)


class LSTMModel(BaseModel):
    """Deep learning sequence model for time-series forecasting.

    Utilizes PyTorch LSTM when installed, or clearly identifies as an MLP sequential regressor fallback.
    """

    def __init__(
        self,
        sequence_length: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        random_state: int = 42,
    ) -> None:
        model_name = "PyTorch-LSTM" if TORCH_AVAILABLE else "MLPRegressor (LSTM Fallback)"
        super().__init__(name=model_name)
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.torch_model: _PyTorchLSTM | None = None
        self.fallback_model = None
        self.torch_available = TORCH_AVAILABLE

    def _create_sequences(self, X: np.ndarray, y: np.ndarray | None = None):
        """Transforms 2D tabular features into 3D (n_samples, seq_len, n_features) sliding windows."""
        seq_len = self.sequence_length
        n_samples = len(X)
        if n_samples < seq_len:
            raise ValueError(f"Sample count ({n_samples}) is smaller than sequence_length ({seq_len}).")

        X_seq = []
        for i in range(seq_len, n_samples + 1):
            X_seq.append(X[i - seq_len : i])
        X_seq = np.array(X_seq, dtype=np.float32)

        if y is not None:
            y_seq = np.array(y[seq_len - 1 :], dtype=np.float32)
            return X_seq, y_seq
        return X_seq

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> LSTMModel:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_mat = X.to_numpy()
        else:
            X_mat = np.asarray(X)

        y_arr = np.asarray(y)

        # Scale features and targets strictly on training data
        X_scaled = self.scaler.fit_transform(X_mat)
        y_scaled = self.y_scaler.fit_transform(y_arr.reshape(-1, 1)).flatten()

        if TORCH_AVAILABLE:
            torch.manual_seed(self.random_state)
            X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)

            input_dim = X_seq.shape[2]
            self.torch_model = _PyTorchLSTM(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
            )

            dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq))
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()

            self.torch_model.train()
            for epoch in range(self.epochs):
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    preds = self.torch_model(batch_x)
                    loss = criterion(preds, batch_y)
                    loss.backward()
                    optimizer.step()
        else:
            logger.info("PyTorch not installed; utilizing scikit-learn MLP sequence regressor fallback.")
            from sklearn.neural_network import MLPRegressor

            self.fallback_model = MLPRegressor(
                hidden_layer_sizes=(self.hidden_dim, 16),
                max_iter=max(250, self.epochs * 15),
                random_state=self.random_state,
                alpha=0.05,
                early_stopping=True,
                n_iter_no_change=10,
            )
            self.fallback_model.fit(X_scaled, y_scaled)

        self.is_fitted = True
        return self

    def _create_padded_sequences_for_inference(self, X: np.ndarray) -> np.ndarray:
        """Creates sequences for each row in X. If fewer than seq_len bars precede a row,
        left-pads with the first observation (causal replication) to maintain strict 1-to-1 alignment."""
        seq_len = self.sequence_length
        n_samples, n_features = X.shape
        X_seq = np.empty((n_samples, seq_len, n_features), dtype=np.float32)

        for i in range(n_samples):
            if i + 1 >= seq_len:
                X_seq[i] = X[i + 1 - seq_len : i + 1]
            else:
                pad_count = seq_len - (i + 1)
                padding = np.repeat(X[0:1], pad_count, axis=0)
                X_seq[i] = np.vstack([padding, X[: i + 1]])

        return X_seq

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        warmup_history: np.ndarray | pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Generates point predictions strictly aligned 1-to-1 with input observations.

        Args:
            X: Input feature array or DataFrame for prediction.
            warmup_history: Optional preceding feature observations (e.g. from the continuous
                            timeline prior to this split) to build complete historical sequence windows
                            for the initial rows without artificial padding.

        Returns:
            1D numpy array of predictions matching len(X).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            X_mat = X.to_numpy()
        else:
            X_mat = np.asarray(X)

        n_samples = len(X_mat)
        if n_samples == 0:
            return np.array([], dtype=np.float64)

        X_scaled = self.scaler.transform(X_mat)

        if TORCH_AVAILABLE and self.torch_model is not None:
            self.torch_model.eval()
            with torch.no_grad():
                if warmup_history is not None:
                    if isinstance(warmup_history, pd.DataFrame):
                        w_mat = warmup_history.to_numpy()
                    else:
                        w_mat = np.asarray(warmup_history)
                    w_scaled = self.scaler.transform(w_mat)
                    # Take up to sequence_length - 1 rows from end of warmup
                    needed_history = w_scaled[-(self.sequence_length - 1) :]
                    combined = np.vstack([needed_history, X_scaled])
                    X_seq = self._create_padded_sequences_for_inference(combined)
                    # Suffix corresponding to X
                    X_seq = X_seq[-n_samples:]
                else:
                    X_seq = self._create_padded_sequences_for_inference(X_scaled)

                tensor_seq = torch.from_numpy(X_seq)
                preds_scaled = self.torch_model(tensor_seq).cpu().numpy().flatten()
                preds = self.y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
                return preds.astype(np.float64)
        elif self.fallback_model is not None:
            preds_scaled = self.fallback_model.predict(X_scaled)
            preds = self.y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            return preds.astype(np.float64)
        else:
            return np.zeros(n_samples, dtype=np.float64)
