"""Layer Normalization (from scratch, NumPy).

LayerNorm normalizes per-sample across the feature dimension.
For inputs of shape (N, D):
  mu_i  = mean(x_i, axis=features)
  var_i = var(x_i, axis=features)
  xhat  = (x - mu) / sqrt(var + eps)
  y     = gamma * xhat + beta

Unlike BatchNorm, LayerNorm does NOT use running averages.

This implementation is explicit and educational.
"""

from __future__ import annotations

import numpy as np


class Layer_LayerNormalization:
    """LayerNorm for dense layers: normalizes across features per sample.

    Expected input shape: (N, D)
    """

    def __init__(self, n_features: int, epsilon: float = 1e-5):
        self.n_features = int(n_features)
        self.epsilon = float(epsilon)

        self.gamma = np.ones((1, self.n_features), dtype=np.float64)
        self.beta = np.zeros((1, self.n_features), dtype=np.float64)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2:
            raise ValueError(f"LayerNorm expects 2D inputs (N, D), got shape {inputs.shape}")
        if inputs.shape[1] != self.n_features:
            raise ValueError(
                f"Expected inputs with D={self.n_features} features, got {inputs.shape[1]}"
            )

        self.inputs = inputs

        # Per-sample mean/var across features.
        self.mean = np.mean(inputs, axis=1, keepdims=True)   # (N,1)
        self.var = np.var(inputs, axis=1, keepdims=True)     # (N,1)
        self.std_inv = 1.0 / np.sqrt(self.var + self.epsilon)

        self.x_hat = (inputs - self.mean) * self.std_inv
        self.output = self.gamma * self.x_hat + self.beta
        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        dvalues = np.asarray(dvalues, dtype=np.float64)
        if dvalues.shape != self.output.shape:
            raise ValueError(f"dvalues shape {dvalues.shape} must match output shape {self.output.shape}")

        N, D = dvalues.shape

        self.dbeta = np.sum(dvalues, axis=0, keepdims=True)
        self.dgamma = np.sum(dvalues * self.x_hat, axis=0, keepdims=True)

        dxhat = dvalues * self.gamma

        # For each sample, normalize over D features.
        # Same compact derivative as BN, but with D replacing N and per-sample stats.
        sum_dxhat = np.sum(dxhat, axis=1, keepdims=True)                   # (N,1)
        sum_dxhat_xhat = np.sum(dxhat * self.x_hat, axis=1, keepdims=True) # (N,1)

        self.dinputs = (1.0 / D) * self.std_inv * (D * dxhat - sum_dxhat - self.x_hat * sum_dxhat_xhat)
        return self.dinputs
