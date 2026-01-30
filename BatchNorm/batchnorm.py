"""Batch Normalization (from scratch, NumPy).

This implementation is intentionally explicit and educational.
It follows the common BN formulation for 2D inputs of shape (batch, features).

- During training:
    mu = mean(x)
    var = var(x)
    x_hat = (x - mu) / sqrt(var + eps)
    y = gamma * x_hat + beta
  and we update running_mean/running_var.

- During inference:
    use running_mean/running_var.

Backward pass implements analytic gradients w.r.t. inputs, gamma, beta.

No external framework/autograd is used.
"""

from __future__ import annotations

import numpy as np


class Layer_BatchNormalization:
    """BatchNorm for dense layers: normalizes over the batch dimension.

    Expected input shape: (N, D)
    """

    def __init__(
        self,
        n_features: int,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
    ):
        self.n_features = int(n_features)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)

        # Learnable scale/shift.
        self.gamma = np.ones((1, self.n_features), dtype=np.float64)
        self.beta = np.zeros((1, self.n_features), dtype=np.float64)

        # Running statistics for inference.
        self.running_mean = np.zeros((1, self.n_features), dtype=np.float64)
        self.running_var = np.ones((1, self.n_features), dtype=np.float64)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2:
            raise ValueError(f"BatchNorm expects 2D inputs (N, D), got shape {inputs.shape}")
        if inputs.shape[1] != self.n_features:
            raise ValueError(
                f"Expected inputs with D={self.n_features} features, got {inputs.shape[1]}"
            )

        self.inputs = inputs

        if training:
            # Batch statistics.
            self.batch_mean = np.mean(inputs, axis=0, keepdims=True)
            self.batch_var = np.var(inputs, axis=0, keepdims=True)

            # Normalize.
            self.std_inv = 1.0 / np.sqrt(self.batch_var + self.epsilon)
            self.x_hat = (inputs - self.batch_mean) * self.std_inv

            # Update running stats.
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * self.batch_var
        else:
            # Inference: normalize using running stats.
            self.std_inv = 1.0 / np.sqrt(self.running_var + self.epsilon)
            self.x_hat = (inputs - self.running_mean) * self.std_inv

        self.output = self.gamma * self.x_hat + self.beta
        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        dvalues = np.asarray(dvalues, dtype=np.float64)
        if dvalues.shape != self.output.shape:
            raise ValueError(f"dvalues shape {dvalues.shape} must match output shape {self.output.shape}")

        N, D = dvalues.shape

        # Gradients for gamma/beta.
        self.dbeta = np.sum(dvalues, axis=0, keepdims=True)
        self.dgamma = np.sum(dvalues * self.x_hat, axis=0, keepdims=True)

        # Gradient w.r.t. normalized input.
        dxhat = dvalues * self.gamma

        # Backprop through normalization.
        # Using standard compact BN derivative:
        # dx = (1/N) * std_inv * (N*dxhat - sum(dxhat) - x_hat*sum(dxhat*x_hat))
        sum_dxhat = np.sum(dxhat, axis=0, keepdims=True)
        sum_dxhat_xhat = np.sum(dxhat * self.x_hat, axis=0, keepdims=True)

        self.dinputs = (1.0 / N) * self.std_inv * (N * dxhat - sum_dxhat - self.x_hat * sum_dxhat_xhat)
        return self.dinputs
