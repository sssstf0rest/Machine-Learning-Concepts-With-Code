"""Quick CPU tests for BatchNorm (synthetic).

Run:
  python -m BatchNorm.test_batchnorm

No pytest required.
"""

from __future__ import annotations

import numpy as np

from batchnorm import Layer_BatchNormalization


def numerical_grad(f, x, eps=1e-6):
    """Finite-diff gradient for a scalar-valued function f(x)."""
    x = x.astype(np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fx1 = f(x)
        x[idx] = old - eps
        fx2 = f(x)
        x[idx] = old
        grad[idx] = (fx1 - fx2) / (2 * eps)
        it.iternext()
    return grad


def test_forward_training_normalizes_batch():
    np.random.seed(0)
    N, D = 64, 8
    x = 3.0 * np.random.randn(N, D) + 5.0

    bn = Layer_BatchNormalization(D, momentum=0.9, epsilon=1e-5)
    y = bn.forward(x, training=True)

    # After BN (with gamma=1, beta=0), mean≈0, var≈1 over batch.
    mean = np.mean(y, axis=0)
    var = np.var(y, axis=0)

    assert np.allclose(mean, 0.0, atol=1e-7), mean
    assert np.allclose(var, 1.0, atol=1e-6), var


def test_backward_matches_numerical_grad_inputs_gamma_beta():
    np.random.seed(1)
    N, D = 7, 5
    x = np.random.randn(N, D)

    bn = Layer_BatchNormalization(D, momentum=0.0, epsilon=1e-5)
    bn.gamma = np.random.randn(1, D)
    bn.beta = np.random.randn(1, D)

    # Define a simple scalar loss: L = sum(y * upstream)
    upstream = np.random.randn(N, D)

    def loss_wrt_x(x_):
        y = bn.forward(x_, training=True)
        return float(np.sum(y * upstream))

    # Forward/backward analytic.
    y = bn.forward(x, training=True)
    bn.backward(upstream)

    # Numerical grads.
    dx_num = numerical_grad(loss_wrt_x, x.copy())
    assert np.allclose(bn.dinputs, dx_num, atol=2e-6, rtol=2e-5)

    # Gamma grad numeric.
    def loss_wrt_gamma(gamma_):
        old = bn.gamma
        bn.gamma = gamma_.reshape(1, D)
        y_ = bn.forward(x, training=True)
        out = float(np.sum(y_ * upstream))
        bn.gamma = old
        return out

    dgamma_num = numerical_grad(loss_wrt_gamma, bn.gamma.copy()).reshape(1, D)
    assert np.allclose(bn.dgamma, dgamma_num, atol=2e-6, rtol=2e-5)

    # Beta grad numeric.
    def loss_wrt_beta(beta_):
        old = bn.beta
        bn.beta = beta_.reshape(1, D)
        y_ = bn.forward(x, training=True)
        out = float(np.sum(y_ * upstream))
        bn.beta = old
        return out

    dbeta_num = numerical_grad(loss_wrt_beta, bn.beta.copy()).reshape(1, D)
    assert np.allclose(bn.dbeta, dbeta_num, atol=2e-6, rtol=2e-5)


def main():
    test_forward_training_normalizes_batch()
    test_backward_matches_numerical_grad_inputs_gamma_beta()
    print("BatchNorm tests: OK")


if __name__ == "__main__":
    main()
