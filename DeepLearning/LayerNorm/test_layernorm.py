"""Quick CPU tests for LayerNorm (synthetic).

Run:
  python -m LayerNorm.test_layernorm

No pytest required.
"""

from __future__ import annotations

import numpy as np

from layernorm import Layer_LayerNormalization


def numerical_grad(f, x, eps=1e-6):
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


def test_forward_normalizes_per_sample():
    np.random.seed(0)
    N, D = 16, 10
    x = 2.0 * np.random.randn(N, D) + 3.0

    ln = Layer_LayerNormalization(D, epsilon=1e-5)
    y = ln.forward(x)

    # Per-sample mean≈0, var≈1 (with gamma=1, beta=0)
    mean = np.mean(y, axis=1)
    var = np.var(y, axis=1)

    assert np.allclose(mean, 0.0, atol=1e-7), mean
    assert np.allclose(var, 1.0, atol=1e-6), var


def test_backward_matches_numerical_grad_inputs_gamma_beta():
    np.random.seed(1)
    N, D = 5, 7
    x = np.random.randn(N, D)

    ln = Layer_LayerNormalization(D, epsilon=1e-5)
    ln.gamma = np.random.randn(1, D)
    ln.beta = np.random.randn(1, D)

    upstream = np.random.randn(N, D)

    def loss_wrt_x(x_):
        y = ln.forward(x_)
        return float(np.sum(y * upstream))

    ln.forward(x)
    ln.backward(upstream)

    dx_num = numerical_grad(loss_wrt_x, x.copy())
    assert np.allclose(ln.dinputs, dx_num, atol=2e-6, rtol=2e-5)

    def loss_wrt_gamma(gamma_):
        old = ln.gamma
        ln.gamma = gamma_.reshape(1, D)
        y_ = ln.forward(x)
        out = float(np.sum(y_ * upstream))
        ln.gamma = old
        return out

    dgamma_num = numerical_grad(loss_wrt_gamma, ln.gamma.copy()).reshape(1, D)
    assert np.allclose(ln.dgamma, dgamma_num, atol=2e-6, rtol=2e-5)

    def loss_wrt_beta(beta_):
        old = ln.beta
        ln.beta = beta_.reshape(1, D)
        y_ = ln.forward(x)
        out = float(np.sum(y_ * upstream))
        ln.beta = old
        return out

    dbeta_num = numerical_grad(loss_wrt_beta, ln.beta.copy()).reshape(1, D)
    assert np.allclose(ln.dbeta, dbeta_num, atol=2e-6, rtol=2e-5)


def main():
    test_forward_normalizes_per_sample()
    test_backward_matches_numerical_grad_inputs_gamma_beta()
    print("LayerNorm tests: OK")


if __name__ == "__main__":
    main()
