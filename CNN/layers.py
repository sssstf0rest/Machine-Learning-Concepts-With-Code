"""CNN layers extracted from CNN/playground.ipynb

These are the reference implementations used across this repo.
They follow the same API style as the DNN layers:
- forward(inputs) populates .output
- backward(dvalues) populates .dinputs (and parameter grads if any)

NCHW convention:
inputs: (N, C, H, W)

This file is intentionally dependency-light (NumPy only).
"""

import numpy as np


# ---------------- im2col / col2im utilities (NCHW) ----------------

def get_im2col_indices(x_shape, field_height, field_width, padding, stride):
    N, C, H, W = x_shape

    # Allow non-divisible sizes by using floor output sizing.
    # This matches common conv behavior and avoids assertion errors
    # for inputs like 28x28 with K=3, P=1, S=2.
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)

    j0 = np.tile(np.arange(field_width), field_height)
    j0 = np.tile(j0, C)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j, out_height, out_width)


def im2col(x, field_height, field_width, padding, stride):
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    k, i, j, out_height, out_width = get_im2col_indices(
        x.shape, field_height, field_width, padding, stride
    )

    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * x.shape[1], -1)
    return cols, out_height, out_width


def col2im(cols, x_shape, field_height, field_width, padding, stride):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    k, i, j, out_height, out_width = get_im2col_indices(
        x_shape, field_height, field_width, padding, stride
    )

    cols_reshaped = cols.reshape(C * field_height * field_width, out_height * out_width, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


# ---------------- Conv2D via im2col ----------------

class Layer_Conv2D_Im2Col:
    """2D Convolution layer (NCHW) using im2col."""

    def __init__(
        self,
        n_input_channels,
        n_filters,
        filter_size,
        stride=1,
        padding=0,
        weight_regularizer_l1=0,
        weight_regularizer_l2=0,
        bias_regularizer_l1=0,
        bias_regularizer_l2=0,
    ):
        self.n_input_channels = n_input_channels
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        self.weights = 0.01 * np.random.randn(
            n_filters, n_input_channels, filter_size, filter_size
        ).astype(np.float32)
        self.biases = np.zeros((1, n_filters), dtype=np.float32)

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        N, C, H, W = inputs.shape
        K = self.filter_size
        S = self.stride
        P = self.padding

        assert C == self.n_input_channels, f"Expected C={self.n_input_channels}, got {C}"

        X_col, H_out, W_out = im2col(inputs, K, K, P, S)
        self.X_col = X_col
        self.H_out = H_out
        self.W_out = W_out

        W_col = self.weights.reshape(self.n_filters, -1)
        self.W_col = W_col

        out_col = W_col @ X_col + self.biases.reshape(-1, 1)
        out = out_col.reshape(self.n_filters, H_out, W_out, N).transpose(3, 0, 1, 2)
        self.output = out

    def backward(self, dvalues):
        N, F, H_out, W_out = dvalues.shape
        K = self.filter_size
        S = self.stride
        P = self.padding

        self.dbiases = np.sum(dvalues, axis=(0, 2, 3)).reshape(1, -1).astype(np.float32)

        dout_col = dvalues.transpose(1, 2, 3, 0).reshape(F, -1)
        dW_col = dout_col @ self.X_col.T
        self.dweights = dW_col.reshape(self.weights.shape).astype(np.float32)

        dX_col = self.W_col.T @ dout_col
        self.dinputs = col2im(dX_col, self.inputs.shape, K, K, P, S).astype(np.float32)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases


class Layer_MaxPool2D:
    """MaxPool2D (NCHW)."""

    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

    def forward(self, inputs):
        self.inputs = inputs
        N, C, H, W = inputs.shape
        P = self.pool_size
        S = self.stride

        H_out = (H - P) // S + 1
        W_out = (W - P) // S + 1

        output = np.zeros((N, C, H_out, W_out), dtype=inputs.dtype)

        self.argmax_h = np.zeros((N, C, H_out, W_out), dtype=np.int32)
        self.argmax_w = np.zeros((N, C, H_out, W_out), dtype=np.int32)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    h_start = i * S
                    h_end = h_start + P
                    for j in range(W_out):
                        w_start = j * S
                        w_end = w_start + P

                        window = inputs[n, c, h_start:h_end, w_start:w_end]
                        idx = np.argmax(window)
                        ih, iw = np.unravel_index(idx, window.shape)

                        output[n, c, i, j] = window[ih, iw]
                        self.argmax_h[n, c, i, j] = ih
                        self.argmax_w[n, c, i, j] = iw

        self.output = output

    def backward(self, dvalues):
        N, C, H, W = self.inputs.shape
        P = self.pool_size
        S = self.stride
        _, _, H_out, W_out = dvalues.shape

        self.dinputs = np.zeros_like(self.inputs)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    h_start = i * S
                    for j in range(W_out):
                        w_start = j * S

                        ih = self.argmax_h[n, c, i, j]
                        iw = self.argmax_w[n, c, i, j]

                        self.dinputs[n, c, h_start + ih, w_start + iw] += dvalues[n, c, i, j]


class Layer_Flatten:
    """Flatten: (N, C, H, W) -> (N, C*H*W)."""

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        self.output = inputs.reshape(inputs.shape[0], -1)

    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.inputs_shape)
