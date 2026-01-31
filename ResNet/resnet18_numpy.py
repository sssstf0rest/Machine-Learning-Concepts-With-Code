"""ResNet-18 (MNIST) using the repo's own CNN/DNN building blocks (NumPy only).

Constraints (per project requirements):
- Do NOT import torch.*
- Reuse CNN layers (Layer_Conv2D_Im2Col, Layer_MaxPool2D, Layer_Flatten)
- Reuse DNN components (Dense, activations, loss, optimizers)

This is a pedagogical implementation intended to match the style of existing
DNN/CNN code in this repo.
"""

from __future__ import annotations

import numpy as np

from CNN.layers import Layer_Conv2D_Im2Col, Layer_MaxPool2D, Layer_Flatten
from DNN.Neuron import Layer_Dense
from DNN.activation_functions import Activation_ReLU


class Layer_BatchNorm2D:
    """BatchNorm for NCHW conv activations.

    Normalizes per-channel over (N, H, W).

    For compatibility with DNN optimizers, exposes:
    - weights == gamma
    - biases == beta
    - dweights == dgamma
    - dbiases == dbeta
    """

    def __init__(self, n_channels, momentum=0.9, epsilon=1e-5):
        self.n_channels = n_channels
        self.momentum = momentum
        self.epsilon = epsilon

        self.weights = np.ones((1, n_channels, 1, 1), dtype=np.float32)  # gamma
        self.biases = np.zeros((1, n_channels, 1, 1), dtype=np.float32)  # beta

        self.running_mean = np.zeros((1, n_channels, 1, 1), dtype=np.float32)
        self.running_var = np.ones((1, n_channels, 1, 1), dtype=np.float32)

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if training:
            mu = inputs.mean(axis=(0, 2, 3), keepdims=True)
            var = inputs.var(axis=(0, 2, 3), keepdims=True)

            self.batch_mean = mu
            self.batch_var = var

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mu = self.running_mean
            var = self.running_var

        self.x_mu = inputs - mu
        self.std_inv = 1.0 / np.sqrt(var + self.epsilon)
        self.x_hat = self.x_mu * self.std_inv

        self.output = self.weights * self.x_hat + self.biases

    def backward(self, dvalues):
        # dvalues: (N, C, H, W)
        N, C, H, W = dvalues.shape
        m = N * H * W

        # dbeta, dgamma
        self.dbiases = dvalues.sum(axis=(0, 2, 3), keepdims=True)
        self.dweights = (dvalues * self.x_hat).sum(axis=(0, 2, 3), keepdims=True)

        dxhat = dvalues * self.weights

        dvar = (dxhat * self.x_mu).sum(axis=(0, 2, 3), keepdims=True) * (-0.5) * (self.std_inv ** 3)
        dmu = (dxhat * -self.std_inv).sum(axis=(0, 2, 3), keepdims=True) + dvar * (self.x_mu * -2.0).sum(
            axis=(0, 2, 3), keepdims=True
        ) / m

        self.dinputs = dxhat * self.std_inv + dvar * 2.0 * self.x_mu / m + dmu / m


class Layer_GlobalAvgPool2D:
    """Global average pooling for NCHW."""

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        self.output = inputs.mean(axis=(2, 3))  # (N, C)

    def backward(self, dvalues):
        # dvalues: (N, C)
        N, C, H, W = self.inputs_shape
        self.dinputs = dvalues[:, :, None, None] * (1.0 / (H * W))
        self.dinputs = np.broadcast_to(self.dinputs, (N, C, H, W)).astype(np.float32)


class BasicBlock:
    """ResNet BasicBlock: (conv3x3->bn->relu)->(conv3x3->bn) + shortcut -> relu"""

    def __init__(self, in_ch, out_ch, stride=1):
        self.conv1 = Layer_Conv2D_Im2Col(in_ch, out_ch, filter_size=3, stride=stride, padding=1)
        self.bn1 = Layer_BatchNorm2D(out_ch)
        self.relu1 = Activation_ReLU()

        self.conv2 = Layer_Conv2D_Im2Col(out_ch, out_ch, filter_size=3, stride=1, padding=1)
        self.bn2 = Layer_BatchNorm2D(out_ch)

        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = Layer_Conv2D_Im2Col(in_ch, out_ch, filter_size=1, stride=stride, padding=0)

        self.relu_out = Activation_ReLU()

    def forward(self, x, training=True):
        self.x_in = x

        self.conv1.forward(x)
        self.bn1.forward(self.conv1.output, training=training)
        self.relu1.forward(self.bn1.output)

        self.conv2.forward(self.relu1.output)
        self.bn2.forward(self.conv2.output, training=training)

        if self.shortcut is None:
            self.x_sc = x
        else:
            self.shortcut.forward(x)
            self.x_sc = self.shortcut.output

        self.add_out = self.bn2.output + self.x_sc
        self.relu_out.forward(self.add_out)
        self.output = self.relu_out.output

    def backward(self, dvalues):
        # Through final ReLU
        self.relu_out.backward(dvalues)
        dadd = self.relu_out.dinputs

        # Split gradient to main path and shortcut
        dbn2 = dadd
        dsc = dadd

        # Main path
        self.bn2.backward(dbn2)
        self.conv2.backward(self.bn2.dinputs)

        self.relu1.backward(self.conv2.dinputs)
        self.bn1.backward(self.relu1.dinputs)
        self.conv1.backward(self.bn1.dinputs)

        # Shortcut path
        if self.shortcut is None:
            dx_sc = dsc
        else:
            self.shortcut.backward(dsc)
            dx_sc = self.shortcut.dinputs

        # Combine
        self.dinputs = self.conv1.dinputs + dx_sc

    def params(self):
        layers = [self.conv1, self.bn1, self.conv2, self.bn2]
        if self.shortcut is not None:
            layers.append(self.shortcut)
        return layers


class ResNet18MNIST:
    """A ResNet-18 style network adapted for 1x28x28 MNIST."""

    def __init__(self, num_classes=10):
        # Stem: 3x3 conv instead of 7x7 (better for 28x28)
        self.conv1 = Layer_Conv2D_Im2Col(1, 16, filter_size=3, stride=1, padding=1)
        self.bn1 = Layer_BatchNorm2D(16)
        self.relu = Activation_ReLU()

        # Stages: [2,2,2,2] blocks, channels [16,32,64,128]
        self.layer1 = [BasicBlock(16, 16, stride=1), BasicBlock(16, 16, stride=1)]
        self.layer2 = [BasicBlock(16, 32, stride=2), BasicBlock(32, 32, stride=1)]
        self.layer3 = [BasicBlock(32, 64, stride=2), BasicBlock(64, 64, stride=1)]
        self.layer4 = [BasicBlock(64, 128, stride=2), BasicBlock(128, 128, stride=1)]

        self.gap = Layer_GlobalAvgPool2D()
        self.fc = Layer_Dense(128, num_classes)

    def forward(self, x, training=True):
        # x: (N,1,28,28)
        self.conv1.forward(x)
        self.bn1.forward(self.conv1.output, training=training)
        self.relu.forward(self.bn1.output)
        out = self.relu.output

        for blk in self.layer1:
            blk.forward(out, training=training)
            out = blk.output
        for blk in self.layer2:
            blk.forward(out, training=training)
            out = blk.output
        for blk in self.layer3:
            blk.forward(out, training=training)
            out = blk.output
        for blk in self.layer4:
            blk.forward(out, training=training)
            out = blk.output

        self.gap.forward(out)
        self.fc.forward(self.gap.output)
        self.output = self.fc.output

    def backward(self, dvalues):
        self.fc.backward(dvalues)
        self.gap.backward(self.fc.dinputs)

        dout = self.gap.dinputs
        for blk in reversed(self.layer4):
            blk.backward(dout)
            dout = blk.dinputs
        for blk in reversed(self.layer3):
            blk.backward(dout)
            dout = blk.dinputs
        for blk in reversed(self.layer2):
            blk.backward(dout)
            dout = blk.dinputs
        for blk in reversed(self.layer1):
            blk.backward(dout)
            dout = blk.dinputs

        self.relu.backward(dout)
        self.bn1.backward(self.relu.dinputs)
        self.conv1.backward(self.bn1.dinputs)
        self.dinputs = self.conv1.dinputs

    def trainable_layers(self):
        layers = [self.conv1, self.bn1]
        for blk in self.layer1 + self.layer2 + self.layer3 + self.layer4:
            layers.extend(blk.params())
        layers.append(self.fc)
        return layers
