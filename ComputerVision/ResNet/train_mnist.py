"""Train ResNet-18 (NumPy) on MNIST.

Uses:
- CNN.layers (Conv/Flatten)
- DNN.* (Dense, activations, loss, optimizers)

Run (CPU):
  python3 -m ResNet.train_mnist --epochs 1 --subset 5000 --batch-size 64
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
import sys

# Resolve repo root (…/Machine-Learning-Concepts-With-Code)
repo_root = Path.cwd().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from DeepLearning.DNN.activation_functions import Activation_Softmax_Loss_CategoricalCrossentropy
from DeepLearning.DNN.optimizers import Optimizer_Adam

from ComputerVision.ResNet.mnist_data import load_mnist
from ComputerVision.ResNet.ResNet18 import ResNet18MNIST


def iterate_minibatches(X, y, batch_size, shuffle=True, seed=0):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


def accuracy(pred_probs, y_true):
    y_pred = np.argmax(pred_probs, axis=1)
    return (y_pred == y_true).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--subset', type=int, default=0, help='Use first N training samples (0 = full)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    (x_train, y_train), (x_test, y_test) = load_mnist()

    if args.subset and args.subset > 0:
        x_train = x_train[: args.subset]
        y_train = y_train[: args.subset]

    model = ResNet18MNIST(num_classes=10)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_Adam(learning_rate=args.lr)

    trainable = model.trainable_layers()

    for epoch in range(1, args.epochs + 1):
        optimizer.pre_update_params()

        losses = []
        accs = []

        for xb, yb in iterate_minibatches(x_train, y_train, args.batch_size, shuffle=True, seed=args.seed + epoch):
            model.forward(xb, training=True)
            loss = loss_activation.forward(model.output, yb)

            # Accuracy
            acc = accuracy(loss_activation.output, yb)

            # Backward
            loss_activation.backward(loss_activation.output, yb)
            model.backward(loss_activation.dinputs)

            # Update
            for layer in trainable:
                if hasattr(layer, 'weights') and hasattr(layer, 'dweights'):
                    optimizer.update_params(layer)

            losses.append(loss)
            accs.append(acc)

        optimizer.post_update_params()

        # Eval (quick): run on first 2000 test samples
        model.forward(x_test[:2000], training=False)
        loss_activation.activation.forward(model.output)
        test_acc = accuracy(loss_activation.activation.output, y_test[:2000])

        print(f"Epoch {epoch}: loss={float(np.mean(losses)):.4f} acc={float(np.mean(accs)):.4f} test_acc@2k={float(test_acc):.4f}")


if __name__ == '__main__':
    main()
