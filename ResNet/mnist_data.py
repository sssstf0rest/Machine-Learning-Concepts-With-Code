"""MNIST loader (NumPy only).

Downloads `mnist.npz` (Keras public mirror) if not present.
No torch/tensorflow dependency.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def load_mnist(root: str | os.PathLike = None):
    root = Path(root or Path(__file__).resolve().parent / "data")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "mnist.npz"

    if not path.exists():
        print(f"Downloading MNIST to {path} ...")
        urllib.request.urlretrieve(MNIST_URL, path)

    with np.load(path) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    # Normalize to [0,1], add channel dim, NCHW
    x_train = (x_train.astype(np.float32) / 255.0)[:, None, :, :]
    x_test = (x_test.astype(np.float32) / 255.0)[:, None, :, :]

    return (x_train, y_train.astype(np.int64)), (x_test, y_test.astype(np.int64))
