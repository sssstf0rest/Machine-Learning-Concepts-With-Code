# ResNet (from scratch, NumPy)

This folder contains a **ResNet-18 style** implementation for MNIST that follows the conventions in this repo:
- Uses **CNN** building blocks: `Layer_Conv2D_Im2Col`, `Layer_Flatten`, etc.
- Uses **DNN** building blocks: Dense/activations/loss/optimizers
- **No torch / no PyTorch / no high-level DL framework**

## Files
- `resnet18_numpy.py` — ResNet-18 style network + BatchNorm2D + GlobalAvgPool
- `mnist_data.py` — MNIST downloader/loader (NumPy only)
- `train_mnist.py` — training script
- `test_resnet_smoke.py` — quick sanity checks
- `playground.ipynb` — step-by-step explanation

## Quick start

### 1) Smoke test
```bash
python3 -m ResNet.test_resnet_smoke
```

### 2) Train (fast CPU run)
```bash
python3 -m ResNet.train_mnist --epochs 1 --subset 5000 --batch-size 64 --lr 1e-3
```

Notes:
- This is a learning-first implementation. It is not optimized for speed.
- Full MNIST training with ResNet-18 in pure NumPy may be slow; use `--subset` for experiments.
