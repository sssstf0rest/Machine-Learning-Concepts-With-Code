# Batch Normalization (from scratch)

This folder contains a **NumPy-only** BatchNorm implementation for **dense layers** (2D tensors shaped `(batch, features)`), plus quick synthetic tests.

## Files
- `batchnorm.py` — `Layer_BatchNormalization` with forward/backward.
- `test_batchnorm.py` — small CPU tests (no pytest required).

## Install
```bash
python -m pip install numpy
```

## Run tests
From the repo root:
```bash
python -m BatchNorm.test_batchnorm
```

## Notes
- Training mode uses per-batch mean/variance and updates running statistics.
- Inference mode uses `running_mean`/`running_var`.
- Backward pass follows the standard compact BN gradient formula.
