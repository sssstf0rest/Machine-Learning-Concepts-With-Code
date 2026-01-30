# Layer Normalization (from scratch)

This folder contains a **NumPy-only** LayerNorm implementation for **dense layers** (2D tensors shaped `(batch, features)`), plus quick synthetic tests.

## Files
- `layernorm.py` — `Layer_LayerNormalization` with forward/backward.
- `test_layernorm.py` — small CPU tests (no pytest required).

## Install
```bash
python -m pip install numpy
```

## Run tests
From the repo root:
```bash
python -m LayerNorm.test_layernorm
```

## Notes
- LayerNorm normalizes **per sample** across the feature dimension.
- There are **no running averages** (unlike BatchNorm).
