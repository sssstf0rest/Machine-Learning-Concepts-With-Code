# ResNet-18 (scratch, low-level PyTorch)

This folder contains a **ResNet-18 style** implementation intended for learning:
- no `torch.nn.Module`
- no `nn.Conv2d` / `nn.BatchNorm2d`
- uses only **tensors + autograd** and **`torch.nn.functional`** ops
- manual SGD updates

It trains/evaluates on **MNIST**.

## Files
- `resnet18_scratch.py` — model definition, scratch BatchNorm2d, manual SGD.
- `train_mnist.py` — training + evaluation script for MNIST.
- `test_resnet_smoke.py` — quick shape/running-stat checks (no dataset download).

## Install
```bash
python -m pip install torch torchvision
```

## Run smoke tests (no dataset)
From the repo root:
```bash
python -m ResNet.test_resnet_smoke
```

## Train on MNIST (CPU)
From the repo root:
```bash
python -m ResNet.train_mnist --epochs 1 --subset 5000 --batch-size 128 --lr 0.05
```

Notes:
- `--subset` uses only the first K training examples to keep runtime short.
- MNIST will be downloaded into `--data-dir` (default `./data`).
- The stem is adjusted for MNIST (3x3 conv, stride=1, no maxpool).
