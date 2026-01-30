"""Train ResNet-18 style (scratch) on MNIST.

Run (from repo root):
  python -m ResNet.train_mnist --epochs 1 --subset 5000

This is intended to run quickly on CPU for demonstration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset

try:
    from torchvision import datasets, transforms
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "torchvision is required for MNIST in this demo. Install: pip install torchvision"
    ) from e

from .resnet18_scratch import ResNet18Scratch, set_seed, sgd_step, accuracy


@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 1
    batch_size: int = 128
    lr: float = 0.05
    weight_decay: float = 0.0
    subset: int = 5000
    data_dir: str = "./data"
    device: str = "cpu"
    log_every: int = 50


def get_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=tfm)

    if cfg.subset and cfg.subset < len(train_ds):
        train_ds = Subset(train_ds, list(range(cfg.subset)))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: ResNet18Scratch, loader: DataLoader, device: torch.device) -> float:
    model_acc = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model.forward(x, training=False)
        model_acc += accuracy(logits, y) * x.shape[0]
        n += x.shape[0]
    return model_acc / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--subset", type=int, default=5000)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()

    cfg = TrainConfig(**vars(args))

    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    train_loader, test_loader = get_loaders(cfg)

    model = ResNet18Scratch(num_classes=10, in_channels=1, device=device)
    params = model.parameters()

    for epoch in range(1, cfg.epochs + 1):
        model_loss = 0.0
        model_acc = 0.0
        n = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)

            logits = model.forward(x, training=True)
            loss = torch.nn.functional.cross_entropy(logits, y)

            model.zero_grad()
            loss.backward()
            sgd_step(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

            with torch.no_grad():
                bs = x.shape[0]
                model_loss += float(loss.item()) * bs
                model_acc += accuracy(logits, y) * bs
                n += bs

            if cfg.log_every and step % cfg.log_every == 0:
                print(
                    f"epoch {epoch}/{cfg.epochs} step {step}/{len(train_loader)} "
                    f"loss={model_loss/n:.4f} acc={model_acc/n:.4f}"
                )

        train_loss = model_loss / n
        train_acc = model_acc / n
        test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
