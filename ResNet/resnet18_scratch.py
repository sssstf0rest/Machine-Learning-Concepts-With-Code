"""ResNet-18 style model (MNIST) using low-level PyTorch ops.

Goal: keep the code educational and explicit.
- No torch.nn.Module / nn.Conv2d / nn.BatchNorm2d usage.
- We use:
    * torch tensors with requires_grad=True
    * torch.nn.functional for conv2d / relu / pooling / cross_entropy
    * manual SGD parameter updates

This is NOT intended to be a full framework; it's a readable reference.

Architecture: ResNet-18 basic blocks.
For MNIST (1x28x28), we use a smaller stem:
  conv3x3(1->64, stride=1) + BN + ReLU
(no maxpool), then standard 4 stages with [2,2,2,2] blocks.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kaiming_normal_(w: torch.Tensor) -> torch.Tensor:
    """Kaiming normal init for ReLU networks."""
    fan_in = w.shape[1] * w.shape[2] * w.shape[3]
    std = (2.0 / fan_in) ** 0.5
    with torch.no_grad():
        w.normal_(0.0, std)
    return w


@dataclass
class BN2dState:
    running_mean: torch.Tensor  # (C,)
    running_var: torch.Tensor   # (C,)


class BatchNorm2dScratch:
    """BatchNorm2d implemented with explicit running stats.

    Input expected: (N, C, H, W)

    Learnable params:
      gamma (C,), beta (C,)

    Running stats are buffers (no grad).
    """

    def __init__(self, num_channels: int, momentum: float = 0.9, eps: float = 1e-5, device=None):
        self.C = int(num_channels)
        self.momentum = float(momentum)
        self.eps = float(eps)
        device = device or torch.device("cpu")

        self.gamma = torch.ones(self.C, device=device, dtype=torch.float32, requires_grad=True)
        self.beta = torch.zeros(self.C, device=device, dtype=torch.float32, requires_grad=True)

        self.state = BN2dState(
            running_mean=torch.zeros(self.C, device=device, dtype=torch.float32),
            running_var=torch.ones(self.C, device=device, dtype=torch.float32),
        )

    def parameters(self) -> List[torch.Tensor]:
        return [self.gamma, self.beta]

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"BN2d expects NCHW tensor, got shape {tuple(x.shape)}")

        if training:
            # Reduce over N,H,W.
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)

            with torch.no_grad():
                self.state.running_mean.mul_(self.momentum).add_((1.0 - self.momentum) * mean)
                self.state.running_var.mul_(self.momentum).add_((1.0 - self.momentum) * var)
        else:
            mean = self.state.running_mean
            var = self.state.running_var

        xhat = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        y = self.gamma[None, :, None, None] * xhat + self.beta[None, :, None, None]
        return y


def conv2d_param(
    in_ch: int,
    out_ch: int,
    k: int,
    stride: int = 1,
    padding: int = 0,
    device=None,
) -> Dict[str, object]:
    device = device or torch.device("cpu")
    w = torch.empty(out_ch, in_ch, k, k, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(out_ch, device=device, dtype=torch.float32, requires_grad=True)
    kaiming_normal_(w)
    return {"w": w, "b": b, "stride": stride, "padding": padding}


def conv2d_forward(x: torch.Tensor, p: Dict[str, object]) -> torch.Tensor:
    return F.conv2d(x, p["w"], p["b"], stride=p["stride"], padding=p["padding"])


class BasicBlockScratch:
    """ResNet basic block (two 3x3 convs)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, device=None):
        self.conv1 = conv2d_param(in_ch, out_ch, k=3, stride=stride, padding=1, device=device)
        self.bn1 = BatchNorm2dScratch(out_ch, device=device)
        self.conv2 = conv2d_param(out_ch, out_ch, k=3, stride=1, padding=1, device=device)
        self.bn2 = BatchNorm2dScratch(out_ch, device=device)

        self.use_proj = (stride != 1) or (in_ch != out_ch)
        if self.use_proj:
            self.proj = conv2d_param(in_ch, out_ch, k=1, stride=stride, padding=0, device=device)
            self.bn_proj = BatchNorm2dScratch(out_ch, device=device)
        else:
            self.proj = None
            self.bn_proj = None

    def parameters(self) -> List[torch.Tensor]:
        ps: List[torch.Tensor] = []
        for c in [self.conv1, self.conv2]:
            ps.extend([c["w"], c["b"]])
        ps.extend(self.bn1.parameters())
        ps.extend(self.bn2.parameters())
        if self.use_proj:
            ps.extend([self.proj["w"], self.proj["b"]])
            ps.extend(self.bn_proj.parameters())
        return ps

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        out = conv2d_forward(x, self.conv1)
        out = self.bn1.forward(out, training=training)
        out = F.relu(out)

        out = conv2d_forward(out, self.conv2)
        out = self.bn2.forward(out, training=training)

        if self.use_proj:
            shortcut = conv2d_forward(x, self.proj)
            shortcut = self.bn_proj.forward(shortcut, training=training)
        else:
            shortcut = x

        out = F.relu(out + shortcut)
        return out


class ResNet18Scratch:
    def __init__(self, num_classes: int = 10, in_channels: int = 1, device=None):
        device = device or torch.device("cpu")
        self.device = device

        # MNIST-friendly stem (no 7x7, no maxpool)
        self.stem_conv = conv2d_param(in_channels, 64, k=3, stride=1, padding=1, device=device)
        self.stem_bn = BatchNorm2dScratch(64, device=device)

        # Stages: (out_channels, num_blocks, stride_for_first_block)
        self.layer1 = [BasicBlockScratch(64, 64, stride=1, device=device) for _ in range(2)]
        self.layer2 = [
            BasicBlockScratch(64, 128, stride=2, device=device),
            BasicBlockScratch(128, 128, stride=1, device=device),
        ]
        self.layer3 = [
            BasicBlockScratch(128, 256, stride=2, device=device),
            BasicBlockScratch(256, 256, stride=1, device=device),
        ]
        self.layer4 = [
            BasicBlockScratch(256, 512, stride=2, device=device),
            BasicBlockScratch(512, 512, stride=1, device=device),
        ]

        # Classifier (global average pool -> linear)
        self.fc_w = torch.empty(512, num_classes, device=device, dtype=torch.float32, requires_grad=True)
        self.fc_b = torch.zeros(num_classes, device=device, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            self.fc_w.normal_(0.0, 0.01)

    def parameters(self) -> List[torch.Tensor]:
        ps: List[torch.Tensor] = [self.stem_conv["w"], self.stem_conv["b"], self.fc_w, self.fc_b]
        ps.extend(self.stem_bn.parameters())
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                ps.extend(block.parameters())
        return ps

    def zero_grad(self) -> None:
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = conv2d_forward(x, self.stem_conv)
        x = self.stem_bn.forward(x, training=training)
        x = F.relu(x)

        for block in self.layer1:
            x = block.forward(x, training=training)
        for block in self.layer2:
            x = block.forward(x, training=training)
        for block in self.layer3:
            x = block.forward(x, training=training)
        for block in self.layer4:
            x = block.forward(x, training=training)

        # Global average pooling over H,W.
        x = x.mean(dim=(2, 3))  # (N, 512)
        logits = x @ self.fc_w + self.fc_b
        return logits


def sgd_step(params: List[torch.Tensor], lr: float, weight_decay: float = 0.0) -> None:
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            if weight_decay != 0.0:
                p.grad.add_(weight_decay * p)
            p.add_(-lr * p.grad)


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())
