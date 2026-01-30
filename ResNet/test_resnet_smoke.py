"""Smoke tests for the scratch ResNet.

Run:
  python -m ResNet.test_resnet_smoke

Requires: torch
"""

from __future__ import annotations

import torch

from .resnet18_scratch import ResNet18Scratch


def test_forward_shape_and_running_stats_update():
    device = torch.device("cpu")
    model = ResNet18Scratch(num_classes=10, in_channels=1, device=device)

    x = torch.randn(4, 1, 28, 28, device=device)

    # Training forward should update BN running stats.
    rm_before = model.stem_bn.state.running_mean.clone()
    rv_before = model.stem_bn.state.running_var.clone()

    logits = model.forward(x, training=True)
    assert logits.shape == (4, 10)

    rm_after = model.stem_bn.state.running_mean
    rv_after = model.stem_bn.state.running_var

    assert not torch.allclose(rm_before, rm_after)
    assert not torch.allclose(rv_before, rv_after)

    # Inference forward should keep running stats unchanged.
    rm_before2 = rm_after.clone()
    rv_before2 = rv_after.clone()
    _ = model.forward(x, training=False)
    assert torch.allclose(rm_before2, model.stem_bn.state.running_mean)
    assert torch.allclose(rv_before2, model.stem_bn.state.running_var)


def main():
    test_forward_shape_and_running_stats_update()
    print("ResNet smoke tests: OK")


if __name__ == "__main__":
    main()
