import numpy as np

from ResNet18 import ResNet18MNIST


def test_forward_shapes():
    np.random.seed(0)
    model = ResNet18MNIST(num_classes=10)
    x = np.random.randn(2, 1, 28, 28).astype(np.float32)
    model.forward(x, training=True)
    assert model.output.shape == (2, 10)


def test_bn_running_stats_update():
    np.random.seed(0)
    model = ResNet18MNIST(num_classes=10)
    x = np.random.randn(8, 1, 28, 28).astype(np.float32)

    # snapshot
    rm0 = model.bn1.running_mean.copy()
    rv0 = model.bn1.running_var.copy()

    model.forward(x, training=True)
    rm1 = model.bn1.running_mean.copy()
    rv1 = model.bn1.running_var.copy()

    assert np.any(np.abs(rm1 - rm0) > 0)
    assert np.any(np.abs(rv1 - rv0) > 0)

    # eval should NOT update
    model.forward(x, training=False)
    rm2 = model.bn1.running_mean.copy()
    rv2 = model.bn1.running_var.copy()

    assert np.allclose(rm2, rm1)
    assert np.allclose(rv2, rv1)


if __name__ == '__main__':
    test_forward_shapes()
    test_bn_running_stats_update()
    print('ResNet smoke tests passed.')
