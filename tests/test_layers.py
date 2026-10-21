import numpy as np
import pytest

from aie.data import digits
from aie.gradcheck import numerical_gradient, relative_error
from aie.layers import (
    SGD,
    Adam,
    AvgPool2D,
    Conv2D,
    Dense,
    Flatten,
    MaxPool2D,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)
from aie.losses import softmax_cross_entropy


def _check(model, x, tol=1e-6):
    rng = np.random.default_rng(42)
    out = model.forward(x)
    R = rng.standard_normal(out.shape)  # random projection -> scalar loss

    def loss():
        return float(np.sum(model.forward(x) * R))

    model.forward(x)
    dx = model.backward(R)
    for p, _, layer, name in model.parameters():
        analytic = layer.grads[name].copy()
        assert relative_error(analytic, numerical_gradient(loss, p)) < tol, name
    assert relative_error(dx, numerical_gradient(loss, x)) < tol


@pytest.mark.parametrize("act", [Sigmoid, Tanh, ReLU])
def test_dense_stack(act):
    rng = np.random.default_rng(0)
    model = Sequential(Dense(4, 5, rng), act(), Dense(5, 3, rng))
    _check(model, rng.standard_normal((6, 4)) + 0.05)


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1)])
def test_conv(stride, padding):
    rng = np.random.default_rng(0)
    model = Sequential(Conv2D(2, 3, 3, stride=stride, padding=padding, rng=rng), Flatten())
    _check(model, rng.standard_normal((2, 2, 6, 6)))


@pytest.mark.parametrize("pool", [MaxPool2D, AvgPool2D])
def test_pooling(pool):
    rng = np.random.default_rng(0)
    model = Sequential(Conv2D(1, 2, 3, padding=1, rng=rng), pool(2), Flatten(), Dense(18, 2, rng))
    _check(model, rng.standard_normal((2, 1, 6, 6)))


@pytest.mark.parametrize("opt", [lambda m: SGD(m, 0.5, 0.9), lambda m: Adam(m, 1e-2)])
def test_small_cnn_learns_digits(opt):
    X, y = digits(n_per_class=30, seed=0)
    X = X[:, None]
    rng = np.random.default_rng(0)
    model = Sequential(
        Conv2D(1, 4, 3, padding=1, rng=rng), ReLU(), MaxPool2D(2),
        Flatten(), Dense(4 * 6 * 6, 10, rng),
    )
    o = opt(model)
    for _ in range(60):
        loss, g = softmax_cross_entropy(model.forward(X), y)
        model.backward(g)
        o.step()
    acc = np.mean(model.forward(X).argmax(1) == y)
    assert acc > 0.9, acc
