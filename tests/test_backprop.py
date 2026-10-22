import numpy as np
import pytest

from aie.data import xor_dataset, xor_dataset_01
from aie.gradcheck import numerical_gradient, relative_error
from scripts.backprop import BackpropNetwork


@pytest.mark.parametrize("output", ["linear", "sigmoid"])
@pytest.mark.parametrize("n_out", [1, 3])
def test_gradients_match_error(n_out, output):
    rng = np.random.default_rng(0)
    X, T = rng.standard_normal((7, 2)), rng.random((7, n_out))
    net = BackpropNetwork(2, 4, n_out, seed=2, output=output)
    g1, g2, _ = net.gradients(X, T)
    assert relative_error(g1, numerical_gradient(lambda: net.error(X, T), net.W1)) < 1e-6
    assert relative_error(g2, numerical_gradient(lambda: net.error(X, T), net.W2)) < 1e-6


@pytest.mark.parametrize("seed", range(3))
def test_learns_xor(seed):
    X, T = xor_dataset()
    net = BackpropNetwork(2, 2, 1, lr=0.5, seed=seed).fit(X, T, n_epochs=5000)
    assert np.all(np.sign(net.predict(X).ravel()) == np.sign(T.ravel()))


def test_sigmoid_output_solves_xor_quickly():
    """Book Sec 4.2: sigmoidal hidden+output units solve 0/1 XOR in < 40 iterations."""
    X, T = xor_dataset_01()
    T = T.reshape(-1, 1)
    net = BackpropNetwork(2, 2, 1, lr=20.0, seed=13, weight_scale=2.0, output="sigmoid")
    net.fit(X, T, n_epochs=40)
    Y = net.predict(X)
    assert np.all((Y > 0) & (Y < 1))
    assert np.all(np.abs(Y - T) < 0.4)


def test_invalid_output_raises():
    with pytest.raises(ValueError):
        BackpropNetwork(2, 2, 1, output="softmax")
