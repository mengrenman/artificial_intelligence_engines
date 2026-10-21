import numpy as np
import pytest

from aie.data import xor_dataset
from aie.gradcheck import numerical_gradient, relative_error
from scripts.backprop import BackpropNetwork


@pytest.mark.parametrize("n_out", [1, 3])
def test_gradients_match_error(n_out):
    rng = np.random.default_rng(0)
    X, T = rng.standard_normal((7, 2)), rng.standard_normal((7, n_out))
    net = BackpropNetwork(2, 4, n_out, seed=2)
    g1, g2, _ = net.gradients(X, T)
    assert relative_error(g1, numerical_gradient(lambda: net.error(X, T), net.W1)) < 1e-6
    assert relative_error(g2, numerical_gradient(lambda: net.error(X, T), net.W2)) < 1e-6


@pytest.mark.parametrize("seed", range(3))
def test_learns_xor(seed):
    X, T = xor_dataset()
    net = BackpropNetwork(2, 2, 1, lr=0.5, seed=seed).fit(X, T, n_epochs=5000)
    assert np.all(np.sign(net.predict(X).ravel()) == np.sign(T.ravel()))
