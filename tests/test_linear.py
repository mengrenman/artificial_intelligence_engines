import numpy as np

from aie.gradcheck import numerical_gradient, relative_error
from aie.losses import mean_sse_grad
from scripts.linear import LinearAssociativeNetwork


def test_gradient_matches_error():
    rng = np.random.default_rng(0)
    X, T = rng.standard_normal((8, 3)), rng.standard_normal((8, 2))
    net = LinearAssociativeNetwork(3, 2, seed=1)
    analytic = mean_sse_grad(net.predict(X), T).T @ X
    num = numerical_gradient(lambda: net.error(X, T), net.W)
    # Regression test: the old code's gradient was exactly 2x too large for K > 1.
    assert relative_error(analytic, num) < 1e-7


def test_recovers_true_weights():
    rng = np.random.default_rng(0)
    W_true = rng.standard_normal((2, 3))
    X = rng.standard_normal((200, 3))
    net = LinearAssociativeNetwork(3, 2, lr=0.5, seed=1).fit(X, X @ W_true.T, n_epochs=500)
    assert np.allclose(net.W, W_true, atol=1e-4)
    assert net.history[0] > net.history[-1]
