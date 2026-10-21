import numpy as np

from aie.gradcheck import numerical_gradient, relative_error
from aie.losses import mean_sse, mean_sse_grad, softmax_cross_entropy


def test_mean_sse_grad_matches_numerical():
    rng = np.random.default_rng(0)
    Y, T = rng.standard_normal((6, 3)), rng.standard_normal((6, 3))
    num = numerical_gradient(lambda: mean_sse(Y, T), Y)
    assert relative_error(mean_sse_grad(Y, T), num) < 1e-7


def test_mean_sse_value():
    Y = np.array([[1.0, 2.0], [0.0, 0.0]])
    T = np.zeros((2, 2))
    # (1/T) * 1/2 * sum of squares = 0.5 * 0.5 * 5
    assert np.isclose(mean_sse(Y, T), 1.25)


def test_mean_sse_1d_targets():
    assert np.isclose(mean_sse(np.array([1.0, 3.0]), np.array([0.0, 0.0])), 2.5)


def test_softmax_cross_entropy_grad_and_stability():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((5, 4))
    labels = rng.integers(0, 4, 5)
    loss, grad = softmax_cross_entropy(logits, labels)
    num = numerical_gradient(lambda: softmax_cross_entropy(logits, labels)[0], logits)
    assert relative_error(grad, num) < 1e-6
    big, _ = softmax_cross_entropy(logits + 1000.0, labels)
    assert np.isfinite(big) and np.isclose(big, loss)
