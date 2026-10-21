import numpy as np

from aie.data import two_class_2d
from scripts.perceptron import Perceptron


def test_converges_on_separable_data():
    X, t = two_class_2d(seed=0)
    p = Perceptron(2, lr=0.01).fit(X, t, n_epochs=200)
    assert p.errors_per_epoch[-1] == 0
    assert np.all(p.predict(X) == t)
    assert p.correct_per_epoch[-1] == len(t)


def test_bias_handles_offset_boundary():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (200, 2))
    t = np.where(X.sum(axis=1) > 1.0, 1.0, -1.0)
    margin = np.abs(X.sum(axis=1) - 1.0) > 0.05
    X, t = X[margin], t[margin]
    without = Perceptron(2, lr=0.05).fit(X, t, n_epochs=300)
    with_bias = Perceptron(2, lr=0.05, bias=True).fit(X, t, n_epochs=300)
    assert np.all(with_bias.predict(X) == t)
    assert np.mean(without.predict(X) == t) < 1.0


def test_update_rule_sign():
    p = Perceptron(2, lr=1.0)
    p.w[:] = [-1.0, 0.0]
    x = np.array([[1.0, 0.0]])
    p.fit(x, np.array([1.0]), n_epochs=1, shuffle=False)
    assert p.w[0] > -1.0  # moved towards x for a positive target
