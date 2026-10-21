import numpy as np
import pytest

from aie.data import digit_glyphs
from scripts.hopfield import (
    HopfieldNet,
    corrupt,
    recall_fraction,
    sign_activation,
    to_bipolar,
)


def _random_patterns(n_patterns=3, n_units=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(n_patterns, n_units))


def test_sign_activation_zero_maps_to_minus_one():
    assert np.array_equal(sign_activation(np.array([-2.0, 0.0, 3.0])), [-1.0, -1.0, 1.0])


def test_weights_symmetric_zero_diagonal_and_scaled():
    X = _random_patterns()
    net = HopfieldNet(X.shape[1]).fit(X)
    assert np.allclose(net.W, net.W.T)
    assert np.all(np.diag(net.W) == 0)
    expected = X.T @ X / len(X)
    np.fill_diagonal(expected, 0)
    assert np.allclose(net.W, expected)


def test_stored_patterns_are_fixed_points():
    X = _random_patterns()
    net = HopfieldNet(X.shape[1]).fit(X)
    rng = np.random.default_rng(1)
    for x in X:
        assert net.is_stable(x)
        assert np.array_equal(net.recall(x, rng=rng), x)


def test_complement_is_also_stable():
    X = _random_patterns()
    net = HopfieldNet(X.shape[1]).fit(X)
    for x in X:
        assert net.is_stable(-x)


def test_single_pattern_seven_units_book_example():
    x = np.ones(7)
    net = HopfieldNet(7).fit(x[None])
    cue = x.copy()
    cue[6] = -1
    assert np.array_equal(net.recall(cue, rng=np.random.default_rng(0)), x)


def test_corrupted_patterns_recalled():
    X = _random_patterns(n_patterns=4, n_units=100)
    net = HopfieldNet(100).fit(X)
    rng = np.random.default_rng(2)
    for x in X:
        y = net.recall(corrupt(x, 15, rng), rng=rng)
        assert np.array_equal(y, x)


def test_digit_glyphs_recalled_from_noise():
    glyphs = to_bipolar(digit_glyphs()[[0, 1, 4]].reshape(3, -1))
    net = HopfieldNet(glyphs.shape[1]).fit(glyphs)
    rng = np.random.default_rng(0)
    for x in glyphs:
        assert net.is_stable(x)
        assert np.array_equal(net.recall(corrupt(x, 3, rng), rng=rng), x)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_energy_non_increasing_along_recall(seed):
    X = _random_patterns(n_patterns=8, n_units=50, seed=seed)
    net = HopfieldNet(50).fit(X)
    rng = np.random.default_rng(seed)
    cue = rng.choice([-1.0, 1.0], size=50)
    y, energies = net.recall(cue, rng=rng, return_energies=True)
    assert np.all(np.diff(energies) <= 1e-12)
    assert net.is_stable(y)
    assert energies[-1] == pytest.approx(net.energy(y))


def test_energy_matches_pairwise_sum():
    X = _random_patterns(n_units=10)
    net = HopfieldNet(10).fit(X)
    y = X[0]
    pairwise = -sum(net.W[i, j] * y[i] * y[j] for i in range(10) for j in range(i + 1, 10))
    assert net.energy(y) == pytest.approx(pairwise)


def test_capacity_degrades_with_load():
    low = recall_fraction(100, 5, n_flips=5, seed=0, n_trials=2)
    high = recall_fraction(100, 40, n_flips=5, seed=0, n_trials=1)
    assert low == 1.0
    assert high < 0.5
