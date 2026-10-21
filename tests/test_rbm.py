import itertools

import numpy as np
import pytest

from aie.data import digits
from scripts.rbm import RBM, RBMStack, encode, fine_tune, random_autoencoder, unroll

PATTERNS = np.array(
    [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]], dtype=float
)


@pytest.fixture(scope="module")
def digit_data():
    X, y = digits(n_per_class=30, seed=0)
    return X.reshape(len(X), -1), y


def test_log_partition_matches_brute_force_energy():
    rbm = RBM(3, 2, seed=1, weight_scale=1.0)
    rbm.b_v[:] = [0.3, -0.2, 0.1]
    rbm.b_h[:] = [-0.5, 0.4]
    total = 0.0
    for v in itertools.product([0, 1], repeat=3):
        for h in itertools.product([0, 1], repeat=2):
            v_, h_ = np.array(v, float), np.array(h, float)
            total += np.exp(v_ @ rbm.W @ h_ + v_ @ rbm.b_v + h_ @ rbm.b_h)
    assert np.isclose(rbm.log_partition(), np.log(total))
    all_v = np.array(list(itertools.product([0, 1], repeat=3)), float)
    probs = np.exp(-rbm.free_energy(all_v) - rbm.log_partition())
    assert np.isclose(probs.sum(), 1.0)


def test_wake_statistic_matches_eq_7_9():
    rbm = RBM(4, 3, seed=0, weight_scale=0.5)
    s = rbm.cd_statistics(PATTERNS)
    P = 1.0 / (1.0 + np.exp(-(PATTERNS @ rbm.W + rbm.b_h)))
    expected = sum(np.outer(v, p) for v, p in zip(PATTERNS, P, strict=True)) / len(PATTERNS)
    assert np.allclose(s["wake"], expected)
    assert s["sleep"].shape == (4, 3)


@pytest.mark.parametrize("n_cd", [1, 3])
def test_cd_increases_exact_log_likelihood(n_cd):
    rbm = RBM(4, 3, seed=0, weight_scale=0.1)
    ll0 = rbm.log_likelihood(PATTERNS)
    rbm.fit(PATTERNS, n_epochs=300, lr=0.1, batch_size=5, n_cd=n_cd)
    assert rbm.log_likelihood(PATTERNS) > ll0 + 0.5


def test_reconstruction_error_decreases_on_digits(digit_data):
    X, _ = digit_data
    rbm = RBM(144, 32, seed=0)
    err0 = rbm.reconstruction_error(X)
    rbm.fit(X, n_epochs=10, lr=0.1, momentum=0.5)
    assert rbm.history[-1] < 0.6 * err0
    assert rbm.history[-1] < rbm.history[0]


def test_unrolled_matches_stack_and_fine_tuning_helps(digit_data):
    X, _ = digit_data
    stack = RBMStack([144, 32, 16, 4], seed=0).pretrain(X, n_epochs=20, lr=0.1, momentum=0.5)
    model = unroll(stack)
    assert np.allclose(model.forward(X), stack.reconstruct(X))
    assert np.allclose(encode(model, X), stack.up(X))
    rbm_W = stack.rbms[0].W.copy()
    history = fine_tune(model, X, n_epochs=30, lr=0.1)
    assert history[-1] < 0.9 * history[0]
    assert np.array_equal(stack.rbms[0].W, rbm_W)  # weights were copied


def test_random_autoencoder_shapes():
    model = random_autoencoder([144, 16, 4])
    assert model.forward(np.zeros((3, 144))).shape == (3, 144)
    assert encode(model, np.zeros((3, 144))).shape == (3, 4)
