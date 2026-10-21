import numpy as np
import pytest

from aie.gradcheck import numerical_gradient, relative_error
from scripts.boltzmann import (
    BoltzmannMachine,
    all_binary_states,
    geometric_schedule,
    ising_to_binary,
    quadratic_energy,
    simulated_annealing,
    states_to_index,
)

RING = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]], dtype=float)
PARITY = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)


def test_state_enumeration_roundtrip():
    S = all_binary_states(4)
    assert S.shape == (16, 4)
    assert np.array_equal(S[5], [0, 1, 0, 1])
    assert np.array_equal(states_to_index(S), np.arange(16))


def test_probabilities_sum_to_one():
    bm = BoltzmannMachine(3, 2, weight_scale=1.0, seed=0)
    assert bm.joint_probs().sum() == pytest.approx(1.0)
    assert bm.visible_probs().sum() == pytest.approx(1.0)
    assert bm.data_distribution(RING[:, :3]).sum() == pytest.approx(1.0)


def test_weights_symmetric_zero_diagonal_and_bias_unit():
    bm = BoltzmannMachine(4, 1, seed=0).fit(RING, n_epochs=3, track_kl=False)
    assert np.allclose(bm.W, bm.W.T)
    assert np.all(np.diag(bm.W) == 0)
    assert bm.full_state(np.zeros(5))[-1] == -1.0


def test_energy_matches_pairwise_sum():
    bm = BoltzmannMachine(2, 1, weight_scale=1.0, seed=3)
    y = bm.full_state(np.array([1.0, 0.0, 1.0]))
    pairwise = -sum(bm.W[i, j] * y[i] * y[j] for i in range(4) for j in range(i + 1, 4))
    assert bm.energy(y) == pytest.approx(pairwise)


@pytest.mark.parametrize("n_hidden,beta", [(0, 1.0), (2, 1.0), (2, 0.5)])
def test_exact_gradient_matches_numerical_log_likelihood(n_hidden, beta):
    bm = BoltzmannMachine(3, n_hidden, beta=beta, weight_scale=0.5, seed=1)
    data = RING[:, :3]
    num = numerical_gradient(lambda: bm.log_likelihood(data) / len(data), bm.W)
    upper = np.triu(np.ones_like(bm.W, dtype=bool), 1)
    assert relative_error(bm.exact_gradient(data)[upper], num[upper]) < 1e-6
    assert np.allclose(num[~upper], 0.0)


def test_exact_gradient_is_negative_kl_gradient():
    bm = BoltzmannMachine(4, 1, weight_scale=0.5, seed=2)
    num = numerical_gradient(lambda: bm.kl_divergence(RING), bm.W)
    upper = np.triu(np.ones_like(bm.W, dtype=bool), 1)
    assert relative_error(bm.exact_gradient(RING)[upper], -num[upper]) < 1e-6


def test_exact_learning_reduces_kl():
    bm = BoltzmannMachine(4, 1, seed=0).fit(RING, n_epochs=300, lr=1.0)
    kl = np.array(bm.kl_history)
    assert np.all(np.diff(kl) <= 1e-9)
    assert kl[-1] < 0.05 < kl[0]


def test_hidden_units_needed_for_parity():
    visible_only = BoltzmannMachine(3, 0, seed=0).fit(PARITY, n_epochs=300, lr=1.0)
    with_hidden = BoltzmannMachine(3, 3, weight_scale=0.5, seed=0).fit(
        PARITY, n_epochs=500, lr=1.0
    )
    assert visible_only.kl_history[-1] == pytest.approx(np.log(2), abs=1e-3)
    assert with_hidden.kl_history[-1] < 0.2


def test_sampled_stats_approximate_exact_stats():
    bm = BoltzmannMachine(3, 2, weight_scale=1.0, seed=4)
    data = RING[:, :3]
    wake_e, sleep_e = bm.exact_stats(data)
    wake_s, sleep_s = bm.sampled_stats(data, n_samples=300, n_chains=20)
    assert np.max(np.abs(wake_s - wake_e)) < 0.06
    assert np.max(np.abs(sleep_s - sleep_e)) < 0.06


def test_sampled_learning_reduces_kl():
    bm = BoltzmannMachine(4, 1, seed=0).fit(RING, n_epochs=100, lr=1.0, method="sample")
    kl = bm.kl_history
    assert kl[-1] < 0.4 * kl[0]


def test_annealing_respects_clamped_units():
    rng = np.random.default_rng(0)
    W = np.ones((4, 4)) - np.eye(4)
    clamped = np.array([True, False, False, True])
    y, energies = simulated_annealing(W, [1, 0, 0, 0], [2.0, 1.0], 3, rng, clamped=clamped)
    assert y[0] == 1 and y[3] == 0
    assert len(energies) == 6


def test_annealing_finds_global_minimum():
    K = 12
    rng = np.random.default_rng(6)
    A = rng.choice([-1.0, 1.0], (K, K))
    W, b = ising_to_binary(A)
    E = np.array([quadratic_energy(W, y, b) for y in all_binary_states(K)])
    e_min = E.min()
    schedule = geometric_schedule(10.0, 0.2, 20)
    n_runs, annealed, greedy = 30, 0, 0
    for _ in range(n_runs):
        y, _ = simulated_annealing(W, rng.integers(0, 2, K), schedule, 5, rng, b=b)
        annealed += np.isclose(quadratic_energy(W, y, b), e_min)
        y, _ = simulated_annealing(W, rng.integers(0, 2, K), [0.01], 100, rng, b=b)
        greedy += np.isclose(quadratic_energy(W, y, b), e_min)
    assert annealed / n_runs >= 0.7
    assert annealed > greedy
