import math

import numpy as np

from scripts.reinforcement import (
    RW_TRUE_VALUES,
    CartPole,
    CliffWalk,
    Discretiser,
    balance_pole,
    epsilon_greedy,
    greedy_path,
    pole_reward,
    q_learning,
    random_walk_episodes,
    sarsa,
    td0,
    td_lambda,
)


def test_td0_converges_to_true_random_walk_values():
    episodes = random_walk_episodes(2000, seed=0)
    V, errors = td0(episodes, lr=0.01, true_values=RW_TRUE_VALUES)
    assert V[0] == V[-1] == 0.0
    np.testing.assert_allclose(V[1:-1], RW_TRUE_VALUES, atol=0.06)
    assert errors[-1] < 0.05 < errors[0]


def test_td_lambda_zero_equals_td0():
    episodes = random_walk_episodes(100, seed=1)
    V0, e0 = td0(episodes, lr=0.1, true_values=RW_TRUE_VALUES)
    Vl, el = td_lambda(episodes, lam=0.0, lr=0.1, true_values=RW_TRUE_VALUES)
    np.testing.assert_allclose(Vl, V0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(el, e0, rtol=0, atol=1e-12)


def test_td_lambda_learns_faster_early_on():
    errs = {lam: [] for lam in (0.0, 0.8)}
    for seed in range(20):
        episodes = random_walk_episodes(10, seed=seed)
        for lam in errs:
            errs[lam].append(td_lambda(episodes, lam, lr=0.1, true_values=RW_TRUE_VALUES)[1][-1])
    assert np.mean(errs[0.8]) < np.mean(errs[0.0])


def test_epsilon_greedy_exploration_rate():
    rng = np.random.default_rng(0)
    q = np.array([0.0, 1.0, 0.0])
    a = np.array([epsilon_greedy(q, 0.3, rng) for _ in range(5000)])
    assert abs(np.mean(a != 1) - 0.3 * 2 / 3) < 0.03
    assert all(epsilon_greedy(q, 0.0, rng) == 1 for _ in range(100))


def test_q_learning_finds_optimal_cliff_edge_path():
    env = CliffWalk()
    Q, _ = q_learning(env, n_episodes=500, seed=0)
    path = greedy_path(env, Q)
    assert path[-1] == env.to_rc(env.goal)
    assert len(path) - 1 == 13
    assert all(r >= 2 for r, _ in path)


def test_sarsa_learns_safe_path():
    env = CliffWalk()
    Q, returns_s = sarsa(env, n_episodes=500, seed=0)
    path = greedy_path(env, Q)
    assert path[-1] == env.to_rc(env.goal)
    assert len(path) - 1 > 13
    assert not any(r == 2 and 0 < c < 11 for r, c in path)
    _, returns_q = q_learning(env, n_episodes=500, seed=0)
    assert returns_s[-200:].mean() > returns_q[-200:].mean()  # safer online


def test_cliff_step():
    env = CliffWalk()
    assert env.step(env.start, 1) == (env.start, -100.0, False)
    assert env.step(env.start, 3) == (env.start, -1.0, False)
    assert env.step(env.to_state(2, 11), 2) == (env.goal, -1.0, True)


def test_pole_falls_without_control():
    cart = CartPole()
    s = (0.0, 0.0, 0.01, 0.0)
    for t in range(1000):  # noqa: B007
        s = cart.step(s, 1)
        if cart.failed(s):
            break
    assert t < 200 and s[2] > math.radians(45)


def test_cartpole_mirror_symmetry():
    cart = CartPole()
    s = (0.3, -0.2, 0.1, 0.4)
    mirror = tuple(-v for v in s)
    for a in range(3):
        np.testing.assert_allclose(cart.step(mirror, 2 - a), [-v for v in cart.step(s, a)])


def test_push_right_accelerates_cart_right():
    cart = CartPole()
    s0 = (0.0, 0.0, 0.0, 0.0)
    assert cart.step(cart.step(s0, 2), 2)[1] > 0
    assert cart.step(cart.step(s0, 2), 2)[3] < 0  # pole tips back


def test_pole_reward_matches_book():
    s = (0.5, -0.2, 0.1, 3.0)
    assert math.isclose(pole_reward(s, True), 10 - 1000 * 0.01 - 5 * 0.5 - 10 * 0.2)
    assert math.isclose(pole_reward(s, False), -10000 - 50 * 0.5 - 100 * 0.1)


def test_discretiser_indices_in_range():
    disc = Discretiser()
    rng = np.random.default_rng(0)
    idx = {disc(rng.normal(size=4)) for _ in range(2000)}
    assert min(idx) >= 0 and max(idx) < disc.n_boxes
    assert disc((0, 0, 0.001, 0)) != disc((0, 0, -0.001, 0))


def test_balance_pole_improves():
    res = balance_pole(n_episodes=150, seed=1)
    assert res.steps[-30:].mean() > 2 * res.steps[:30].mean()
