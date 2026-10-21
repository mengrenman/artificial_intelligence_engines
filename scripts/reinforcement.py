"""Reinforcement learning — Chapter 10 of Stone (2020).

Tabular temporal-difference (TD) learning:

* TD(0) state-value prediction (Eq 10.27–10.28)::

      δ_t = R_{t+1} + γ V(s_{t+1}) − V(s_t),      V(s_t) ← V(s_t) + ε δ_t

* TD(λ) with accumulating eligibility traces (Eq 10.30–10.32).
* SARSA, on-policy control (Eq 10.41, pseudocode page 160 / PDF 173)::

      Q(s_t,a_t) ← Q(s_t,a_t) + ε[R_{t+1} + γ Q(s_{t+1},a_{t+1}) − Q(s_t,a_t)]

* Q-learning, off-policy control (Eq 10.42, pseudocode page 161 / PDF 174)::

      Q(s_t,a_t) ← Q(s_t,a_t) + ε[R_{t+1} + γ max_a Q(s_{t+1},a) − Q(s_t,a_t)]

* Balancing a pole on a cart (Section 10.10, pseudocode page 165 / PDF 178).

Environments: the 5-state random walk and the cliff-walking gridworld from
Sutton & Barto (2018, Examples 6.2 and 6.6), and the cart–pole.

Notation follows the book: ε (``lr``) is the learning rate, the exploration
probability of the ε-greedy policy is called ``explore`` (the book uses a
different-looking epsilon, ε, for it), γ is the discount factor and λ the
trace-decay parameter. Terminal states have value zero (Q(s_T, a) = 0).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Random walk (Sutton & Barto Example 6.2) and TD prediction
# ---------------------------------------------------------------------------

RW_N_STATES = 5
#: True values of the non-terminal states A..E under the random policy (γ = 1).
RW_TRUE_VALUES = np.arange(1, RW_N_STATES + 1) / (RW_N_STATES + 1)


@dataclass
class Episode:
    """One episode: visited states s_0..s_T and rewards R_1..R_T."""

    states: list[int]
    rewards: list[float]


def random_walk_episode(rng: np.random.Generator, n_states: int = RW_N_STATES) -> Episode:
    """Generate one random-walk episode.

    States are numbered 0..n_states+1; 0 and n_states+1 are terminal. Every
    episode starts in the centre and moves left/right with probability ½.
    The reward is +1 for exiting on the right, otherwise 0.
    """
    s = (n_states + 1) // 2
    states, rewards = [s], []
    while 0 < s < n_states + 1:
        s += 1 if rng.random() < 0.5 else -1
        states.append(s)
        rewards.append(1.0 if s == n_states + 1 else 0.0)
    return Episode(states, rewards)


def random_walk_episodes(n: int, seed: int = 0, n_states: int = RW_N_STATES) -> list[Episode]:
    rng = np.random.default_rng(seed)
    return [random_walk_episode(rng, n_states) for _ in range(n)]


def _rms(V: np.ndarray, true_values: np.ndarray) -> float:
    return float(np.sqrt(np.mean((V[1:-1] - true_values) ** 2)))


def td0(
    episodes: Sequence[Episode],
    n_states: int = RW_N_STATES,
    lr: float = 0.1,
    gamma: float = 1.0,
    v0: float = 0.5,
    true_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Online TD(0) prediction, Eq 10.28.

    Returns the final value table V (including the two terminal entries,
    which stay at 0) and the RMS error against ``true_values`` after every
    episode (empty if ``true_values`` is None).
    """
    V = np.full(n_states + 2, v0)
    V[0] = V[-1] = 0.0
    errors = []
    for ep in episodes:
        for t, r in enumerate(ep.rewards):
            s, s_next = ep.states[t], ep.states[t + 1]
            delta = r + gamma * V[s_next] - V[s]  # TD error, Eq 10.27
            V[s] += lr * delta
        if true_values is not None:
            errors.append(_rms(V, true_values))
    return V, np.array(errors)


def td_lambda(
    episodes: Sequence[Episode],
    lam: float,
    n_states: int = RW_N_STATES,
    lr: float = 0.1,
    gamma: float = 1.0,
    v0: float = 0.5,
    true_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Online TD(λ) prediction with accumulating eligibility traces.

    At each step all traces decay, e ← γλ e (Eq 10.31), the current state's
    trace is boosted, e(s_t) ← e(s_t) + 1 (Eq 10.30), and then *every* state
    is updated in proportion to its trace, V(s) ← V(s) + ε δ_t e(s).

    Notes
    -----
    Eq 10.32 in the book is written as an update of V(s_t) only,
    V(s_t) ← V(s_t) + ε e(s_t) δ_t. That would make the trace pointless for
    credit assignment (earlier states would never receive the TD error). We
    implement the standard backward-view TD(λ) (Sutton & Barto 2018,
    Section 12.2), which updates all states by δ_t e(s). For λ = 0 the trace
    is the indicator of s_t, so both forms reduce exactly to TD(0), as the
    book states. Traces are reset to zero at the start of each episode.
    """
    V = np.full(n_states + 2, v0)
    V[0] = V[-1] = 0.0
    errors = []
    for ep in episodes:
        e = np.zeros_like(V)
        for t, r in enumerate(ep.rewards):
            s, s_next = ep.states[t], ep.states[t + 1]
            delta = r + gamma * V[s_next] - V[s]
            e *= gamma * lam
            e[s] += 1.0
            V += lr * delta * e
            V[0] = V[-1] = 0.0
        if true_values is not None:
            errors.append(_rms(V, true_values))
    return V, np.array(errors)


# ---------------------------------------------------------------------------
# ε-greedy action selection
# ---------------------------------------------------------------------------


def greedy_action(q: np.ndarray, rng: np.random.Generator | None = None) -> int:
    """argmax_a Q(s, a) (Eq 10.39); ties are broken at random if ``rng`` is given."""
    if rng is None:
        return int(np.argmax(q))
    best = np.flatnonzero(q == q.max())
    return int(best[0] if len(best) == 1 else rng.choice(best))


def epsilon_greedy(q: np.ndarray, explore: float, rng: np.random.Generator) -> int:
    """With probability ``explore`` choose a random action, else the greedy one.

    This is the book's rule on page 159: draw P ∈ [0, 1]; a = a* if P ≥ ε,
    otherwise a random action.
    """
    if rng.random() < explore:
        return int(rng.integers(len(q)))
    return greedy_action(q, rng)


# ---------------------------------------------------------------------------
# Cliff walking (Sutton & Barto Example 6.6), SARSA and Q-learning
# ---------------------------------------------------------------------------

#: Actions: up, right, down, left as (row, column) offsets.
GRID_ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


class CliffWalk:
    """4 × 12 gridworld. Start bottom-left, goal bottom-right.

    The cells between them on the bottom row are the cliff: stepping into it
    gives reward −100 and sends the agent back to the start. Every other step
    gives −1. Moves off the grid leave the position unchanged. States are
    integers ``row * n_cols + col``.
    """

    def __init__(self, n_rows: int = 4, n_cols: int = 12):
        self.n_rows, self.n_cols = n_rows, n_cols
        self.n_states = n_rows * n_cols
        self.n_actions = len(GRID_ACTIONS)
        self.start = self.to_state(n_rows - 1, 0)
        self.goal = self.to_state(n_rows - 1, n_cols - 1)

    def to_state(self, row: int, col: int) -> int:
        return row * self.n_cols + col

    def to_rc(self, s: int) -> tuple[int, int]:
        return divmod(s, self.n_cols)

    def is_cliff(self, row: int, col: int) -> bool:
        return row == self.n_rows - 1 and 0 < col < self.n_cols - 1

    def step(self, s: int, a: int) -> tuple[int, float, bool]:
        """Return (s_{t+1}, R_{t+1}, done)."""
        r, c = self.to_rc(s)
        dr, dc = GRID_ACTIONS[a]
        r = min(max(r + dr, 0), self.n_rows - 1)
        c = min(max(c + dc, 0), self.n_cols - 1)
        if self.is_cliff(r, c):
            return self.start, -100.0, False
        s_next = self.to_state(r, c)
        return s_next, -1.0, s_next == self.goal


def _td_control(
    env: CliffWalk,
    method: str,
    n_episodes: int,
    lr: float,
    gamma: float,
    explore: float,
    seed: int,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))  # Q(s_T, a) = 0 for the goal
    returns = np.zeros(n_episodes)
    for k in range(n_episodes):
        s = env.start
        a = epsilon_greedy(Q[s], explore, rng)
        total = 0.0
        for _ in range(max_steps):
            if method == "q":
                a = epsilon_greedy(Q[s], explore, rng)
            s_next, r, done = env.step(s, a)
            total += r
            a_next = epsilon_greedy(Q[s_next], explore, rng)
            if done:
                target = r
            elif method == "sarsa":
                target = r + gamma * Q[s_next, a_next]  # Eq 10.41
            else:
                target = r + gamma * Q[s_next].max()  # Eq 10.42
            Q[s, a] += lr * (target - Q[s, a])
            s, a = s_next, a_next
            if done:
                break
        returns[k] = total
    return Q, returns


def sarsa(
    env: CliffWalk,
    n_episodes: int = 500,
    lr: float = 0.5,
    gamma: float = 1.0,
    explore: float = 0.1,
    seed: int = 0,
    max_steps: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """On-policy SARSA (Eq 10.41). Returns Q and the sum of rewards per episode."""
    return _td_control(env, "sarsa", n_episodes, lr, gamma, explore, seed, max_steps)


def q_learning(
    env: CliffWalk,
    n_episodes: int = 500,
    lr: float = 0.5,
    gamma: float = 1.0,
    explore: float = 0.1,
    seed: int = 0,
    max_steps: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Off-policy Q-learning (Eq 10.42). Returns Q and the sum of rewards per episode."""
    return _td_control(env, "q", n_episodes, lr, gamma, explore, seed, max_steps)


def greedy_path(env: CliffWalk, Q: np.ndarray, max_steps: int = 200) -> list[tuple[int, int]]:
    """Cells (row, col) visited by the greedy policy from start to goal."""
    s = env.start
    path = [env.to_rc(s)]
    for _ in range(max_steps):
        s, _, done = env.step(s, greedy_action(Q[s]))
        path.append(env.to_rc(s))
        if done:
            break
    return path


# ---------------------------------------------------------------------------
# Cart–pole (Section 10.10)
# ---------------------------------------------------------------------------


@dataclass
class CartPole:
    """Cart–pole dynamics of Barto, Sutton & Anderson (1983), Euler-integrated.

    State s = (x, ẋ, θ, θ̇) with x in metres, θ in radians (θ = 0 is upright;
    positive θ leans in the +x direction). Equations of motion (frictionless,
    as corrected by Florian 2007)::

        θ̈ = [g sinθ + cosθ (−F − m l θ̇² sinθ)/M] / [l (4/3 − m cos²θ / M)]
        ẍ = [F + m l (θ̇² sinθ − θ̈ cosθ)] / M,       M = m_cart + m_pole

    where l is the half-length of the pole. Actions 0, 1, 2 are the book's
    a₋₁ (push left), a₀ (do nothing), a₊₁ (push right): F = (a − 1) × force.

    Notes
    -----
    The book does not give physical parameters; we use the classic values
    (g = 9.8, m_cart = 1, m_pole = 0.1, l = 0.5, |F| = 10 N, dt = 0.02 s).
    Termination follows the book's box: |x| > 4 (interpreted as metres),
    |θ| > 45°, or 1000 actions.
    """

    gravity: float = 9.8
    m_cart: float = 1.0
    m_pole: float = 0.1
    half_length: float = 0.5
    force: float = 10.0
    dt: float = 0.02
    x_limit: float = 4.0
    theta_limit: float = math.radians(45.0)
    n_actions: int = 3

    def step(self, s: Sequence[float], a: int) -> tuple[float, float, float, float]:
        x, x_dot, theta, theta_dot = s
        F = (a - 1) * self.force
        M = self.m_cart + self.m_pole
        sin, cos = math.sin(theta), math.cos(theta)
        temp = (F + self.m_pole * self.half_length * theta_dot**2 * sin) / M
        theta_acc = (self.gravity * sin - cos * temp) / (
            self.half_length * (4.0 / 3.0 - self.m_pole * cos**2 / M)
        )
        x_acc = temp - self.m_pole * self.half_length * theta_acc * cos / M
        return (
            x + self.dt * x_dot,
            x_dot + self.dt * x_acc,
            theta + self.dt * theta_dot,
            theta_dot + self.dt * theta_acc,
        )

    def failed(self, s: Sequence[float]) -> bool:
        return abs(s[0]) > self.x_limit or abs(s[2]) > self.theta_limit


def pole_reward(s: Sequence[float], balancing: bool) -> float:
    """Feedback from the book's box on page 165.

    While balancing: R = 10 − 1000 θ² − 5|x| − 10|ẋ|;
    on failure:      R = −10000 − 50|x| − 100|θ|.

    θ is in radians (the book does not state the unit; in degrees the
    1000 θ² term would swamp every other term).
    """
    x, x_dot, theta, _ = s
    if balancing:
        return 10.0 - 1000.0 * theta**2 - 5.0 * abs(x) - 10.0 * abs(x_dot)
    return -10000.0 - 50.0 * abs(x) - 100.0 * abs(theta)


@dataclass
class Discretiser:
    """Map a continuous cart–pole state onto a box index (BOXES, Michie & Chambers 1968).

    Each variable is split at the given interior bin edges, so a variable with
    k edges has k + 1 bins. The defaults were chosen by trial and error (the
    book does not specify them); angles in radians, velocities per second.
    """

    x_edges: tuple[float, ...] = (-1.0, 1.0)
    x_dot_edges: tuple[float, ...] = (-0.5, 0.5)
    theta_edges: tuple[float, ...] = tuple(
        math.radians(d) for d in (-12.0, -6.0, -1.0, 0.0, 1.0, 6.0, 12.0)
    )
    theta_dot_edges: tuple[float, ...] = tuple(
        math.radians(d) for d in (-50.0, -15.0, 0.0, 15.0, 50.0)
    )
    edges: list[np.ndarray] = field(init=False)
    shape: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.edges = [
            np.asarray(e)
            for e in (self.x_edges, self.x_dot_edges, self.theta_edges, self.theta_dot_edges)
        ]
        self.shape = tuple(len(e) + 1 for e in self.edges)

    @property
    def n_boxes(self) -> int:
        return int(np.prod(self.shape))

    def __call__(self, s: Sequence[float]) -> int:
        idx = [int(np.searchsorted(e, v, side="right")) for e, v in zip(self.edges, s, strict=True)]
        return int(np.ravel_multi_index(idx, self.shape))


@dataclass
class PoleResult:
    Q: np.ndarray
    steps: np.ndarray  # actions taken (steps balanced) per episode
    returns: np.ndarray  # sum of rewards per episode


def balance_pole(
    n_episodes: int = 500,
    method: str = "sarsa",
    lr: float = 0.3,
    gamma: float = 0.99,
    explore: float = 0.001,
    explore_decay: float = 0.99,
    max_steps: int = 1000,
    init_noise: float = 0.05,
    cart: CartPole | None = None,
    disc: Discretiser | None = None,
    seed: int = 0,
) -> PoleResult:
    """Learn to balance a pole with tabular SARSA (book) or Q-learning.

    Follows the box "SARSA: Balancing a Pole" (page 165): learning rate
    ε = 0.3, exploration probability ε = 0.001 multiplied by 0.99 at the start
    of every episode, the book's reward/punishment, and at most 1000 actions.

    Notes
    -----
    * The box chooses a random action "if ε < a random number", i.e. with
      probability 1 − ε, which contradicts the ε-greedy rule on page 159.
      This is a typo; we use the correct ε-greedy policy.
    * The box omits γ; we use γ = 0.99 (the text quotes γ = 0.98 from
      Schulman (2016) for a different method; both work).
    * "Cart in the centre" is used for x = ẋ = 0; θ and θ̇ start uniformly in
      ±``init_noise`` (rad, rad/s) so that episodes differ even though the
      policy is almost greedy. Greedy ties are broken at random.
    * Q is tabular over the boxes of :class:`Discretiser`, initialised to 0;
      the terminal state has Q = 0 (target = R on failure).
    """
    cart = cart or CartPole()
    disc = disc or Discretiser()
    rng = np.random.default_rng(seed)
    Q = np.zeros((disc.n_boxes, cart.n_actions))
    steps = np.zeros(n_episodes, dtype=int)
    returns = np.zeros(n_episodes)
    for k in range(n_episodes):
        explore *= explore_decay
        s = (0.0, 0.0, *rng.uniform(-init_noise, init_noise, 2))
        b = disc(s)
        a = epsilon_greedy(Q[b], explore, rng)
        total, t = 0.0, 0
        while t < max_steps:
            t += 1
            s = cart.step(s, a)
            balancing = not cart.failed(s)
            r = pole_reward(s, balancing)
            total += r
            if not balancing:
                Q[b, a] += lr * (r - Q[b, a])
                break
            b_next = disc(s)
            a_next = epsilon_greedy(Q[b_next], explore, rng)
            q_next = Q[b_next, a_next] if method == "sarsa" else Q[b_next].max()
            Q[b, a] += lr * (r + gamma * q_next - Q[b, a])
            b, a = b_next, a_next
        steps[k] = t
        returns[k] = total
    return PoleResult(Q, steps, returns)


def run_greedy_pole(
    Q: np.ndarray,
    s0: Sequence[float] = (0.0, 0.0, 0.02, 0.0),
    max_steps: int = 1000,
    cart: CartPole | None = None,
    disc: Discretiser | None = None,
) -> np.ndarray:
    """Roll out the greedy policy from ``s0``; return the state trajectory (T+1, 4)."""
    cart = cart or CartPole()
    disc = disc or Discretiser()
    s = tuple(s0)
    traj = [s]
    for _ in range(max_steps):
        s = cart.step(s, greedy_action(Q[disc(s)]))
        traj.append(s)
        if cart.failed(s):
            break
    return np.array(traj)
