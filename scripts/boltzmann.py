"""Boltzmann machines and simulated annealing — Chapter 6 of Stone (2020).

A Boltzmann machine is a stochastic Hopfield net: binary units y_j ∈ {0, 1},
symmetric weights with zero diagonal, and the stochastic update rule

    P(y_j = 1) = 1 / (1 + exp(−β u_j)),   u_j = Σ_i w_ij y_i,   β = 1/Temp
                                                     (Eq 6.1, Eq 6.2).

At thermal equilibrium the network visits state y with probability given by
the Boltzmann distribution p(y) = exp(−β E(y)) / Z (Eq 6.4, 6.5), with the
Hopfield energy E(y) = −Σ_{i<j} w_ij y_i y_j (Eq 6.3). Units are split into
visible units v and hidden units h (Section 6.5). Learning maximises the
log likelihood L(w) = Σ_t log p_s(v_t) (Eq 6.13, 6.21), equivalently
minimises the KL divergence G between the data distribution p_w(v) and the
model's free-running distribution p_s(v) (Eq 6.7), using

    Δw_ij = ε β ( ⟨y_i y_j⟩_wake − ⟨y_i y_j⟩_sleep )   (pseudocode, page 82).

References
----------
- Section 6.3, Eq 6.1–6.5: activation, energy and Boltzmann distribution.
- Section 6.4: simulated annealing; pseudocode on page 77 (PDF page 90).
- Sections 6.5–6.7, Eq 6.7–6.27: KL divergence, likelihood, and the
  learning algorithm; pseudocode on page 82 (PDF page 95).

Notes
-----
- **Bias unit.** As in Sections 5.2 and 6.3, biases are implemented with an
  extra (K+1)th unit whose state is clamped at −1 for ever. Its weights
  w_{j,bias} are learned like any other weight, so −w_{j,bias} plays the
  role of unit j's bias (equivalently w_{j,bias} is its threshold). The
  energy Eq 6.3 is evaluated over all K+1 units, which includes these bias
  terms (the book drops them "for simplicity").
- **Unit order.** The book writes y = (h, v); here the state vector is
  ordered y = (v, h, bias) so that visible unit i is simply y[i].
- **Statistics are averages.** ⟨y_i y_j⟩_wake is averaged (not summed, as in
  the pseudocode) over the T training vectors, so that it is on the same
  scale as ⟨y_i y_j⟩_sleep. Then β(⟨·⟩_wake − ⟨·⟩_sleep) is exactly the
  gradient ∂(L/T)/∂w_ij = −∂G/∂w_ij, and ε absorbs the factor 1/T.
- **Equilibrium temperature.** The model distribution is defined at inverse
  temperature β. Annealing schedules therefore end at Temp = 1/β, where
  correlations are measured; the earlier, hotter temperatures only help the
  chain reach equilibrium quickly (Section 6.4).
- **Exact version.** For tiny nets (K ≲ 16) all 2^K states can be
  enumerated, giving Z, p_s(v), the likelihood, G and the exact wake/sleep
  statistics. This is used in the tests and to show that the sampled
  (annealing-based) learning rule follows the exact gradient.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

BIAS_STATE = -1.0


def all_binary_states(n: int) -> np.ndarray:
    """All 2^n binary vectors as rows of a (2^n, n) array; row k is k in binary (MSB first)."""
    k = np.arange(2**n)[:, None]
    return ((k >> np.arange(n - 1, -1, -1)) & 1).astype(float)


def states_to_index(states: np.ndarray) -> np.ndarray:
    """Inverse of :func:`all_binary_states`: binary rows → integer indices."""
    states = np.atleast_2d(states).astype(int)
    n = states.shape[1]
    return states @ (1 << np.arange(n - 1, -1, -1))


def logistic(x: float) -> float:
    """Overflow-free scalar sigmoid 1/(1 + e^(−x)) = ½(1 + tanh(x/2)) (Eq 6.1).

    Equivalent to ``aie.activations.sigmoid`` but avoids overflow warnings at
    very low temperatures, where β u can be huge.
    """
    return 0.5 * (1.0 + math.tanh(0.5 * x))


def ising_to_binary(J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rewrite a ±1 spin-glass energy −Σ_{i<j} J_ij s_i s_j for 0/1 units.

    With s = 2y − 1, the energy becomes (up to a constant)
    −Σ_{i<j} w_ij y_i y_j − Σ_i b_i y_i with W = 4J and b_i = −2 Σ_j J_ij.
    Such energies have many local minima, which makes them a good test of
    simulated annealing.
    """
    J = np.triu(J, 1) + np.triu(J, 1).T
    return 4.0 * J, -2.0 * J.sum(axis=1)


def quadratic_energy(W: np.ndarray, y: np.ndarray, b: np.ndarray | None = None) -> float:
    """E(y) = −Σ_{i<j} w_ij y_i y_j − Σ_i b_i y_i (Eq 6.3, plus optional explicit biases)."""
    y = np.asarray(y, dtype=float)
    e = -float(y @ np.triu(W, 1) @ y)
    if b is not None:
        e -= float(b @ y)
    return e


def gibbs_sweep(
    W: np.ndarray,
    y: np.ndarray,
    beta: float,
    rng: np.random.Generator,
    free: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> np.ndarray:
    """One settling iteration (inner loop of the annealing pseudocode), in place.

    Every unclamped unit j (indices in ``free``) is visited once, in random
    order without replacement, and set to 1 with probability
    P_j = 1/(1 + exp(−β u_j)) (Eq 6.1), where u_j = Σ_i w_ij y_i (+ b_j).
    """
    free = np.arange(len(y)) if free is None else free
    for j in rng.permutation(free):
        u = W[j] @ y + (0.0 if b is None else b[j])
        P = logistic(beta * u)
        y[j] = 1.0 if P > rng.random() else 0.0
    return y


def simulated_annealing(
    W: np.ndarray,
    y0: np.ndarray,
    temperatures: Sequence[float],
    n_settle: int,
    rng: np.random.Generator,
    clamped: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Simulated annealing of a network of 0/1 units (pseudocode, Section 6.4).

    Parameters
    ----------
    W : (K, K) symmetric weights with zero diagonal.
    y0 : initial state; clamped units keep their values throughout.
    temperatures : decreasing schedule Tmax … Tmin; β = 1/Temp at each step.
    n_settle : S, the number of settling sweeps at each temperature.
    clamped : optional boolean mask of clamped (never updated) units.
    b : optional explicit bias vector (alternative to a bias unit).

    Returns
    -------
    y : final state.
    energies : energy E(y) (Eq 6.3) after every settling sweep.
    """
    y = np.array(y0, dtype=float)
    free = np.arange(len(y)) if clamped is None else np.flatnonzero(~np.asarray(clamped))
    energies: list[float] = []
    for temp in temperatures:
        beta = 1.0 / temp
        for _ in range(n_settle):
            gibbs_sweep(W, y, beta, rng, free, b)
            energies.append(quadratic_energy(W, y, b))
    return y, energies


def geometric_schedule(t_max: float, t_min: float, n_steps: int) -> np.ndarray:
    """Temperatures decreasing geometrically from t_max to t_min (inclusive)."""
    return np.geomspace(t_max, t_min, n_steps)


class BoltzmannMachine:
    """Fully connected Boltzmann machine with visible, hidden and one bias unit.

    Parameters
    ----------
    n_visible, n_hidden : int
        Numbers of visible (V) and hidden (H) units.
    beta : float
        Inverse temperature β at which the model distribution is defined.
    weight_scale : float
        Standard deviation of the random initial weights.
    seed : int
        Seed for weight initialisation and for sampling.

    Attributes
    ----------
    W : ndarray, shape (N, N), N = V + H + 1
        Symmetric weights with zero diagonal; the last row/column connects
        each unit to the bias unit (state fixed at −1).
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int = 0,
        beta: float = 1.0,
        weight_scale: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.V, self.H = n_visible, n_hidden
        self.N = n_visible + n_hidden + 1
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        A = self.rng.standard_normal((self.N, self.N)) * weight_scale
        self.W: np.ndarray = np.triu(A, 1) + np.triu(A, 1).T
        self.kl_history: list[float] = []

    # ----- helpers ---------------------------------------------------------
    def full_state(self, units: np.ndarray) -> np.ndarray:
        """Append the bias unit (state −1) to a (…, V+H) array of unit states."""
        units = np.asarray(units, dtype=float)
        bias = np.full(units.shape[:-1] + (1,), BIAS_STATE)
        return np.concatenate([units, bias], axis=-1)

    def energy(self, y: np.ndarray) -> np.ndarray:
        """Energy Eq 6.3 of full state(s) y (bias included), using w_ij for i < j only."""
        y = np.asarray(y, dtype=float)
        return -np.einsum("...i,ij,...j->...", y, np.triu(self.W, 1), y)

    def symmetrise(self) -> None:
        """Copy the upper triangle into the lower one and zero the diagonal."""
        U = np.triu(self.W, 1)
        self.W = U + U.T

    # ----- exact computations by enumeration (tiny nets only) --------------
    def _joint(self) -> tuple[np.ndarray, np.ndarray]:
        """All full states (2^(V+H), N) and their Boltzmann probabilities (Eq 6.4, 6.5)."""
        if self.V + self.H > 20:
            raise ValueError("exact enumeration is only feasible for tiny networks")
        Y = self.full_state(all_binary_states(self.V + self.H))
        log_unnorm = -self.beta * self.energy(Y)
        log_unnorm -= log_unnorm.max()
        p = np.exp(log_unnorm)
        return Y, p / p.sum()

    def joint_probs(self) -> np.ndarray:
        """p(h, v) for all 2^(V+H) states, index = v_index · 2^H + h_index (Eq 6.4)."""
        return self._joint()[1]

    def visible_probs(self) -> np.ndarray:
        """Free-running visible distribution p_s(v) = Σ_h p(h, v) (Eq 6.17), length 2^V."""
        return self.joint_probs().reshape(2**self.V, 2**self.H).sum(axis=1)

    def data_distribution(self, data: np.ndarray) -> np.ndarray:
        """Empirical p_w(v) ≈ n_v / T (Eq 6.10) over the 2^V visible states."""
        idx = states_to_index(data)
        return np.bincount(idx, minlength=2**self.V) / len(idx)

    def log_likelihood(self, data: np.ndarray) -> float:
        """L(w) = Σ_t log p_s(v_t) (Eq 6.21)."""
        ps = self.visible_probs()
        return float(np.sum(np.log(ps[states_to_index(data)])))

    def kl_divergence(self, data: np.ndarray) -> float:
        """G = Σ_v p_w(v) log(p_w(v) / p_s(v)) (Eq 6.7)."""
        pw, ps = self.data_distribution(data), self.visible_probs()
        m = pw > 0
        return float(np.sum(pw[m] * np.log(pw[m] / ps[m])))

    def exact_stats(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact ⟨y_i y_j⟩_wake and ⟨y_i y_j⟩_sleep by enumerating all states.

        Wake: visible units clamped to each training vector, hidden units at
        equilibrium p(h | v_t); averaged over the training set (weights p_w(v)).
        Sleep: all units free-running, weights p(h, v).
        """
        Y, p = self._joint()
        joint = p.reshape(2**self.V, 2**self.H)
        pv = joint.sum(axis=1, keepdims=True)
        cond = joint / pv
        q_wake = (self.data_distribution(data)[:, None] * cond).ravel()
        wake = Y.T @ (q_wake[:, None] * Y)
        sleep = Y.T @ (p[:, None] * Y)
        return wake, sleep

    def exact_gradient(self, data: np.ndarray) -> np.ndarray:
        """β(⟨y_i y_j⟩_wake − ⟨y_i y_j⟩_sleep) = ∂(L/T)/∂w_ij = −∂G/∂w_ij, zero diagonal."""
        wake, sleep = self.exact_stats(data)
        g = self.beta * (wake - sleep)
        np.fill_diagonal(g, 0.0)
        return g

    # ----- sampled statistics via simulated annealing ----------------------
    def default_schedule(self) -> np.ndarray:
        """Anneal from Temp = 4/β down to the model temperature 1/β."""
        return geometric_schedule(4.0 / self.beta, 1.0 / self.beta, 4)

    def _anneal_and_collect(
        self,
        y: np.ndarray,
        free: np.ndarray,
        temperatures: Sequence[float],
        n_settle: int,
        n_samples: int,
    ) -> np.ndarray:
        """Anneal y, then average y yᵀ over ``n_samples`` sweeps at equilibrium (Temp = 1/β)."""
        for temp in temperatures:
            for _ in range(n_settle):
                gibbs_sweep(self.W, y, 1.0 / temp, self.rng, free)
        corr = np.zeros((self.N, self.N))
        for _ in range(n_samples):
            gibbs_sweep(self.W, y, self.beta, self.rng, free)
            corr += np.outer(y, y)
        return corr / n_samples

    def sampled_stats(
        self,
        data: np.ndarray,
        temperatures: Sequence[float] | None = None,
        n_settle: int = 2,
        n_samples: int = 10,
        n_chains: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate ⟨y_i y_j⟩_wake and ⟨y_i y_j⟩_sleep by annealing (pseudocode, page 82).

        Wake: for each training vector, clamp the visible units, anneal the
        hidden units and average y yᵀ at equilibrium. (Skipped sampling if
        there are no hidden units: the clamped state is then deterministic.)
        Sleep: ``n_chains`` (default T) free-running chains, each annealed
        from a random state, then sampled at equilibrium.
        """
        data = np.atleast_2d(np.asarray(data, dtype=float))
        temps = self.default_schedule() if temperatures is None else temperatures
        n_chains = len(data) if n_chains is None else n_chains
        hidden = np.arange(self.V, self.V + self.H)
        unclamped = np.arange(self.V + self.H)

        wake = np.zeros((self.N, self.N))
        for v in data:
            h0 = self.rng.integers(0, 2, self.H).astype(float)
            y = self.full_state(np.concatenate([v, h0]))
            if self.H == 0:
                wake += np.outer(y, y)
            else:
                wake += self._anneal_and_collect(y, hidden, temps, n_settle, n_samples)
        wake /= len(data)

        sleep = np.zeros((self.N, self.N))
        for _ in range(n_chains):
            y = self.full_state(self.rng.integers(0, 2, self.V + self.H).astype(float))
            sleep += self._anneal_and_collect(y, unclamped, temps, n_settle, n_samples)
        sleep /= n_chains
        return wake, sleep

    # ----- learning ---------------------------------------------------------
    def fit(
        self,
        data: np.ndarray,
        n_epochs: int = 100,
        lr: float = 0.5,
        method: str = "exact",
        track_kl: bool = True,
        **sample_kwargs,
    ) -> BoltzmannMachine:
        """Boltzmann machine learning (pseudocode, page 82).

        Each epoch computes wake and sleep statistics — exactly (``method=
        "exact"``) or by simulated annealing (``method="sample"``) — and
        applies Δw_ij = ε β (⟨y_i y_j⟩_wake − ⟨y_i y_j⟩_sleep) with ε = ``lr``.
        If ``track_kl``, the exact KL divergence G (Eq 6.7) is recorded in
        ``kl_history`` before training and after every epoch.
        """
        data = np.atleast_2d(np.asarray(data, dtype=float))
        if track_kl:
            self.kl_history.append(self.kl_divergence(data))
        for _ in range(n_epochs):
            if method == "exact":
                wake, sleep = self.exact_stats(data)
            elif method == "sample":
                wake, sleep = self.sampled_stats(data, **sample_kwargs)
            else:
                raise ValueError(f"unknown method {method!r}")
            dW = lr * self.beta * (wake - sleep)
            np.fill_diagonal(dW, 0.0)
            self.W += dW
            self.symmetrise()
            if track_kl:
                self.kl_history.append(self.kl_divergence(data))
        return self

    def sample_visible(
        self,
        n_samples: int,
        temperatures: Sequence[float] | None = None,
        n_settle: int = 2,
        thin: int = 2,
    ) -> np.ndarray:
        """Generate visible vectors from one free-running chain (anneal, then sample)."""
        temps = self.default_schedule() if temperatures is None else temperatures
        free = np.arange(self.V + self.H)
        y = self.full_state(self.rng.integers(0, 2, self.V + self.H).astype(float))
        for temp in temps:
            for _ in range(n_settle):
                gibbs_sweep(self.W, y, 1.0 / temp, self.rng, free)
        out = np.empty((n_samples, self.V))
        for k in range(n_samples):
            for _ in range(thin):
                gibbs_sweep(self.W, y, self.beta, self.rng, free)
            out[k] = y[: self.V]
        return out
