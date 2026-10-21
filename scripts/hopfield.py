"""Hopfield nets — Chapter 5 of Stone (2020).

A Hopfield net is a fully connected network of K binary units with states
y_j ∈ {−1, +1} and symmetric weights w_ij = w_ji, with no self-connections
(w_jj = 0) (Section 5.2, Fig 5.1). It acts as a content addressable memory:
a corrupted or partial cue is "cleaned up" by letting the network settle
into the nearest stored (low-energy) state (Section 5.4).

References
----------
- Section 5.2, Eq 5.4: y_j = f(Σ_i w_ij y_i), with f(u) = −1 if u ≤ 0 else +1.
- Section 5.3, Eq 5.8 and Eq 5.11: w_ij = ε Σ_t x_it x_jt with ε = 1/T,
  i.e. W = ε Σ_t x_t x_tᵀ (outer products). Learning pseudocode, page 66
  of the book (PDF page 79).
- Section 5.4: recall pseudocode, page 67 of the book (PDF page 80) —
  asynchronous updates of units chosen in random order *without
  replacement*, repeated until a full sweep changes no unit.
- Section 5.6, Eq 5.12: E(y) = −Σ_i Σ_{j>i} w_ij y_i y_j, which never
  increases under asynchronous updates.

Notes
-----
- The learning pseudocode sums outer products without the factor ε = 1/T;
  we include it (Eq 5.11). The scale of W does not affect recall because
  f only looks at the sign of u_j.
- The pseudocode forms W_t = x_t x_tᵀ, whose diagonal is all ones; the
  diagonal is zeroed afterwards because units have no self-connections.
- The capacity rule of thumb (about 0.14 K random patterns can be stored
  before recall breaks down) is the classical result of Amit, Gutfreund &
  Sompolinsky (1985); it is not derived in the book.
"""

from __future__ import annotations

import numpy as np


def sign_activation(u: np.ndarray | float) -> np.ndarray:
    """Step activation of Eq 5.4: −1 if u ≤ 0, else +1."""
    return np.where(np.asarray(u) > 0, 1.0, -1.0)


class HopfieldNet:
    """Hopfield net with K bipolar (±1) units.

    Parameters
    ----------
    n_units : int
        Number of units K.

    Attributes
    ----------
    W : ndarray, shape (K, K)
        Symmetric weight matrix with zero diagonal; W[i, j] = w_ij.
    """

    def __init__(self, n_units: int) -> None:
        self.K = n_units
        self.W: np.ndarray = np.zeros((n_units, n_units))

    def fit(self, X: np.ndarray) -> HopfieldNet:
        """Store the T patterns in the rows of ``X`` (shape (T, K), ±1).

        Implements the learning pseudocode: W ← W + x_t x_tᵀ for every t,
        scaled by ε = 1/T (Eq 5.11), then zero the diagonal.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.K:
            raise ValueError(f"expected patterns of length {self.K}, got {X.shape[1]}")
        if not np.all(np.isin(X, (-1.0, 1.0))):
            raise ValueError("patterns must be bipolar (±1)")
        T = X.shape[0]
        W = np.zeros((self.K, self.K))
        for x in X:
            W += np.outer(x, x)
        W *= 1.0 / T
        np.fill_diagonal(W, 0.0)
        self.W = W
        return self

    def energy(self, y: np.ndarray) -> float:
        """E(y) = −Σ_{i<j} w_ij y_i y_j (Eq 5.12) = −½ yᵀWy (W symmetric, zero diag)."""
        y = np.asarray(y, dtype=float)
        return float(-0.5 * y @ self.W @ y)

    def recall(
        self,
        x_cue: np.ndarray,
        rng: np.random.Generator | None = None,
        max_sweeps: int = 100,
        return_energies: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[float]]:
        """Recall a stored pattern from a cue ``x_cue`` (recall pseudocode, Section 5.4).

        Units are updated asynchronously, in a random order without
        replacement within each sweep; sweeps repeat until one sweep changes
        no unit (the state is stable) or ``max_sweeps`` is reached.

        If ``return_energies`` is True, also return the energy after every
        single-unit update (the first entry is the energy of the cue).
        """
        rng = np.random.default_rng(0) if rng is None else rng
        y = np.array(x_cue, dtype=float)
        energies = [self.energy(y)] if return_energies else []
        for _ in range(max_sweeps):
            stable = True
            for j in rng.permutation(self.K):
                u_j = self.W[:, j] @ y
                y_last = y[j]
                y[j] = sign_activation(u_j)
                if y[j] != y_last:
                    stable = False
                if return_energies:
                    energies.append(self.energy(y))
            if stable:
                break
        return (y, energies) if return_energies else y

    def is_stable(self, y: np.ndarray) -> bool:
        """True if no unit would change state, i.e. y = f(Wy) (Eq 5.7)."""
        y = np.asarray(y, dtype=float)
        return bool(np.all(sign_activation(self.W @ y) == y))


def corrupt(x: np.ndarray, n_flips: int, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of bipolar pattern ``x`` with ``n_flips`` randomly chosen units flipped."""
    y = np.array(x, dtype=float)
    idx = rng.choice(y.size, size=n_flips, replace=False)
    y.flat[idx] *= -1
    return y


def to_bipolar(x01: np.ndarray) -> np.ndarray:
    """Map {0, 1} values to {−1, +1}."""
    return 2.0 * np.asarray(x01, dtype=float) - 1.0


def recall_fraction(
    n_units: int, n_patterns: int, n_flips: int = 0, seed: int = 0, n_trials: int = 1
) -> float:
    """Fraction of random patterns recalled perfectly (capacity experiment).

    ``n_patterns`` random ±1 patterns of length ``n_units`` are stored; each is
    corrupted by ``n_flips`` flips, recalled, and counted as correct if the
    recalled state equals the stored pattern exactly. Averaged over trials.
    """
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(n_trials):
        X = rng.choice([-1.0, 1.0], size=(n_patterns, n_units))
        net = HopfieldNet(n_units).fit(X)
        for x in X:
            y = net.recall(corrupt(x, n_flips, rng), rng=rng)
            correct += int(np.array_equal(y, x))
    return correct / (n_trials * n_patterns)
