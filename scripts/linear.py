"""Linear associative network — Chapter 2 of Stone (2020).

Implements the model y = W x with no thresholds and no hidden units, trained
by gradient descent on a sum-of-squares error (Eq 2.5, Eq 2.32).

References
----------
- Section 2.4 (Gradient Descent) — derivation for one weight, Eq 2.11–2.13
- Section 2.5 (Learning Two Associations) — Eq 2.27–2.32
- Pseudocode boxes on pages 16 and 22
"""

from __future__ import annotations

import numpy as np

from aie.losses import mean_sse, mean_sse_grad


class LinearAssociativeNetwork:
    """Fully-connected linear network with I inputs and K outputs.

    Parameters
    ----------
    n_inputs, n_outputs : int
        Layer sizes I and K. The network has K * I weights.
    lr : float
        Learning-rate parameter ε in Eq 2.11. Step size for gradient descent.
    seed : int
        Random seed for the initial weights.

    Notes
    -----
    The book uses the convention y = W x with W of shape (K, I); this class
    follows that convention. The forward pass over a batch X of shape (T, I)
    is implemented as Y = X @ W.T to keep batch dimension first.

    The error is the per-association average of Eq 2.5 (see
    :mod:`aie.losses`), E = (1/T) Σ_t ½ Σ_k (y_kt − ŷ_kt)². Its gradient
    with respect to a weight w_kj (Eq 2.29, summed as in Eq 2.32) is

        dE/dw_kj = (1/T) Σ_t δ_kt x_jt,    δ_kt = y_kt − ŷ_kt

    which over a batch is ``delta.T @ X / T``.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        lr: float = 0.01,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.W: np.ndarray = rng.standard_normal((n_outputs, n_inputs)) * 0.1
        self.lr = lr
        self.history: list[float] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute Y = X W^T for a batch X of shape (T, I)."""
        return X @ self.W.T

    def error(self, X: np.ndarray, T: np.ndarray) -> float:
        """Eq 2.5 averaged over the T associations in the batch."""
        return mean_sse(self.predict(X), T)

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        n_epochs: int = 100,
        verbose: bool = False,
    ) -> LinearAssociativeNetwork:
        """Batch gradient descent on Eq 2.5 (Eq 2.32 over many associations).

        ``history[i]`` is the error E *before* weight update i.
        """
        for epoch in range(n_epochs):
            Y = self.predict(X)               # (T, K)
            delta = mean_sse_grad(Y, T)       # (T, K)  Eq 2.27, scaled by 1/T
            grad = delta.T @ X                # (K, I)  Eq 2.29 / 2.32
            self.history.append(mean_sse(Y, T))
            self.W -= self.lr * grad
            if verbose and epoch % max(1, n_epochs // 10) == 0:
                print(f"epoch {epoch:5d}  E = {self.history[-1]:.6f}")
        return self
