"""Perceptron — Chapter 3 of Stone (2020).

A perceptron is a linear associative network with a step (sign) activation
on its single output unit (Fig 3.1, Fig 3.3). Trained with the perceptron
learning rule: correctly classified inputs leave the weights unchanged,
misclassified inputs move the weight vector w towards (or away from) x.

References
----------
- Section 3.2, Eq 3.3–3.4
- Pseudocode box on page 35

Note on notation
----------------
The book writes the *target* as a sans-serif y (often rendered as ŷ in
plain text) and the *actual output* as an italic y (Section 2.3). With that
distinction Eq 3.4 reads

    Δw = −ε x   if target < 0 and output > 0,
    Δw = +ε x   if target > 0 and output < 0,

which is the standard perceptron rule, written compactly as

    Δw = ε · target · x      (applied only on misclassification).

Text extracted from the PDF loses the font distinction, which can make the
rule look as if its signs were inverted; they are not.
"""

from __future__ import annotations

import numpy as np


class Perceptron:
    """Single-output perceptron with binary {-1, +1} targets.

    Parameters
    ----------
    n_inputs : int
        Number of input units.
    lr : float
        Learning rate ε in Eq 3.4 (the book uses 0.01).
    seed : int
        Seed for the random initial weights.
    bias : bool
        If True, append a bias input fixed at −1 (the threshold θ of Eq 3.3
        treated as an extra weight, as in Section 5.2). The book's Ch 3
        example assumes θ = 0, so the default is False and the decision
        boundary passes through the origin.
    """

    def __init__(
        self, n_inputs: int, lr: float = 0.01, seed: int = 0, bias: bool = False
    ) -> None:
        rng = np.random.default_rng(seed)
        self.bias = bias
        self.w: np.ndarray = rng.standard_normal(n_inputs + int(bias)) * 0.1
        self.lr = lr
        self.errors_per_epoch: list[int] = []
        # Number of correctly classified vectors after each epoch (Fig 3.4a).
        self.correct_per_epoch: list[int] = []

    @staticmethod
    def _sign(u: np.ndarray | float) -> np.ndarray | float:
        """Step activation (Eq 3.3 with θ = 0); sign(0) = −1 as in the book."""
        return np.where(np.asarray(u) > 0, 1.0, -1.0)

    def _augment(self, X: np.ndarray) -> np.ndarray:
        if not self.bias:
            return X
        X = np.atleast_2d(X)
        return np.hstack([X, -np.ones((X.shape[0], 1))])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Total input u = w · x to the output unit (Eq 3.2)."""
        return self._augment(X) @ self.w

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._sign(self.decision_function(X))

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        n_epochs: int = 100,
        shuffle: bool = True,
        seed: int = 0,
    ) -> Perceptron:
        """Online perceptron learning; stops early once an epoch has no errors."""
        rng = np.random.default_rng(seed)
        Xa = self._augment(X)
        for _ in range(n_epochs):
            n_misclassified = 0
            order = rng.permutation(len(t)) if shuffle else range(len(t))
            for i in order:
                x_i, t_i = Xa[i], t[i]
                y_i = self._sign(x_i @ self.w)
                if y_i != t_i:
                    self.w += self.lr * t_i * x_i      # Eq 3.4
                    n_misclassified += 1
            self.errors_per_epoch.append(n_misclassified)
            self.correct_per_epoch.append(int(np.sum(self._sign(Xa @ self.w) == t)))
            if n_misclassified == 0:
                break
        return self
