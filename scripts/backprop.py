"""The backpropagation algorithm — Chapter 4 of Stone (2020).

A 3-layer feed-forward network with one hidden layer of sigmoidal units.
The output layer is linear by default (Section 4.3: input and output units
need not be sigmoidal, by Cybenko's theorem); pass ``output="sigmoid"`` for
the sigmoidal output units used in the book's XOR example (Figs 4.5–4.6),
which solves XOR with 0/1 targets in under 40 iterations. Bias is
implemented as an extra input fixed at -1, exactly as in Fig 4.2.

References
----------
- Sections 4.2 (forward and backward passes), Eq 4.3–4.30
- Pseudocode boxes on pages 43 and 47

Conventions
-----------
- Input batch X has shape (T, n_in).
- Hidden states Z have shape (T, n_hidden).
- Output Y has shape (T, n_out).
- Weight matrices include a column for the bias unit (the book's W_{I+1, j}
  convention from Eq 4.4): W1 has shape (n_hidden, n_in + 1) and W2 has
  shape (n_out, n_hidden + 1).

A bias column is appended (rather than prepended) so the bias weight
appears at index `-1`, consistent with the rest of the codebase.
"""

from __future__ import annotations

import numpy as np

from aie.activations import sigmoid, sigmoid_deriv
from aie.losses import mean_sse, mean_sse_grad


class BackpropNetwork:
    """3-layer perceptron trained with backprop on the halved sum-of-squares error."""

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        n_outputs: int,
        lr: float = 0.1,
        seed: int = 0,
        weight_scale: float = 0.5,
        output: str = "linear",
    ) -> None:
        if output not in ("linear", "sigmoid"):
            raise ValueError(f"output must be 'linear' or 'sigmoid', got {output!r}")
        self.output = output
        rng = np.random.default_rng(seed)
        self.W1: np.ndarray = rng.standard_normal((n_hidden, n_inputs + 1)) * weight_scale
        self.W2: np.ndarray = rng.standard_normal((n_outputs, n_hidden + 1)) * weight_scale
        self.lr = lr
        self.history: list[float] = []

    @staticmethod
    def _augment(A: np.ndarray) -> np.ndarray:
        """Append a constant -1 bias column (Eq 4.4 / Fig 4.2)."""
        return np.hstack([A, -np.ones((A.shape[0], 1))])

    def _forward_cached(self, X: np.ndarray) -> tuple[np.ndarray, ...]:
        """Forward pass that also returns the intermediates backprop needs."""
        Xa = self._augment(X)              # (T, n_in + 1)
        U1 = Xa @ self.W1.T                # (T, n_hidden)
        Z = sigmoid(U1)                    # (T, n_hidden)       Eq 4.1
        Za = self._augment(Z)              # (T, n_hidden + 1)
        Y = Za @ self.W2.T                 # (T, n_out) — linear output
        if self.output == "sigmoid":
            Y = sigmoid(Y)                 # sigmoidal output units (Fig 4.5)
        return Y, Z, Xa, Za

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass. Returns (Y, Z) where Z is the hidden activation."""
        Y, Z, _, _ = self._forward_cached(X)
        return Y, Z

    def predict(self, X: np.ndarray) -> np.ndarray:
        Y, _ = self.forward(X)
        return Y

    def error(self, X: np.ndarray, T: np.ndarray) -> float:
        """Eq 4.8 averaged over the T associations (see :mod:`aie.losses`)."""
        return mean_sse(self.predict(X), T)

    def gradients(self, X: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (dE/dW1, dE/dW2, E) for the current weights."""
        Y, Z, Xa, Za = self._forward_cached(X)

        # ---- output-layer delta: δ_k = (y_k − ŷ_k) f'(u_k)  (Eq 4.16); f' = 1 if linear
        delta_out = mean_sse_grad(Y, T)    # (T, n_out), includes the 1/T factor
        if self.output == "sigmoid":
            delta_out = delta_out * sigmoid_deriv(Y)
        grad_W2 = delta_out.T @ Za         # (n_out, n_hidden + 1)

        # ---- hidden-layer delta (Eq 4.27); the bias unit has no incoming weights,
        # so its column of W2 is dropped before back-propagating.
        W2_no_bias = self.W2[:, :-1]       # (n_out, n_hidden)
        delta_hidden = (delta_out @ W2_no_bias) * sigmoid_deriv(Z)  # (T, n_hidden)
        grad_W1 = delta_hidden.T @ Xa      # (n_hidden, n_in + 1)

        return grad_W1, grad_W2, mean_sse(Y, T)

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        n_epochs: int = 1000,
        verbose: bool = False,
    ) -> BackpropNetwork:
        """Batch gradient descent (Eq 4.37). ``history[i]`` is E before update i."""
        for epoch in range(n_epochs):
            grad_W1, grad_W2, err = self.gradients(X, T)
            self.W1 -= self.lr * grad_W1
            self.W2 -= self.lr * grad_W2
            self.history.append(err)
            if verbose and epoch % max(1, n_epochs // 10) == 0:
                print(f"epoch {epoch:5d}  E = {err:.6f}")
        return self
