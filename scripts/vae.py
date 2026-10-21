"""Variational autoencoders — Chapter 8 of Stone (2020).

The encoder maps an input x_t to the parameters (μ_t, σ_t) of a diagonal
Gaussian q(z|x_t) over a J-dimensional latent space; the decoder maps a
sample z to a reconstruction x′. Following the pseudocode box "Training a
Variational Autoencoder" (book p. 118, PDF p. 131), the per-vector
objective (Eq 8.37, dropping the constant c) is

    L_t = L_t^dec + L_t^enc
    L_t^dec = −(1/(2 N_z)) Σ_{i=1}^{N_z} |x_t − x′_it|²                (Eq 8.35, σ_x² = 1)
    L_t^enc = ½ Σ_{j=1}^{J} (1 + log σ²_jt − σ²_jt − μ²_jt)            (Eq 8.36)

where N_z is the number of z samples drawn per input vector (it is *not*
the latent dimension J). L^enc is minus the KL divergence between q(z|x_t)
and the prior p(z) = N(0, I), maximal (zero) when μ = 0 and σ² = 1.
Samples are drawn with the reparametrisation trick z = μ + η σ, η ~ N(0, I)
(Section 8.6), so gradients flow through μ and σ. Weights are changed by
gradient *ascent* on L (w ← w + ε∇L); here we equivalently minimise
E = −L with the optimisers in ``aie.layers``.

Gradients through the sampling layer and KL term (with E = −L):

    ∂E/∂z      = backprop of (1/(N_z σ_x²))(x′ − x) through the decoder
    ∂E/∂μ      = Σ_i ∂E/∂z_i + μ
    ∂E/∂logσ²  = Σ_i ∂E/∂z_i · η_i σ/2 + ½(σ² − 1)

using ∂z/∂μ = 1 and ∂z/∂logσ² = η σ/2 (because σ = exp(½ logσ²)).

Notes
-----
- Parametrisation: the encoder's final Dense layer outputs 2J numbers per
  input, the first J are μ and the last J are log σ² (an unconstrained real
  number, so σ² = exp(log σ²) > 0 automatically).
- The book sums L_t over a batch (L = Σ_t L_t); we report and differentiate
  the batch *mean* (1/T) Σ_t L_t, which only rescales the learning rate and
  matches the ``aie.losses`` convention.
- Eq 8.35 assumes a Gaussian decoder with σ_x² = 1 (stated before Eq 8.33).
  ``sigma_x2`` exposes that variance, giving L^dec = −(1/(2 σ_x² N_z)) Σ_i
  |x_t − x′_it|²; the default 1 is the book's choice. On the small, noisy
  12×12 digits a 2-D latent code reduces |x − x′|² so little that with
  σ_x² = 1 the optimum is "posterior collapse" (q(z|x) = p(z), KL = 0), so
  the notebook uses a smaller σ_x² (i.e. a sharper decoder likelihood).
- The decoder output uses a logistic (Sigmoid) layer because pixels lie in
  [0, 1].
"""

from __future__ import annotations

import numpy as np

from aie.layers import Adam, Dense, Layer, ReLU, Sequential, Sigmoid, Tanh

_ACTIVATIONS: dict[str, type[Layer]] = {"tanh": Tanh, "relu": ReLU, "sigmoid": Sigmoid}


def kl_term(mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    """L^enc_t = ½ Σ_j (1 + log σ² − σ² − μ²) per row (Eq 8.36), i.e. −KL(q‖p)."""
    return 0.5 * np.sum(1.0 + logvar - np.exp(logvar) - mu**2, axis=-1)


class VAE:
    """Encoder/decoder MLP VAE built from ``aie.layers``."""

    def __init__(
        self,
        n_input: int,
        n_latent: int = 2,
        hidden: tuple[int, ...] = (64,),
        activation: str = "tanh",
        seed: int = 0,
        sigma_x2: float = 1.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 1)
        act = _ACTIVATIONS[activation]
        init = "he" if activation == "relu" else "xavier"
        self.n_latent = n_latent
        self.sigma_x2 = sigma_x2

        enc: list[Layer] = []
        n_prev = n_input
        for n_h in hidden:
            enc += [Dense(n_prev, n_h, rng, init), act()]
            n_prev = n_h
        enc.append(Dense(n_prev, 2 * n_latent, rng))       # outputs [μ, log σ²]
        self.encoder = Sequential(*enc)

        dec: list[Layer] = []
        n_prev = n_latent
        for n_h in reversed(hidden):
            dec += [Dense(n_prev, n_h, rng, init), act()]
            n_prev = n_h
        dec += [Dense(n_prev, n_input, rng), Sigmoid()]
        self.decoder = Sequential(*dec)

        self.model = Sequential(self.encoder, self.decoder)  # container for optimisers
        self.history: dict[str, list[float]] = {"loss": [], "recon": [], "kl": []}

    # ---- forward ----
    def encode(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encoder outputs (μ, log σ²), each (T, J)."""
        out = self.encoder.forward(X)
        return out[:, : self.n_latent], out[:, self.n_latent :]

    def decode(self, Z: np.ndarray) -> np.ndarray:
        return self.decoder.forward(Z)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Deterministic reconstruction using z = μ."""
        return self.decode(self.encode(X)[0])

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Decode z ~ p(z) = N(0, I)."""
        rng = rng if rng is not None else self.rng
        return self.decode(rng.standard_normal((n, self.n_latent)))

    def forward(self, X: np.ndarray, eta: np.ndarray) -> dict[str, np.ndarray | float]:
        """Evaluate L for a batch, with noise η of shape (N_z, T, J) held fixed.

        Returns the batch-mean objective ``L`` and its parts ``recon`` (L^dec)
        and ``kl`` (L^enc), plus the cached intermediates used by backward.
        """
        n_z, T, J = eta.shape
        mu, logvar = self.encode(X)
        sigma = np.exp(0.5 * logvar)
        Z = (mu[None] + eta * sigma[None]).reshape(n_z * T, J)   # z = μ + η σ
        Xr = self.decode(Z).reshape(n_z, T, -1)
        l_dec = -np.sum((X[None] - Xr) ** 2, axis=(0, 2)) / (2 * n_z * self.sigma_x2)  # Eq 8.35
        l_enc = kl_term(mu, logvar)                                     # Eq 8.36, per t
        recon, kl = float(l_dec.mean()), float(l_enc.mean())
        return {
            "L": recon + kl, "recon": recon, "kl": kl,
            "mu": mu, "logvar": logvar, "sigma": sigma, "eta": eta, "Xr": Xr, "X": X,
        }

    def loss(self, X: np.ndarray, eta: np.ndarray) -> float:
        """E = −L (the quantity minimised)."""
        return -float(self.forward(X, eta)["L"])

    # ---- backward ----
    def backward(self, cache: dict) -> None:
        """Fill every layer's grads with ∂E/∂θ for E = −L (batch mean)."""
        X, Xr, eta = cache["X"], cache["Xr"], cache["eta"]
        mu, logvar, sigma = cache["mu"], cache["logvar"], cache["sigma"]
        n_z, T, J = eta.shape
        g_xr = (Xr - X[None]) / (n_z * T * self.sigma_x2)         # ∂E/∂x′
        g_z = self.decoder.backward(g_xr.reshape(n_z * T, -1)).reshape(n_z, T, J)
        g_mu = g_z.sum(axis=0) + mu / T
        g_logvar = (g_z * eta).sum(axis=0) * sigma / 2 + 0.5 * (np.exp(logvar) - 1.0) / T
        self.encoder.backward(np.hstack([g_mu, g_logvar]))

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray, Layer, str]]:
        return self.model.parameters()

    # ---- training ----
    def fit(
        self,
        X: np.ndarray,
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        n_z: int = 1,
    ) -> VAE:
        """Maximise L by mini-batch gradient ascent (Adam on −L).

        ``history`` records per-epoch means of the positive costs
        ``loss`` = −L, ``recon`` = −L^dec and ``kl`` = −L^enc = KL(q‖p).
        """
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        opt = Adam(self.model, lr)
        for _ in range(n_epochs):
            perm = self.rng.permutation(len(X))
            sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
            for start in range(0, len(X), batch_size):
                batch = X[perm[start : start + batch_size]]
                eta = self.rng.standard_normal((n_z, len(batch), self.n_latent))
                cache = self.forward(batch, eta)
                self.backward(cache)
                opt.step()
                sums["loss"] += -cache["L"] * len(batch)
                sums["recon"] += -cache["recon"] * len(batch)
                sums["kl"] += -cache["kl"] * len(batch)
            for k, v in sums.items():
                self.history[k].append(v / len(X))
        return self
