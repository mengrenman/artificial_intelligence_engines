"""Restricted Boltzmann machines and deep autoencoders — Chapter 7 of Stone (2020).

A binary RBM has a visible layer v (V units) and a hidden layer h (H units)
with symmetric weights w_ij between layers and none within a layer, so
p(h|v) and p(v|h) factorise (Eqs 7.3–7.6) and a whole layer can be sampled
in one step. Training uses contrastive divergence (CD), following the
pseudocode box "Pre-Training an RBM on Digits" (book p. 93, PDF p. 106):

- Wake: clamp v_t = x_t (pixel values in [0, 1] are treated as
  probabilities), compute u_jt (Eq 7.7) and P_jt = σ(u_jt) (Eq 7.8), and
  record E[v_i P_j]_Wake ≈ (1/T) Σ_t v_it P_jt (Eq 7.9).
- Sleep: starting from the wake probabilities P_jt, repeat N_CD times:
  sample binary h^k, compute P^k_it (Eq 7.10, 7.8), sample binary v^k,
  compute P^k_jt. Record E[P_i P_j]_Sleep ≈ (1/T) Σ_t P^NCD_it P^NCD_jt
  (Eq 7.11).
- Update Δw_ij = ε (E[v_i P_j]_Wake − E[P_i P_j]_Sleep) (Eq 7.12).

Stacking RBMs (Section 7.4, Figs 7.4–7.5): train RBM 1 on the data, then
use its hidden probabilities as the data for RBM 2, and so on (greedy
layer-wise pre-training). The stack is then "unrolled" into a deep
autoencoder whose decoder uses the transposed weights, every binary unit is
replaced by its real-valued probability P, and the whole network is
fine-tuned with backprop. The book's MNIST network is 784-1000-500-250-30;
for the 12×12 ``aie.data.digits`` images we use e.g. 144-64-32-8.

Notes
-----
- Bias: the book adds a (V+1)th / (H+1)th unit with constant state −1
  (Eqs 7.7, 7.10). Here biases are explicit vectors ``b_h`` (hidden) and
  ``b_v`` (visible), with u_j = Σ_i w_ij v_i + b_hj. This is identical with
  b = −w_bias, and the bias update follows from Eq 7.12 applied to the bias
  unit: Δb_hj = ε(E[P_j]_Wake − E[P_j]_Sleep), Δb_vi = ε(E[v_i]_Wake −
  E[P_i]_Sleep). It also matches the ``aie.layers.Dense`` convention
  (weights (V, H), y = v W + b), which makes unrolling trivial.
- ``sample_visible=False`` uses the visible probabilities instead of binary
  visible samples inside the CD chain (as Hinton and Salakhutdinov's code
  does); the default follows the book's box and samples them.
- The book's DAN used linear units in the 30-unit code layer; here all
  units, including the code layer, are logistic for simplicity.
- Free energy F(v) = −v·b_v − Σ_j log(1 + e^{u_j}) satisfies
  p(v) = e^{−F(v)}/Z, so for tiny RBMs the exact log-likelihood is obtained
  by enumerating all 2^V visible vectors to compute log Z.
"""

from __future__ import annotations

import itertools

import numpy as np

from aie.activations import sigmoid
from aie.layers import SGD, Adam, Dense, Layer, Sequential, Sigmoid
from aie.losses import mean_sse, mean_sse_grad


class RBM:
    """Binary restricted Boltzmann machine trained by contrastive divergence."""

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        seed: int = 0,
        weight_scale: float = 0.01,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.W: np.ndarray = self.rng.standard_normal((n_visible, n_hidden)) * weight_scale
        self.b_v: np.ndarray = np.zeros(n_visible)
        self.b_h: np.ndarray = np.zeros(n_hidden)
        self.history: list[float] = []
        self._vel = [np.zeros_like(self.W), np.zeros_like(self.b_v), np.zeros_like(self.b_h)]

    @property
    def n_visible(self) -> int:
        return self.W.shape[0]

    @property
    def n_hidden(self) -> int:
        return self.W.shape[1]

    # ---- conditional distributions (Eqs 7.7, 7.8, 7.10) ----
    def hidden_probs(self, V: np.ndarray) -> np.ndarray:
        """P_j = σ(u_j), u_j = Σ_i w_ij v_i + b_hj (Eqs 7.7, 7.8)."""
        return sigmoid(V @ self.W + self.b_h)

    def visible_probs(self, H: np.ndarray) -> np.ndarray:
        """P_i = σ(u_i), u_i = Σ_j w_ij h_j + b_vi (Eqs 7.10, 7.8)."""
        return sigmoid(H @ self.W.T + self.b_v)

    def sample(self, P: np.ndarray) -> np.ndarray:
        """Binary states: a unit is on if a uniform random r < P (p. 89)."""
        return (self.rng.random(P.shape) < P).astype(float)

    # ---- contrastive divergence ----
    def cd_statistics(
        self, V: np.ndarray, n_cd: int = 1, sample_visible: bool = True
    ) -> dict[str, np.ndarray]:
        """Wake and sleep correlations for one batch (Eqs 7.9, 7.11)."""
        T = V.shape[0]
        Ph = self.hidden_probs(V)                      # wake P_jt
        wake = V.T @ Ph / T                            # E[v_i P_j]_Wake
        Ph_k = Ph
        for _ in range(n_cd):
            H = self.sample(Ph_k)                      # binary h^k
            Pv_k = self.visible_probs(H)               # P^k_it
            Vk = self.sample(Pv_k) if sample_visible else Pv_k
            Ph_k = self.hidden_probs(Vk)               # P^k_jt
        sleep = Pv_k.T @ Ph_k / T                      # E[P_i P_j]_Sleep
        return {
            "wake": wake,
            "sleep": sleep,
            "dW": wake - sleep,
            "db_v": V.mean(axis=0) - Pv_k.mean(axis=0),
            "db_h": Ph.mean(axis=0) - Ph_k.mean(axis=0),
        }

    def train_step(
        self,
        V: np.ndarray,
        lr: float = 0.1,
        n_cd: int = 1,
        momentum: float = 0.0,
        sample_visible: bool = True,
    ) -> None:
        """One CD-N_CD update, Δw_ij = ε(E[v_i P_j]_Wake − E[P_i P_j]_Sleep) (Eq 7.12)."""
        s = self.cd_statistics(V, n_cd, sample_visible)
        params = (self.W, self.b_v, self.b_h)
        grads = (s["dW"], s["db_v"], s["db_h"])
        for vel, p, g in zip(self._vel, params, grads, strict=True):
            vel *= momentum
            vel += lr * g
            p += vel

    def fit(
        self,
        X: np.ndarray,
        n_epochs: int = 10,
        lr: float = 0.1,
        batch_size: int = 20,
        n_cd: int = 1,
        momentum: float = 0.0,
        sample_visible: bool = True,
    ) -> RBM:
        """Mini-batch CD training; records reconstruction error per epoch."""
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        for _ in range(n_epochs):
            perm = self.rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                batch = X[perm[start : start + batch_size]]
                self.train_step(batch, lr, n_cd, momentum, sample_visible)
            self.history.append(self.reconstruction_error(X))
        return self

    # ---- monitoring ----
    def reconstruct(self, V: np.ndarray) -> np.ndarray:
        """Deterministic up-down pass v → P(h|v) → P(v|h)."""
        return self.visible_probs(self.hidden_probs(V))

    def reconstruction_error(self, V: np.ndarray) -> float:
        """mean_sse between data and its deterministic reconstruction."""
        return mean_sse(self.reconstruct(V), V)

    def free_energy(self, V: np.ndarray) -> np.ndarray:
        """F(v) = −v·b_v − Σ_j log(1 + exp(u_j)); p(v) ∝ exp(−F(v))."""
        return -(V @ self.b_v) - np.logaddexp(0.0, V @ self.W + self.b_h).sum(axis=-1)

    def log_partition(self) -> float:
        """Exact log Z by enumerating all 2^V visible vectors (tiny RBMs only)."""
        if self.n_visible > 20:
            raise ValueError("exact enumeration is only feasible for small n_visible")
        all_v = np.array(list(itertools.product([0.0, 1.0], repeat=self.n_visible)))
        neg_f = -self.free_energy(all_v)
        m = neg_f.max()
        return float(m + np.log(np.exp(neg_f - m).sum()))

    def log_likelihood(self, V: np.ndarray) -> float:
        """Exact mean log p(v) = −F(v) − log Z over the rows of V."""
        return float(np.mean(-self.free_energy(V)) - self.log_partition())

    def gibbs_chain(self, V0: np.ndarray, n_steps: int) -> np.ndarray:
        """Run block Gibbs sampling from V0; returns visible probabilities P(v|h) at the end."""
        V = V0
        for _ in range(n_steps):
            H = self.sample(self.hidden_probs(V))
            Pv = self.visible_probs(H)
            V = self.sample(Pv)
        return Pv


class RBMStack:
    """A stack of RBMs pre-trained greedily, layer by layer (Section 7.4, Fig 7.4)."""

    def __init__(self, layer_sizes: list[int], seed: int = 0, weight_scale: float = 0.01) -> None:
        self.layer_sizes = list(layer_sizes)
        self.rbms = [
            RBM(n_v, n_h, seed=seed + k, weight_scale=weight_scale)
            for k, (n_v, n_h) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:], strict=True))
        ]

    def pretrain(self, X: np.ndarray, **fit_kwargs) -> RBMStack:
        """Train RBM k on the hidden probabilities of RBM k−1 (Fig 7.4a–d)."""
        data = np.asarray(X, dtype=float).reshape(len(X), -1)
        for rbm in self.rbms:
            rbm.fit(data, **fit_kwargs)
            data = rbm.hidden_probs(data)
        return self

    def up(self, X: np.ndarray) -> np.ndarray:
        """Deterministic encoder pass using probabilities P (p. 95)."""
        for rbm in self.rbms:
            X = rbm.hidden_probs(X)
        return X

    def down(self, H: np.ndarray) -> np.ndarray:
        """Deterministic decoder pass through the transposed weights."""
        for rbm in reversed(self.rbms):
            H = rbm.visible_probs(H)
        return H

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self.down(self.up(X))


def unroll(stack: RBMStack) -> Sequential:
    """Build the deep autoencoder of Fig 7.5 from a pre-trained stack.

    Encoder layer k is Dense(W_k, b_h,k) + Sigmoid; the decoder mirrors it
    with Dense(W_kᵀ, b_v,k) + Sigmoid ("duplicated and flipped"). Weights are
    copied, so fine-tuning does not change the RBMs.
    """
    layers: list[Layer] = []
    for rbm in stack.rbms:
        d = Dense(rbm.n_visible, rbm.n_hidden)
        d.params["W"][...] = rbm.W
        d.params["b"][...] = rbm.b_h
        layers += [d, Sigmoid()]
    for rbm in reversed(stack.rbms):
        d = Dense(rbm.n_hidden, rbm.n_visible)
        d.params["W"][...] = rbm.W.T
        d.params["b"][...] = rbm.b_v
        layers += [d, Sigmoid()]
    return Sequential(*layers)


def random_autoencoder(layer_sizes: list[int], seed: int = 0) -> Sequential:
    """Same architecture as :func:`unroll`, but randomly initialised (no pre-training)."""
    rng = np.random.default_rng(seed)
    sizes = list(layer_sizes) + list(reversed(layer_sizes[:-1]))
    layers: list[Layer] = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=True):
        layers += [Dense(n_in, n_out, rng), Sigmoid()]
    return Sequential(*layers)


def encode(model: Sequential, X: np.ndarray) -> np.ndarray:
    """Code-layer activity: run the first half of an unrolled autoencoder."""
    half = len(model.layers) // 2
    for layer in model.layers[:half]:
        X = layer.forward(X)
    return X


def fine_tune(
    model: Sequential,
    X: np.ndarray,
    n_epochs: int = 50,
    lr: float = 0.1,
    batch_size: int = 20,
    optimizer: str = "sgd",
    momentum: float = 0.9,
    seed: int = 0,
) -> list[float]:
    """Backprop fine-tuning on mean_sse(x′, x) (Section 7.4); returns per-epoch error."""
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    rng = np.random.default_rng(seed)
    opt = Adam(model, lr) if optimizer == "adam" else SGD(model, lr, momentum)
    history = [mean_sse(model.forward(X), X)]
    for _ in range(n_epochs):
        perm = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            batch = X[perm[start : start + batch_size]]
            out = model.forward(batch)
            model.backward(mean_sse_grad(out, batch))
            opt.step()
        history.append(mean_sse(model.forward(X), X))
    return history
