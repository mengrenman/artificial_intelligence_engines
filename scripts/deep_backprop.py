"""Deep backprop networks — Chapter 9 of Stone (2020).

Small, runnable versions of the chapter's main ideas, built from the layer
library in :mod:`aie.layers`:

- Convolutional networks (Sections 9.2–9.4, Eq 9.1–9.2): a LeNet-style CNN
  for the 12×12 :func:`aie.data.digits` images, plus an MLP baseline.
- ReLU units and vanishing gradients (Section 9.5, Fig 9.8) and residual
  networks (Section 9.7, Eq 9.3, Fig 9.9): a :class:`Residual` block
  y = x + F(x) and a per-layer gradient-norm probe for deep MLPs.
- Denoising autoencoders (Section 9.9).
- Fooling neural networks (Section 9.10, Eq 9.4–9.7): the gradient of the
  error with respect to the *input* and the fast gradient sign method.
- Generative adversarial networks (Section 9.11, Eq 9.8–9.10) on a 2-D toy
  distribution.

Conventions
-----------
- Image batches have shape (T, 1, H, W); vector batches (T, n).
- Classifiers output logits and are trained with
  :func:`aie.losses.softmax_cross_entropy` (Section 7.4's softmax output).
- Reconstruction error is :func:`aie.losses.mean_sse` (Eq 4.8, averaged).

Notes
-----
- LeNet1 (Fig 9.5) uses tanh units, 5×5 kernels with stride 2 and no
  pooling on 16×16 images. Our images are 12×12, so :func:`lenet` uses 3×3
  "same" convolutions each followed by 2×2 pooling (max pooling as in
  AlexNet, or average pooling as in LeNet5's sub-sampling), and ReLU or
  tanh units. Layer sizes are scaled down accordingly.
- The book writes the residual mapping as y = F(x) = H(x) − x (Eq 9.3); the
  block's *output* is H(x) = x + F(x) (Fig 9.9), which is what
  :class:`Residual` computes.
- Section 9.10 writes FGSM as ϵ sign(∇ₓE) (the "∆w" there is a typo for
  ∆x); :func:`fgsm` also clips the result to the valid pixel range [0, 1].
- For the GAN generator we minimise the *non-saturating* loss
  −(1/m) Σ log D(G(zᵢ)) rather than Eq 9.10's (1/m) Σ log(1 − D(G(zᵢ))).
  Both push D(G(z)) towards 1, but Eq 9.10 has vanishing gradient early in
  training, when D confidently rejects the fakes (Goodfellow et al., 2014,
  recommend the same switch). The discriminator maximises Eq 9.9 by
  minimising its negative, i.e. the binary cross-entropy of D's logits.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aie.activations import sigmoid
from aie.layers import (
    Adam,
    AvgPool2D,
    Conv2D,
    Dense,
    Flatten,
    Layer,
    MaxPool2D,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)
from aie.losses import mean_sse, mean_sse_grad, softmax_cross_entropy

# ---------------------------------------------------------------------------
# 9.2–9.4 Convolutional networks
# ---------------------------------------------------------------------------


def lenet(
    rng: np.random.Generator | None = None,
    n_maps: tuple[int, int] = (6, 12),
    n_hidden: int = 30,
    activation: str = "relu",
    pool: str = "max",
    size: int = 12,
) -> Sequential:
    """A LeNet-style CNN for (T, 1, size, size) images with 10 output logits.

    conv3×3 → act → pool2 → conv3×3 → act → pool2 → dense(n_hidden) → act →
    dense(10). Each conv layer is a bank of feature maps with shared weights
    (Eq 9.2); pooling halves the resolution (Section 9.4). ``size`` must be
    divisible by 4.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    act = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid}[activation]
    pool_cls = {"max": MaxPool2D, "avg": AvgPool2D}[pool]
    c1, c2 = n_maps
    side = size // 4
    init = "he" if activation == "relu" else "xavier"
    return Sequential(
        Conv2D(1, c1, 3, padding=1, rng=rng),
        act(),
        pool_cls(2),
        Conv2D(c1, c2, 3, padding=1, rng=rng),
        act(),
        pool_cls(2),
        Flatten(),
        Dense(c2 * side * side, n_hidden, rng, init=init),
        act(),
        Dense(n_hidden, 10, rng),
    )


def mlp_classifier(
    n_in: int = 144, n_hidden: int = 64, rng: np.random.Generator | None = None
) -> Sequential:
    """Fully connected baseline: flatten → dense → ReLU → dense(10)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    return Sequential(Flatten(), Dense(n_in, n_hidden, rng, init="he"), ReLU(),
                      Dense(n_hidden, 10, rng))


def as_images(X: np.ndarray) -> np.ndarray:
    """(T, H, W) → (T, 1, H, W), the single-channel layout Conv2D expects."""
    return X[:, None, :, :] if X.ndim == 3 else X


def predict_logits(model: Layer, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Forward pass in batches (keeps memory small for conv layers)."""
    return np.concatenate(
        [model.forward(X[i : i + batch_size]) for i in range(0, len(X), batch_size)]
    )


def accuracy(model: Layer, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of inputs whose largest logit is the correct label."""
    return float(np.mean(predict_logits(model, X).argmax(axis=1) == y))


def train_classifier(
    model: Sequential,
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-2,
    seed: int = 0,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Minibatch training with Adam on the softmax cross-entropy.

    Returns a history dict with per-epoch mean training ``loss``, training
    accuracy ``acc`` and, if validation data is given, ``val_acc``.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(model, lr=lr)
    hist: dict[str, list[float]] = {"loss": [], "acc": [], "val_acc": []}
    n = len(X)
    for _ in range(n_epochs):
        perm = rng.permutation(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss, grad = softmax_cross_entropy(model.forward(X[idx]), y[idx])
            model.backward(grad)
            opt.step()
            total += loss * len(idx)
        hist["loss"].append(total / n)
        hist["acc"].append(accuracy(model, X, y))
        if X_val is not None and y_val is not None:
            hist["val_acc"].append(accuracy(model, X_val, y_val))
    return hist


# ---------------------------------------------------------------------------
# 9.5 & 9.7 ReLU, vanishing gradients and residual networks
# ---------------------------------------------------------------------------


class Residual(Sequential):
    """Residual block, H(x) = x + F(x) (Eq 9.3, Fig 9.9).

    ``F`` is the stack of ``layers`` (which must preserve the input shape).
    By the chain rule, dE/dx = dE/dH + (dE/dH)(∂F/∂x): the identity "skip"
    path carries the gradient back unattenuated, which is why very deep
    ResNets remain trainable.

    It subclasses :class:`~aie.layers.Sequential` only so that
    ``Sequential.parameters()`` (and hence the optimisers) find the
    parameters of the inner layers.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x + super().forward(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad + super().backward(grad)


def deep_mlp(
    kind: str,
    depth: int = 20,
    width: int = 32,
    n_in: int = 144,
    n_out: int = 10,
    rng: np.random.Generator | None = None,
) -> Sequential:
    """A deep fully connected network of ``depth`` hidden layers.

    kind = "sigmoid": Dense → Sigmoid repeated (Xavier init);
    kind = "relu":    Dense → ReLU repeated (He init);
    kind = "residual": an input projection followed by ``depth − 1``
    blocks Residual(Dense → ReLU), whose branch weights are scaled by
    1/√depth so that the forward activations do not grow with depth.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    layers: list[Layer] = []
    if kind in ("sigmoid", "relu"):
        act = Sigmoid if kind == "sigmoid" else ReLU
        init = "xavier" if kind == "sigmoid" else "he"
        sizes = [n_in] + [width] * depth
        for a, b in zip(sizes[:-1], sizes[1:], strict=True):
            layers += [Dense(a, b, rng, init=init), act()]
    elif kind == "residual":
        layers += [Dense(n_in, width, rng, init="he"), ReLU()]
        for _ in range(depth - 1):
            d = Dense(width, width, rng, init="he")
            d.params["W"] /= np.sqrt(depth)
            layers.append(Residual(d, ReLU()))
    else:
        raise ValueError(f"unknown kind {kind!r}")
    layers.append(Dense(width, n_out, rng))
    return Sequential(*layers)


def dense_layers(model: Sequential) -> list[Dense]:
    """All Dense layers of ``model`` (including inside residual blocks), in order."""
    out: list[Dense] = []
    for layer in model.layers:
        if isinstance(layer, Sequential):
            out.extend(dense_layers(layer))
        elif isinstance(layer, Dense):
            out.append(layer)
    return out


def layer_gradient_norms(model: Sequential, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """‖∂E/∂W‖ for every Dense layer, from input side to output side.

    One forward/backward pass of the softmax cross-entropy. Vanishing
    gradients (Section 9.5) show up as norms that shrink towards the input.
    """
    _, grad = softmax_cross_entropy(model.forward(X), y)
    model.backward(grad)
    return np.array([np.linalg.norm(d.grads["W"]) for d in dense_layers(model)])


# ---------------------------------------------------------------------------
# 9.9 Denoising autoencoders
# ---------------------------------------------------------------------------


def corrupt(
    X: np.ndarray,
    rng: np.random.Generator,
    drop_prob: float = 0.3,
    noise_std: float = 0.2,
) -> np.ndarray:
    """Corrupt inputs as in Vincent et al. (2010): zero a random fraction of
    pixels ("masking noise") and add Gaussian noise, clipped to [0, 1]."""
    keep = rng.random(X.shape) >= drop_prob
    return np.clip(X * keep + rng.normal(0.0, noise_std, X.shape), 0.0, 1.0)


def autoencoder(
    n_in: int = 144, n_hidden: int = 48, rng: np.random.Generator | None = None
) -> Sequential:
    """Dense(n_in → n_hidden) → ReLU → Dense(n_hidden → n_in) → Sigmoid."""
    rng = rng if rng is not None else np.random.default_rng(0)
    return Sequential(Dense(n_in, n_hidden, rng, init="he"), ReLU(),
                      Dense(n_hidden, n_in, rng), Sigmoid())


def train_denoising_autoencoder(
    model: Sequential,
    X: np.ndarray,
    n_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 0,
    drop_prob: float = 0.3,
    noise_std: float = 0.2,
) -> list[float]:
    """Train ``model`` to map corrupt(x) → x by minimising mean_sse.

    ``X`` is (T, n) clean vectors; fresh corruption is drawn every epoch.
    Returns the per-epoch mean training error.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(model, lr=lr)
    hist: list[float] = []
    n = len(X)
    for _ in range(n_epochs):
        perm = rng.permutation(n)
        total = 0.0
        for i in range(0, n, batch_size):
            clean = X[perm[i : i + batch_size]]
            recon = model.forward(corrupt(clean, rng, drop_prob, noise_std))
            model.backward(mean_sse_grad(recon, clean))
            opt.step()
            total += mean_sse(recon, clean) * len(clean)
        hist.append(total / n)
    return hist


# ---------------------------------------------------------------------------
# 9.10 Fooling neural networks
# ---------------------------------------------------------------------------


def input_gradient(model: Layer, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Loss E and ∇ₓE, the gradient of the error w.r.t. the *input* (Eq 9.6).

    Backprop already computes this as a by-product: it is what
    ``Sequential.backward`` returns.
    """
    loss, grad = softmax_cross_entropy(model.forward(X), y)
    return loss, model.backward(grad)


def fgsm(model: Layer, X: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    """Fast gradient sign method: x' = clip(x + ϵ sign(∇ₓE), 0, 1) (Section 9.10).

    Each pixel moves by at most ϵ, in the direction that increases the error.
    """
    _, g = input_gradient(model, X, y)
    return np.clip(X + eps * np.sign(g), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 9.11 Generative adversarial networks
# ---------------------------------------------------------------------------


def bce_with_logits(logits: np.ndarray, targets: np.ndarray | float) -> tuple[float, np.ndarray]:
    """Mean binary cross-entropy of D = σ(a) against targets, and dL/da.

    L = −mean[t log σ(a) + (1 − t) log(1 − σ(a))], computed stably as
    max(a, 0) − a t + log(1 + e^{−|a|}); its gradient is (σ(a) − t)/m.
    """
    a = np.asarray(logits, dtype=float)
    t = np.broadcast_to(np.asarray(targets, dtype=float), a.shape)
    loss = np.maximum(a, 0.0) - a * t + np.log1p(np.exp(-np.abs(a)))
    return float(np.mean(loss)), (sigmoid(a) - t) / a.size


def ring_of_gaussians(
    n: int, rng: np.random.Generator, n_modes: int = 8, radius: float = 2.0, std: float = 0.05
) -> np.ndarray:
    """Samples from a mixture of ``n_modes`` Gaussians spaced evenly on a circle."""
    k = rng.integers(0, n_modes, n)
    angle = 2 * np.pi * k / n_modes
    centres = radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    return centres + std * rng.standard_normal((n, 2))


def gan_mlp(
    n_in: int, n_hidden: int, n_out: int, rng: np.random.Generator, n_layers: int = 2
) -> Sequential:
    """Small MLP with ``n_layers`` ReLU hidden layers and a linear output."""
    layers: list[Layer] = []
    a = n_in
    for _ in range(n_layers):
        layers += [Dense(a, n_hidden, rng, init="he"), ReLU()]
        a = n_hidden
    layers.append(Dense(a, n_out, rng))
    return Sequential(*layers)


class GAN:
    """Generator G(z) and discriminator D(x) = σ(logit(x)) (Fig 9.13).

    The discriminator network outputs a *logit*; D is its sigmoid, so the
    cross-entropy and its gradient can be computed stably by
    :func:`bce_with_logits`.
    """

    def __init__(
        self,
        z_dim: int = 2,
        x_dim: int = 2,
        n_hidden: int = 64,
        lr_d: float = 1e-3,
        lr_g: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.z_dim = z_dim
        self.G = gan_mlp(z_dim, n_hidden, x_dim, self.rng)
        self.D = gan_mlp(x_dim, n_hidden, 1, self.rng)
        self.opt_d = Adam(self.D, lr=lr_d, betas=(0.5, 0.999))
        self.opt_g = Adam(self.G, lr=lr_g, betas=(0.5, 0.999))

    def sample_z(self, m: int) -> np.ndarray:
        """Noise inputs z ~ p_z(z) = N(0, I)."""
        return self.rng.standard_normal((m, self.z_dim))

    def generate(self, z: np.ndarray) -> np.ndarray:
        return self.G.forward(z)

    def discriminate(self, x: np.ndarray) -> np.ndarray:
        """D(x), the estimated probability that x came from the training data."""
        return sigmoid(self.D.forward(x)).ravel()

    def discriminator_loss(self, x_real: np.ndarray, z: np.ndarray) -> float:
        """−E_D of Eq 9.9: −mean log D(x) − mean log(1 − D(G(z)))."""
        l_real, _ = bce_with_logits(self.D.forward(x_real), 1.0)
        l_fake, _ = bce_with_logits(self.D.forward(self.G.forward(z)), 0.0)
        return l_real + l_fake

    def discriminator_gradients(self, x_real: np.ndarray, z: np.ndarray) -> float:
        """Fill ``self.D``'s grads with ∂(−E_D)/∂θ_D; return the loss.

        The real and fake batches are separate forward passes, so their
        parameter gradients are summed.
        """
        x_fake = self.G.forward(z)
        l_real, g = bce_with_logits(self.D.forward(x_real), 1.0)
        self.D.backward(g)
        g_real = [layer.grads[name].copy() for _, _, layer, name in self.D.parameters()]
        l_fake, g = bce_with_logits(self.D.forward(x_fake), 0.0)
        self.D.backward(g)
        for gr, (_, _, layer, name) in zip(g_real, self.D.parameters(), strict=True):
            layer.grads[name] = layer.grads[name] + gr
        return l_real + l_fake

    def generator_loss(self, z: np.ndarray) -> float:
        """Non-saturating generator loss −mean log D(G(z)) (see Notes)."""
        loss, _ = bce_with_logits(self.D.forward(self.G.forward(z)), 1.0)
        return loss

    def generator_gradients(self, z: np.ndarray) -> float:
        """Fill ``self.G``'s grads with ∂L_G/∂θ_G by backprop *through* D.

        D's own grads are overwritten as a side effect but are not used.
        """
        loss, g = bce_with_logits(self.D.forward(self.G.forward(z)), 1.0)
        self.G.backward(self.D.backward(g))
        return loss

    def train(
        self,
        sample_real: Callable[[int, np.random.Generator], np.ndarray],
        n_steps: int = 2000,
        batch_size: int = 128,
        k_d: int = 1,
    ) -> dict[str, list[float]]:
        """Alternate ``k_d`` discriminator steps with one generator step
        (Section 9.11). Returns per-step losses."""
        hist: dict[str, list[float]] = {"d_loss": [], "g_loss": []}
        for _ in range(n_steps):
            for _ in range(k_d):
                d_loss = self.discriminator_gradients(
                    sample_real(batch_size, self.rng), self.sample_z(batch_size)
                )
                self.opt_d.step()
            g_loss = self.generator_gradients(self.sample_z(batch_size))
            self.opt_g.step()
            hist["d_loss"].append(d_loss)
            hist["g_loss"].append(g_loss)
        return hist
