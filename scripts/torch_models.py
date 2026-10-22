"""Optional PyTorch counterparts of the Chapter 8 VAE and Chapter 9 CNN.

The NumPy code in :mod:`scripts.vae` and :mod:`scripts.deep_backprop` spells
out every backward pass by hand. Here the same two models are written with
``torch.nn`` and trained with automatic differentiation, so the hand-derived
gradients can be checked against autograd and the models trained faster:

- :class:`TorchVAE` mirrors :class:`scripts.vae.VAE`: an MLP encoder that
  outputs [μ, log σ²], the reparametrisation z = μ + η σ (Section 8.6), a
  sigmoid-output decoder, and the objective L = L^dec + L^enc of Eq 8.37
  (Eq 8.35 with decoder variance σ_x², Eq 8.36 for the KL term).
- :class:`TorchLeNet` mirrors :func:`scripts.deep_backprop.lenet`, the
  scaled-down LeNet of Sections 9.2–9.4: conv3×3 → act → pool2 → conv3×3 →
  act → pool2 → dense → act → dense(10).

:func:`load_numpy_weights` copies the parameters of a NumPy model built
from :mod:`aie.layers` into the matching torch module, and
:func:`torch_grads_as_numpy` returns autograd's gradients in the NumPy
layout, so that forward passes and gradients can be compared directly
(they agree to ~1e-12 in float64).

Notes
-----
- ``torch`` is a hard import: callers should guard with
  ``pytest.importorskip("torch")`` or ``try: import torch``.
- Layouts: NumPy ``Dense`` stores W as (n_in, n_out) (y = x W + b) whereas
  ``nn.Linear`` stores the transpose (n_out, n_in). ``Conv2D`` and
  ``nn.Conv2d`` both use (out, in, k, k) and both compute a
  cross-correlation, so conv weights copy unchanged.
- Initialisation mimics :mod:`aie.layers`: Dense weights ~ N(0, 1/n_in)
  ("xavier") or N(0, 2/n_in) ("he"), conv weights ~ N(0, 2/fan_in), zero
  biases, drawn from a seeded ``torch.Generator``. The random numbers
  differ from NumPy's, so models with the same seed are *not* identical;
  use :func:`load_numpy_weights` (or the ``from_numpy`` constructors) for
  that.
- As in the NumPy code, losses are batch *means* and both models are
  trained with Adam (``torch.optim.Adam``).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from aie import layers as npl

_ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU,
                                            "sigmoid": nn.Sigmoid}
_NP_ACTIVATIONS = {npl.Tanh: "tanh", npl.ReLU: "relu", npl.Sigmoid: "sigmoid"}


def _as_tensor(X: np.ndarray | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.asarray(X) if not torch.is_tensor(X) else X, dtype=dtype)


def _init_(module: nn.Module, gen: torch.Generator, init: str) -> None:
    """Re-initialise Linear/Conv2d weights the way :mod:`aie.layers` does."""
    with torch.no_grad():
        if isinstance(module, nn.Linear):
            scale = np.sqrt((2.0 if init == "he" else 1.0) / module.in_features)
        elif isinstance(module, nn.Conv2d):
            scale = np.sqrt(2.0 / (module.weight[0].numel()))
        else:
            return
        w = torch.randn(module.weight.shape, generator=gen, dtype=torch.float64) * scale
        module.weight.copy_(w)
        module.bias.zero_()


def _param_modules(module: nn.Module) -> list[nn.Module]:
    return [m for m in module.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]


def _param_layers(model: npl.Sequential) -> list[npl.Layer]:
    out: list[npl.Layer] = []
    for layer in model.layers:
        if isinstance(layer, npl.Sequential):
            out.extend(_param_layers(layer))
        elif isinstance(layer, (npl.Dense, npl.Conv2D)):
            out.append(layer)
    return out


def paired_layers(module: nn.Module, model: npl.Sequential) -> list[tuple[nn.Module, npl.Layer]]:
    """(torch layer, NumPy layer) pairs for every Linear/Dense and Conv2d/Conv2D, in order."""
    mods, lays = _param_modules(module), _param_layers(model)
    if len(mods) != len(lays):
        raise ValueError(f"{len(mods)} torch layers vs {len(lays)} NumPy layers")
    for m, lay in zip(mods, lays, strict=True):
        if isinstance(m, nn.Linear) != isinstance(lay, npl.Dense):
            raise TypeError(f"layer mismatch: {type(m).__name__} vs {type(lay).__name__}")
    return list(zip(mods, lays, strict=True))


def load_numpy_weights(module: nn.Module, model: npl.Sequential) -> nn.Module:
    """Copy the parameters of a NumPy ``aie.layers`` model into ``module``.

    Dense W (n_in, n_out) is transposed into nn.Linear's (n_out, n_in);
    Conv2D weights copy unchanged. Returns ``module``.
    """
    with torch.no_grad():
        for m, lay in paired_layers(module, model):
            W = lay.params["W"].T if isinstance(m, nn.Linear) else lay.params["W"]
            m.weight.copy_(torch.as_tensor(W, dtype=m.weight.dtype))
            m.bias.copy_(torch.as_tensor(lay.params["b"], dtype=m.bias.dtype))
    return module


def torch_grads_as_numpy(module: nn.Module) -> list[dict[str, np.ndarray]]:
    """Autograd's ``.grad`` of every Linear/Conv2d, in the NumPy layer layout.

    The list lines up with the parametrised layers of the NumPy model, so
    entry k can be compared with ``layer.grads`` of the k-th Dense/Conv2D.
    """
    out = []
    for m in _param_modules(module):
        W = m.weight.grad.detach().numpy()
        out.append({"W": W.T if isinstance(m, nn.Linear) else W,
                    "b": m.bias.grad.detach().numpy()})
    return out


# ---------------------------------------------------------------------------
# Chapter 8: variational autoencoder
# ---------------------------------------------------------------------------


class TorchVAE(nn.Module):
    """MLP VAE with the architecture and objective of :class:`scripts.vae.VAE`.

    Encoder n_input → hidden… → 2J (outputs [μ, log σ²]); decoder
    J → reversed(hidden)… → n_input → Sigmoid. ``loss`` returns E = −L,
    the batch mean of minus Eq 8.37 with decoder variance ``sigma_x2``.
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int = 2,
        hidden: tuple[int, ...] = (64,),
        activation: str = "tanh",
        seed: int = 0,
        sigma_x2: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        act = _ACTIVATIONS[activation]
        init = "he" if activation == "relu" else "xavier"
        self.n_latent, self.sigma_x2, self.seed = n_latent, sigma_x2, seed

        enc: list[nn.Module] = []
        n_prev = n_input
        for n_h in hidden:
            enc += [nn.Linear(n_prev, n_h), act()]
            n_prev = n_h
        enc.append(nn.Linear(n_prev, 2 * n_latent))       # outputs [μ, log σ²]
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        n_prev = n_latent
        for n_h in reversed(hidden):
            dec += [nn.Linear(n_prev, n_h), act()]
            n_prev = n_h
        dec += [nn.Linear(n_prev, n_input), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

        gen = torch.Generator().manual_seed(seed)
        for seq in (self.encoder, self.decoder):
            for i, m in enumerate(seq):
                # Output layers (no activation after them) use the "xavier" scale, as in NumPy.
                is_hidden = i + 1 < len(seq) and not isinstance(seq[i + 1], nn.Sigmoid)
                _init_(m, gen, init if is_hidden else "xavier")
        self.to(dtype)
        self.history: dict[str, list[float]] = {"loss": [], "recon": [], "kl": []}

    @classmethod
    def from_numpy(cls, vae, dtype: torch.dtype = torch.float64) -> TorchVAE:
        """A TorchVAE with the same architecture and weights as a NumPy ``VAE``."""
        dense = [lay for lay in vae.encoder.layers if isinstance(lay, npl.Dense)]
        hidden = tuple(d.params["W"].shape[1] for d in dense[:-1])
        acts = [lay for lay in vae.encoder.layers if type(lay) in _NP_ACTIVATIONS]
        activation = _NP_ACTIVATIONS[type(acts[0])] if acts else "tanh"
        tv = cls(dense[0].params["W"].shape[0], vae.n_latent, hidden, activation,
                 sigma_x2=vae.sigma_x2, dtype=dtype)
        load_numpy_weights(tv.encoder, vae.encoder)
        load_numpy_weights(tv.decoder, vae.decoder)
        return tv

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder outputs (μ, log σ²), each (T, J)."""
        out = self.encoder(x)
        return out[:, : self.n_latent], out[:, self.n_latent :]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def terms(
        self,
        x: torch.Tensor,
        eta: torch.Tensor | None = None,
        n_z: int = 1,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Batch means of L^dec (``recon``), L^enc (``kl``) and L = L^dec + L^enc.

        ``eta`` is the reparametrisation noise, shape (N_z, T, J); if None it
        is drawn from N(0, I) with ``n_z`` samples per input.
        """
        mu, logvar = self.encode(x)
        sigma = torch.exp(0.5 * logvar)
        if eta is None:
            eta = torch.randn((n_z, *mu.shape), generator=generator, dtype=mu.dtype)
        n_z, T, J = eta.shape
        z = (mu[None] + eta * sigma[None]).reshape(n_z * T, J)          # z = μ + η σ
        xr = self.decode(z).reshape(n_z, T, -1)
        l_dec = -((x[None] - xr) ** 2).sum(dim=(0, 2)) / (2 * n_z * self.sigma_x2)  # Eq 8.35
        l_enc = 0.5 * (1 + logvar - logvar.exp() - mu**2).sum(dim=-1)              # Eq 8.36
        recon, kl = l_dec.mean(), l_enc.mean()
        return {"L": recon + kl, "recon": recon, "kl": kl}

    def loss(self, x, eta=None, n_z: int = 1, generator: torch.Generator | None = None):
        """E = −L, the quantity minimised (a scalar tensor; call ``.backward()``)."""
        x = _as_tensor(x, self.dtype).reshape(len(x), -1)
        if eta is not None:
            eta = _as_tensor(eta, self.dtype)
        return -self.terms(x, eta, n_z, generator)["L"]

    def fit(
        self,
        X,
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        seed: int = 0,
        n_z: int = 1,
    ) -> dict[str, list[float]]:
        """Minimise E = −L with Adam on shuffled mini-batches.

        Returns (and appends to ``self.history``) per-epoch means of the
        positive costs ``loss`` = −L, ``recon`` = −L^dec, ``kl`` = KL(q‖p).
        """
        X = _as_tensor(X, self.dtype).reshape(len(X), -1)
        gen = torch.Generator().manual_seed(seed)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(n_epochs):
            perm = torch.randperm(len(X), generator=gen)
            sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
            for start in range(0, len(X), batch_size):
                batch = X[perm[start : start + batch_size]]
                t = self.terms(batch, n_z=n_z, generator=gen)
                opt.zero_grad()
                (-t["L"]).backward()
                opt.step()
                for k, key in (("loss", "L"), ("recon", "recon"), ("kl", "kl")):
                    sums[k] += -t[key].item() * len(batch)
            for k, v in sums.items():
                self.history[k].append(v / len(X))
        return self.history

    @torch.no_grad()
    def encode_numpy(self, X) -> tuple[np.ndarray, np.ndarray]:
        """(μ, log σ²) as NumPy arrays, for plotting the latent space."""
        mu, logvar = self.encode(_as_tensor(X, self.dtype).reshape(len(X), -1))
        return mu.numpy(), logvar.numpy()

    @torch.no_grad()
    def decode_numpy(self, Z) -> np.ndarray:
        return self.decode(_as_tensor(Z, self.dtype)).numpy()


# ---------------------------------------------------------------------------
# Chapter 9: convolutional network
# ---------------------------------------------------------------------------


class TorchLeNet(nn.Module):
    """The scaled-down LeNet of :func:`scripts.deep_backprop.lenet` (Sections 9.2–9.4).

    conv3×3 (padding 1) → act → pool2 → conv3×3 → act → pool2 → flatten →
    dense(n_hidden) → act → dense(10 logits). ``self.net`` has one module per
    NumPy layer, in the same order.
    """

    def __init__(
        self,
        n_maps: tuple[int, int] = (6, 12),
        n_hidden: int = 30,
        activation: str = "relu",
        pool: str = "max",
        size: int = 12,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        act = _ACTIVATIONS[activation]
        pool_cls = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}[pool]
        c1, c2 = n_maps
        side = size // 4
        self.net = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), act(), pool_cls(2),
            nn.Conv2d(c1, c2, 3, padding=1), act(), pool_cls(2),
            nn.Flatten(),
            nn.Linear(c2 * side * side, n_hidden), act(),
            nn.Linear(n_hidden, 10),
        )
        gen = torch.Generator().manual_seed(seed)
        init = "he" if activation == "relu" else "xavier"
        for m in self.net:
            _init_(m, gen, "xavier" if m is self.net[-1] else init)
        self.to(dtype)

    @classmethod
    def from_numpy(cls, model: npl.Sequential, dtype: torch.dtype = torch.float64) -> TorchLeNet:
        """A TorchLeNet with the architecture and weights of a NumPy ``lenet()``."""
        conv1, conv2, d1, _ = _param_layers(model)
        n_maps = (conv1.params["W"].shape[0], conv2.params["W"].shape[0])
        act = _NP_ACTIVATIONS[type(model.layers[1])]
        pool = "max" if isinstance(model.layers[2], npl.MaxPool2D) else "avg"
        side = int(round(np.sqrt(d1.params["W"].shape[0] / n_maps[1])))
        tl = cls(n_maps, d1.params["W"].shape[1], act, pool, 4 * side, dtype=dtype)
        load_numpy_weights(tl.net, model)
        return tl

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(T, 1, H, W) or (T, H, W) images → (T, 10) logits."""
        return self.net(x[:, None] if x.ndim == 3 else x)

    @torch.no_grad()
    def predict_logits(self, X) -> np.ndarray:
        self.eval()
        return self(_as_tensor(X, self.dtype)).numpy()

    def predict(self, X) -> np.ndarray:
        """Predicted labels: the index of the largest logit."""
        return self.predict_logits(X).argmax(axis=1)

    def accuracy(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def fit(
        self,
        X,
        y,
        n_epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-2,
        seed: int = 0,
        X_val=None,
        y_val=None,
    ) -> dict[str, list[float]]:
        """Adam on the mean softmax cross-entropy, as in ``train_classifier``.

        Returns per-epoch mean training ``loss``, training accuracy ``acc``
        and, if validation data is given, ``val_acc``.
        """
        Xt = _as_tensor(X, self.dtype)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
        gen = torch.Generator().manual_seed(seed)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        hist: dict[str, list[float]] = {"loss": [], "acc": [], "val_acc": []}
        for _ in range(n_epochs):
            self.train()
            perm = torch.randperm(len(Xt), generator=gen)
            total = 0.0
            for i in range(0, len(Xt), batch_size):
                idx = perm[i : i + batch_size]
                loss = F.cross_entropy(self(Xt[idx]), yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            hist["loss"].append(total / len(Xt))
            hist["acc"].append(self.accuracy(X, y))
            if X_val is not None and y_val is not None:
                hist["val_acc"].append(self.accuracy(X_val, y_val))
        return hist
