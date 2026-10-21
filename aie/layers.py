"""A minimal layer library for the deep-learning chapters (7–9).

Chapters 2–4 spell out every matrix product by hand. From Chapter 7 on the
networks get deep enough that the same forward/backward bookkeeping is
repeated many times, so it is factored out here. Each layer implements

    forward(x)        -> y, caching what backward() needs
    backward(dE/dy)   -> dE/dx, storing dE/dθ for its own parameters θ

which is exactly the chain rule of Section 4.2 applied one layer at a time.

Conventions
-----------
- Batch dimension first. Dense inputs are (T, n_in); image inputs are
  (T, channels, height, width).
- Dense weights have shape (n_in, n_out) so that y = x @ W + b. This is the
  transpose of the book's W (K, I) convention used in ``scripts/linear.py``
  and ``scripts/backprop.py``; it keeps batched code free of transposes.
- Biases are explicit vectors ``b`` rather than an extra input fixed at −1.
  The two are equivalent (b = −w_bias).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from aie.activations import sigmoid


class Layer:
    """Base class. Subclasses override ``forward``/``backward``."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Dense(Layer):
    """Fully connected layer, y = x W + b.

    ``init`` selects the weight scale: "xavier" (std √(1/n_in), suits
    sigmoid/tanh) or "he" (std √(2/n_in), suits ReLU; Section 9.5).
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        rng: np.random.Generator | None = None,
        init: str = "xavier",
    ) -> None:
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng(0)
        if init == "he":
            scale = np.sqrt(2.0 / n_in)
        elif init == "xavier":
            scale = np.sqrt(1.0 / n_in)
        else:
            raise ValueError(f"unknown init {init!r}")
        self.params = {"W": rng.standard_normal((n_in, n_out)) * scale, "b": np.zeros(n_out)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grads["W"] = self._x.T @ grad
        self.grads["b"] = grad.sum(axis=0)
        return grad @ self.params["W"].T


class Sigmoid(Layer):
    """Logistic activation (Eq 4.1); dy/du = y(1 − y) (Eq 4.2)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._y = sigmoid(x)
        return self._y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._y * (1.0 - self._y)


class Tanh(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._y = np.tanh(x)
        return self._y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1.0 - self._y**2)


class ReLU(Layer):
    """Rectified linear unit (Section 9.5)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask


class Flatten(Layer):
    """Reshape (T, C, H, W) feature maps into (T, C·H·W) vectors."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self._shape)


class Conv2D(Layer):
    """2-D convolution (strictly, cross-correlation) with shared weights.

    Each of the ``out_channels`` feature maps is produced by one k×k×C
    kernel slid over the input (Section 9.2): every unit in a feature map
    shares the same weights and bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng(0)
        fan_in = in_channels * kernel_size * kernel_size
        self.k, self.stride, self.padding = kernel_size, stride, padding
        self.params = {
            "W": rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size))
            * np.sqrt(2.0 / fan_in),
            "b": np.zeros(out_channels),
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x: np.ndarray) -> np.ndarray:
        p, s, k = self.padding, self.stride, self.k
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p else x
        # windows[n, c, i, j, a, b] = xp[n, c, i*s + a, j*s + b]
        windows = sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::s, ::s]
        self._xp_shape, self._windows = xp.shape, windows
        out = np.einsum("nchwij,ocij->nohw", windows, self.params["W"], optimize=True)
        return out + self.params["b"][None, :, None, None]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        p, s, k = self.padding, self.stride, self.k
        W = self.params["W"]
        self.grads["W"] = np.einsum("nohw,nchwij->ocij", grad, self._windows, optimize=True)
        self.grads["b"] = grad.sum(axis=(0, 2, 3))
        dxp = np.zeros(self._xp_shape)
        Ho, Wo = grad.shape[2], grad.shape[3]
        for a in range(k):
            for b in range(k):
                dxp[:, :, a : a + s * Ho : s, b : b + s * Wo : s] += np.einsum(
                    "nohw,oc->nchw", grad, W[:, :, a, b], optimize=True
                )
        return dxp[:, :, p:-p, p:-p] if p else dxp


class MaxPool2D(Layer):
    """Non-overlapping max pooling over size×size blocks (Section 9.4)."""

    def __init__(self, size: int = 2) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        s = self.size
        if h % s or w % s:
            raise ValueError(f"input {h}x{w} not divisible by pool size {s}")
        blocks = x.reshape(n, c, h // s, s, w // s, s)
        out = blocks.max(axis=(3, 5))
        self._mask = blocks == out[:, :, :, None, :, None]
        # Route the gradient to a single winner per block, even with ties.
        self._mask &= np.cumsum(np.cumsum(self._mask, axis=3), axis=5) == 1
        self._shape = x.shape
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return (self._mask * grad[:, :, :, None, :, None]).reshape(self._shape)


class AvgPool2D(Layer):
    """Non-overlapping average pooling (the sub-sampling used in LeNet1)."""

    def __init__(self, size: int = 2) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        s = self.size
        if h % s or w % s:
            raise ValueError(f"input {h}x{w} not divisible by pool size {s}")
        self._shape = x.shape
        return x.reshape(n, c, h // s, s, w // s, s).mean(axis=(3, 5))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        s = self.size
        g = np.repeat(np.repeat(grad, s, axis=2), s, axis=3)
        return g / (s * s)


class Sequential(Layer):
    """A stack of layers applied in order; backward runs them in reverse."""

    def __init__(self, *layers: Layer) -> None:
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray, Layer, str]]:
        """(param, grad, owning layer, name) for every trainable array."""
        out = []
        for layer in self.layers:
            if isinstance(layer, Sequential):
                out.extend(layer.parameters())
            else:
                for name, p in layer.params.items():
                    out.append((p, layer.grads[name], layer, name))
        return out


class SGD:
    """Gradient descent with optional momentum (Section 4.6)."""

    def __init__(self, model: Sequential, lr: float = 0.1, momentum: float = 0.0) -> None:
        self.model, self.lr, self.momentum = model, lr, momentum
        self._v = [np.zeros_like(p) for p, *_ in model.parameters()]

    def step(self) -> None:
        for v, (p, _, layer, name) in zip(self._v, self.model.parameters(), strict=True):
            g = layer.grads[name]
            v *= self.momentum
            v -= self.lr * g
            p += v


class Adam:
    """Adam optimiser (Kingma and Ba, 2014), a common default for deep nets."""

    def __init__(
        self,
        model: Sequential,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.model, self.lr, self.betas, self.eps = model, lr, betas, eps
        params = model.parameters()
        self._m = [np.zeros_like(p) for p, *_ in params]
        self._v = [np.zeros_like(p) for p, *_ in params]
        self._t = 0

    def step(self) -> None:
        self._t += 1
        b1, b2 = self.betas
        params = self.model.parameters()
        for m, v, (p, _, layer, name) in zip(self._m, self._v, params, strict=True):
            g = layer.grads[name]
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat = m / (1 - b1**self._t)
            v_hat = v / (1 - b2**self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
