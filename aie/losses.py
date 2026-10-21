"""Loss / error functions used across chapters.

Convention
----------
The book defines the error for one association as E_t = ½ Σ_k (y_kt − ŷ_kt)²
(Eq 2.5 / Eq 4.8) and the total error as the sum over associations. Summing
over the whole training set makes the sensible learning rate depend on T,
so the models in this repo report the *per-association average*

    E = (1/T) Σ_t ½ Σ_k (y_kt − ŷ_kt)²        (``mean_sse``)

whose gradient with respect to y_kt is simply (y_kt − ŷ_kt) / T. Every
model's ``fit`` uses exactly this scaling, so its reported error and its
weight updates always agree (checked by numerical gradient tests).
"""

from __future__ import annotations

import numpy as np


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a


def mean_sse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Halved sum of squared errors per association, averaged over the batch."""
    y_pred, y_true = _as_2d(y_pred), _as_2d(y_true)
    return 0.5 * float(np.sum((y_pred - y_true) ** 2)) / y_pred.shape[0]


def mean_sse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """dE/dy for :func:`mean_sse`."""
    y_pred, y_true = _as_2d(y_pred), _as_2d(y_true)
    return (y_pred - y_true) / y_pred.shape[0]


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Halved mean squared error over *all* elements (batch and outputs)."""
    return 0.5 * float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))


def sse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Halved sum of squared errors over the whole batch — Eq 4.8 exactly."""
    return 0.5 * float(np.sum((np.asarray(y_pred) - np.asarray(y_true)) ** 2))


def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    """Binary cross-entropy averaged over all elements (Eq 4.45)."""
    y = np.clip(y_pred, eps, 1.0 - eps)
    return -float(np.mean(y_true * np.log(y) + (1.0 - y_true) * np.log(1.0 - y)))


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean multi-class cross-entropy of softmax(logits) and its gradient.

    Parameters
    ----------
    logits : (T, K) pre-softmax outputs.
    labels : (T,) integer class labels in [0, K).

    Returns
    -------
    loss : float
    grad : (T, K) array, dL/dlogits = (softmax(logits) − onehot) / T
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    n = logits.shape[0]
    idx = np.arange(n)
    loss = -float(np.mean(log_p[idx, labels]))
    grad = np.exp(log_p)
    grad[idx, labels] -= 1.0
    return loss, grad / n
