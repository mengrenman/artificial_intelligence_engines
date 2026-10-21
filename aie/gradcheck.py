"""Numerical gradient checking.

Backprop bugs are silent: a network with a wrong gradient often still
learns, just badly. Comparing analytic gradients against central finite
differences,

    dE/dw ≈ (E(w + h) − E(w − h)) / 2h,

is the standard way to catch them, and every model in this repo is tested
this way.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def numerical_gradient(
    f: Callable[[], float], param: np.ndarray, h: float = 1e-6
) -> np.ndarray:
    """Central-difference gradient of the scalar ``f()`` w.r.t. ``param``.

    ``param`` is perturbed *in place* (and restored), so ``f`` should read
    it by reference — e.g. a closure over a model whose weight array is
    ``param``.
    """
    grad = np.zeros_like(param, dtype=float)
    it = np.nditer(param, flags=["multi_index"], op_flags=[["readwrite"]])
    for _ in it:
        idx = it.multi_index
        old = param[idx]
        param[idx] = old + h
        f_plus = f()
        param[idx] = old - h
        f_minus = f()
        param[idx] = old
        grad[idx] = (f_plus - f_minus) / (2.0 * h)
    return grad


def relative_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """max |a − b| / max(|a| + |b|), a scale-free gradient-agreement measure."""
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a) + np.abs(b)), eps))
