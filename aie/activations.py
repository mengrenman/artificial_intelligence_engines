"""Activation functions and their derivatives.

Each derivative function takes the *output* y of the activation (rather than
the pre-activation u) so it can be reused on the cached forward-pass values
during backprop. For the sigmoid this means dy/du = y(1 - y) (Eq 4.2).
"""

from __future__ import annotations

import numpy as np


def sigmoid(u: np.ndarray) -> np.ndarray:
    """Logistic sigmoid (Eq 4.1)."""
    return 1.0 / (1.0 + np.exp(-u))


def sigmoid_deriv(y: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid expressed in terms of its output (Eq 4.2)."""
    return y * (1.0 - y)


def identity(u: np.ndarray) -> np.ndarray:
    """Identity activation, used for the input and output layers in Ch 4."""
    return u


def identity_deriv(y: np.ndarray) -> np.ndarray:
    return np.ones_like(y)


def step(u: np.ndarray) -> np.ndarray:
    """Threshold activation used in perceptrons (Fig 3.3) and Hopfield nets."""
    return np.where(u > 0, 1.0, -1.0)


def tanh(u: np.ndarray) -> np.ndarray:
    return np.tanh(u)


def tanh_deriv(y: np.ndarray) -> np.ndarray:
    return 1.0 - y * y


def relu(u: np.ndarray) -> np.ndarray:
    """Rectified linear unit (Section 9.5, Fig 9.8)."""
    return np.maximum(0.0, u)


def relu_deriv_from_input(u: np.ndarray) -> np.ndarray:
    """ReLU has a kink: derivative is taken with respect to the pre-activation."""
    return (u > 0).astype(float)


def softmax(u: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax activation (Eq 7.13)."""
    u_shifted = u - np.max(u, axis=axis, keepdims=True)
    e = np.exp(u_shifted)
    return e / np.sum(e, axis=axis, keepdims=True)
