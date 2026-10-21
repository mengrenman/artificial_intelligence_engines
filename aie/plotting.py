"""Plotting helpers shared across notebooks.

These wrap the most repetitive matplotlib patterns. Notebooks remain free
to call matplotlib directly when a one-off plot is clearer inline.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary_2d(
    X: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Scatter 2D points, with the line {x : w · x = 0} as the boundary."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    pos = t > 0
    ax.scatter(X[pos, 0], X[pos, 1], marker="x", color="C0", label="class +1")
    ax.scatter(
        X[~pos, 0],
        X[~pos, 1],
        marker="o",
        edgecolors="C1",
        facecolors="none",
        label="class -1",
    )
    if abs(w[1]) > 1e-9:
        xs = np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 50)
        ys = -w[0] / w[1] * xs
        ax.plot(xs, ys, "k--", label="decision boundary")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    return ax


def plot_error_surface(
    error_fn: Callable[[np.ndarray], float],
    w_range: tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 60,
    levels: int = 20,
    ax: plt.Axes | None = None,
):
    """Contour plot of an error function E(w_1, w_2)."""
    w1 = np.linspace(*w_range, n_grid)
    w2 = np.linspace(*w_range, n_grid)
    W1, W2 = np.meshgrid(w1, w2)
    E = np.empty_like(W1)
    for i in range(n_grid):
        for j in range(n_grid):
            E[i, j] = error_fn(np.array([W1[i, j], W2[i, j]]))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contour(W1, W2, E, levels=levels, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    return ax, (W1, W2, E)


def show_image_grid(
    images: list[np.ndarray],
    titles: list[str] | None = None,
    cmap: str = "gray",
    cols: int = 3,
    figsize: tuple[float, float] | None = None,
):
    """Lay out a row (or grid) of images for the Ch 2 photograph demo."""
    n = len(images)
    rows = (n + cols - 1) // cols
    if figsize is None:
        figsize = (2.4 * cols, 2.4 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(images[i], cmap=cmap, vmin=0.0, vmax=1.0)
            if titles is not None:
                ax.set_title(titles[i], fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    return fig
