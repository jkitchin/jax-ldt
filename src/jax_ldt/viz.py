"""Lightweight plotting helpers.

These helpers lazy-import :mod:`matplotlib.pyplot` so that the core
package has no hard dependency on matplotlib. Each function raises an
informative :class:`ImportError` if matplotlib cannot be imported.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _require_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by guard tests
        raise ImportError(
            "jax_ldt.viz requires matplotlib. Install it with "
            "`pip install matplotlib`."
        ) from exc
    return plt


def plot_tree_partition_2d(
    model,
    X,
    ax=None,
    cmap: str = "tab20",
):
    """Scatter the points coloured by the leaf id assigned by ``model``.

    Only meaningful for a 2-feature regressor. The resulting figure
    shows how the tree partitions the input space.

    Parameters
    ----------
    model : fitted regressor with ``.apply(X)`` returning leaf ids.
    X : array-like of shape (n_samples, 2)
    ax : optional matplotlib Axes
    cmap : matplotlib colormap name (default "tab20")

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    plt = _require_pyplot()
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(
            f"plot_tree_partition_2d expects (n, 2) inputs, got shape {X.shape}"
        )
    leaf_ids = np.asarray(model.apply(X)).reshape(-1)
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=leaf_ids, cmap=cmap, s=18, edgecolor="none")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(f"Tree partition: {len(np.unique(leaf_ids))} leaves")
    return ax


def plot_calibration(
    y_true,
    y_pred,
    sigma=None,
    intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ax=None,
):
    """Parity plot (y_true vs y_pred) with optional uncertainty overlays.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n,)
    sigma : optional array-like of shape (n,)
        Per-sample standard deviation; drawn as symmetric error bars on
        ``y_pred``.
    intervals : optional tuple of two arrays (lo, hi)
        Lower / upper prediction bounds; drawn as a shaded band sorted
        by ``y_pred``.
    ax : optional matplotlib Axes

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    plt = _require_pyplot()
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}"
        )

    if ax is None:
        _, ax = plt.subplots()

    if intervals is not None:
        lo, hi = intervals
        lo = np.asarray(lo).reshape(-1)
        hi = np.asarray(hi).reshape(-1)
        order = np.argsort(y_pred)
        ax.fill_between(
            y_pred[order],
            lo[order],
            hi[order],
            alpha=0.2,
            color="C0",
            label="prediction band",
        )

    if sigma is not None:
        sigma = np.asarray(sigma).reshape(-1)
        ax.errorbar(
            y_true,
            y_pred,
            yerr=sigma,
            fmt="o",
            ms=3,
            alpha=0.6,
            color="C1",
            ecolor="C1",
            elinewidth=0.5,
            label="±σ",
        )
    else:
        ax.scatter(y_true, y_pred, s=12, alpha=0.7, color="C1")

    lo_lim = float(min(y_true.min(), y_pred.min()))
    hi_lim = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], "k--", lw=1, label="y = ŷ")
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.set_title("Calibration (parity) plot")
    if sigma is not None or intervals is not None:
        ax.legend(loc="best", fontsize="small")
    return ax
