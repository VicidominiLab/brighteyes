# Origin: VicidominiLab/birfi, src/birfi/utils.py at 4ea6c4d.
"""Tensor utilities for Bayesian IRF estimation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from brighteyes.mcs.dataprep import Alignment

from ..convolution import partial_convolution

PaddingMode = Literal["reflect", "replicate", "constant"]
FitParameters = Mapping[str, float | torch.Tensor]

__all__ = [
    "estimate_lifetime",
    "generate_truncated_exponential",
    "median_filter",
    "pad_tensor",
    "partial_convolution",
    "plot_dataset",
]


def pad_tensor(
    x: torch.Tensor,
    pad_left: int,
    pad_right: int,
    dim: int,
    mode: PaddingMode = "reflect",
) -> torch.Tensor:
    """Pad a tensor along one dimension."""
    return cast("torch.Tensor", Alignment.pad_tensor(x, pad_left, pad_right, dim, mode))


def median_filter(
    x: torch.Tensor,
    window_size: int | Sequence[int] = 3,
    dims: Sequence[int] | None = None,
    mode: PaddingMode = "reflect",
) -> torch.Tensor:
    """Apply a median filter along selected dimensions."""
    return cast("torch.Tensor", Alignment.median_filter(x, window_size, dims, mode))


def generate_truncated_exponential(
    t: torch.Tensor | np.ndarray[Any, Any] | Sequence[float], params: FitParameters
) -> torch.Tensor:
    """Generate a truncated exponential curve."""
    A, k, C, t0 = params["A"], params["k"], params["C"], params["t0"]

    t = torch.as_tensor(t)

    y = torch.where(t >= t0, A * torch.exp(-(t - t0) * k) + C, torch.ones_like(t) * C)

    return y


def plot_dataset(
    x: np.ndarray[Any, Any] | torch.Tensor,
    y: np.ndarray[Any, Any] | torch.Tensor,
    color: str = "C0",
    linestyle: str = "solid",
    marker: str | None = None,
    sharex: bool = True,
    sharey: bool = True,
    figsize: tuple[float, float] | None = None,
    xlabel: str = "Time (ns)",
    ylabel: str = "Intensity",
    fig: Figure | None = None,
    ax: Axes | NDArray[Any] | None = None,
) -> tuple[Figure, NDArray[Any]]:
    """Plot each channel in a separate subplot."""
    if np.ndim(y) != 2:
        raise ValueError("y must be a 2D array")

    _, C = y.shape
    nrows = int(np.ceil(np.sqrt(C)))
    ncols = int(np.ceil(C / nrows))

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    if fig is None or ax is None:
        fig, ax = plt.subplots(
            nrows, ncols, sharex=sharex, sharey=sharey, figsize=figsize
        )

    ax = np.array(ax).reshape(-1)

    for c in range(C):
        ax[c].plot(x, y[:, c], color=color, linestyle=linestyle, marker=marker)
        ax[c].set_title(f"Channel {c}")
        if c % ncols == 0:
            ax[c].set_ylabel(ylabel)
        if c // ncols == nrows - 1:
            ax[c].set_xlabel(xlabel)

    for c in range(C, len(ax)):
        ax[c].axis("off")

    fig.tight_layout()

    return fig, ax


def estimate_lifetime(
    x: torch.Tensor, y: torch.Tensor, t0: int | torch.Tensor, t1: int | torch.Tensor
) -> torch.Tensor:
    """Estimate lifetime from the signal centroid."""
    x = torch.as_tensor(x[t0 : t1 + 1], dtype=torch.float32)
    x = x - x.min()  # shift to start at 0

    # Baseline subtraction and clamping
    y = torch.as_tensor(y[t0 : t1 + 1], dtype=torch.float32)
    y_clamped = torch.clamp(y - y.min(), min=0.0)  # subtract baseline and clamp to zero

    # Centroid-based lifetime estimation
    tau = torch.sum(x * y_clamped) / torch.sum(y_clamped)

    return tau
