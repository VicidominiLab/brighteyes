"""Hann window calculations for imaging arrays."""

import numpy as np
from numpy.typing import NDArray


def hann_window_2d(shape: tuple[int, int]) -> NDArray[np.float64]:
    """Return a two-dimensional Hann window."""
    height, width = shape
    x = np.arange(height)
    y = np.arange(width)
    x_grid, y_grid = np.meshgrid(x, y)
    window = 0.5 * (1 - np.cos((2 * np.pi * x_grid) / (height - 1)))
    window *= 0.5 * (1 - np.cos((2 * np.pi * y_grid) / (width - 1)))
    return window.T
