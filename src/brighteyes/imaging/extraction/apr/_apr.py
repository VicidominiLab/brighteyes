# Origin: VicidominiLab/BrightEyes-ISM, src/brighteyes_ism/analysis/APR_lib.py at d799720.
"""Adaptive pixel reassignment and detector shift estimation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.ndimage import fourier_shift, shift
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation

from ..._windows import hann_window_2d

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..._types import Array


def hann2d(shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Return a two-dimensional Hann window for the first two dimensions."""
    return hann_window_2d((shape[0], shape[1]))


def APR(
    dset: Array,
    usf: int,
    ref: int,
    pxsize: float = 1,
    apodize: bool = True,
    filter_sigma: float = 1,
    mode: str = "interp",
) -> tuple[Array, Array]:
    """Reassign detector images using phase-correlation shifts."""
    # Calculate shifts

    shift_vec, _ = cast(Callable[..., tuple[np.ndarray, np.ndarray]], ShiftVectors)(
        dset, usf, ref, apodize=apodize, filter_sigma=filter_sigma
    )

    # Register images

    result_ism_pc = Reassignment(shift_vec, dset, mode=mode)

    shift_vec *= pxsize

    return shift_vec, result_ism_pc


def ShiftVectors(
    dset: Array,
    usf: int,
    ref: int,
    apodize: bool = True,
    filter_sigma: float = 0,
) -> tuple[Array, Array]:
    """Estimate subpixel shifts for each detector channel."""
    sz = dset.shape
    dsetW = dset.copy()

    # Apodize dataset

    if apodize == True:
        W = hann2d(dset.shape)
        dsetW = np.einsum("ijk, ij -> ijk", dsetW, W)

    # Low-pass filter dataset

    if filter_sigma > 0:
        dsetW = cast(Callable[..., np.ndarray], gaussian)(
            dsetW, sigma=filter_sigma, channel_axis=-1
        )

    # Calculate shift-vectors

    shift_vec = np.empty((sz[-1], 2))
    error = np.empty((sz[-1], 2))

    for i in range(sz[-1]):
        shift_vec[i, :], error[i, :], _ = cast(
            Callable[..., tuple[np.ndarray, np.ndarray, float]],
            phase_cross_correlation,
        )(dsetW[:, :, ref], dsetW[:, :, i], upsample_factor=usf, normalization=None)

    return shift_vec, error


def Reassignment(shift_vec: Array, dset: Array, mode: str = "interp") -> Array:
    """Shift each detector channel and clip negative values."""
    sz = dset.shape
    result_ism_pc = np.empty(sz)

    for i in range(sz[-1]):
        if mode == "fourier":
            offset = fourier_shift(np.fft.fftn(dset[:, :, i]), (shift_vec[i, :]))
            result_ism_pc[:, :, i] = np.real(np.fft.ifftn(offset))
        elif mode == "interp":
            result_ism_pc[:, :, i] = shift(dset[:, :, i], shift_vec[i, :])

    result_ism_pc[result_ism_pc < 0] = 0
    return result_ism_pc
