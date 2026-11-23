# Origin: VicidominiLab/s2ISM, src/s2ism/misalignment.py at 10bb5fc.
"""Estimate and correct detector misalignment in point-spread functions."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from brighteyes.imaging._types import Array
from brighteyes.imaging.analysis.APR_lib import ShiftVectors

from .shift_vectors_minimizer import find_parameters, rotation_matrix

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def gaussian_2d(
    params: FloatArray, x: FloatArray | IntArray, y: FloatArray | IntArray
) -> FloatArray:
    """Evaluate a two-dimensional Gaussian."""
    a, x0, y0, sigma, b = params
    return cast(
        np.ndarray,
        a * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)) + b,
    )


def residuals(
    params: FloatArray,
    x: FloatArray | IntArray,
    y: FloatArray | IntArray,
    data: FloatArray,
) -> FloatArray:
    """Return residuals between a Gaussian model and measured data."""
    model = gaussian_2d(params, x, y)
    return (model - data).ravel()


def gaussian_fit(
    image: FloatArray,
) -> tuple[np.float64, np.float64, FloatArray, FloatArray] | None:
    """Fit a two-dimensional Gaussian to an image."""
    image = image / image.max()  # Normalize

    h, w = image.shape
    y, x = np.mgrid[0:h, 0:w]

    # Initial guess
    a0 = image.max() - image.min()
    x0_0 = (w - 1) / 2
    y0_0 = (h - 1) / 2
    sigma0 = min(h, w) / 4
    b0 = image.min()
    p0 = [a0, x0_0, y0_0, sigma0, b0]

    # Parameter bounds
    bounds = (
        [0, 0, 0, 0.5, 0],  # lower bounds
        [image.max(), w, h, w, image.min()],  # upper bounds
    )

    result = least_squares(
        residuals,
        p0,
        args=(x, y, image),
        bounds=bounds,
        method="trf",
        loss="cauchy",
        f_scale=0.1,  # sensitivity to outliers
    )

    gfit = gaussian_2d(result.x, x, y)

    if result.success:
        _, x0, y0, _, _ = result.x
        return x0 - x0_0, y0 - y0_0, result.x, gfit
    else:
        return None  # fitting failed


def find_misalignment(
    dset: Array, pxpitch: float, mag: float, na: float, wl: float
) -> tuple[np.float64, np.float64]:
    """Estimate detector tip and tilt from a PSF dataset."""
    nch = int(np.sqrt(dset.shape[-1]))

    axis_to_sum = tuple(
        np.arange(dset.ndim - 1)
    )  # It automatically takes into account time, if present

    fingerprint: FloatArray = dset.sum(axis_to_sum).reshape(nch, nch)

    gauss = gaussian_fit(fingerprint)

    if gauss is None:
        print("\n Warning: Fitting not successful. Using no tip and no tilt.\n")
        tip, tilt = np.zeros(2)
    else:
        scale = 2 * np.pi * na / wl
        pxsize = pxpitch / mag

        coords = -np.asarray([gauss[1], gauss[0]])

        shift, _ = ShiftVectors(
            dset[5:-5, 5:-5], 50, dset.shape[-1] // 2, filter_sigma=1
        )
        _, rotation, mirroring = find_parameters(shift)

        if mirroring == -1:
            coords[1] *= -1

        rm = rotation_matrix(rotation)

        rot_coords = np.einsum("ij, j -> i", rm, coords)

        tip, tilt = rot_coords * pxsize * scale

    return tip, tilt


def realign_psf(psf: FloatArray) -> FloatArray:
    """Center the brightest pixel in a PSF stack."""
    nz, ny, nx, _ = psf.shape
    patch = psf[0].sum(-1)

    yc, xc = ny // 2, nx // 2  # integer center

    # Find coordinates of the brightest pixel
    peak_index = np.argmax(patch)
    y_peak, x_peak = np.unravel_index(peak_index, patch.shape)

    # Compute integer shift needed
    y_shift = yc - y_peak
    x_shift = xc - x_peak

    # Apply integer pixel shift using

    aligned_psf = np.empty_like(psf)

    for z in range(nz):
        aligned_psf[z] = np.roll(psf[z], shift=y_shift, axis=0)
        aligned_psf[z] = np.roll(aligned_psf[z], shift=x_shift, axis=1)

    return aligned_psf
