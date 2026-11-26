"""Estimate detector magnification from measured channel shifts."""

from __future__ import annotations

# Origin: VicidominiLab/s2ISM, src/s2ism/mag_finder.py at 10bb5fc.
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.signal import convolve
from scipy.special import jv

FloatArray = NDArray[np.floating[Any]]


def scalar_psf(r: FloatArray, wl: float, na: float) -> FloatArray:
    """Calculate a scalar point-spread function for a circular aperture."""
    k = 2 * np.pi / wl
    x = k * r * na

    psf = np.ones_like(x) * 0.5
    np.divide(jv(1, x), x, out=psf, where=(x != 0))
    psf = np.abs(psf) ** 2

    return psf


def scalar_psf_det(
    r: FloatArray,
    wl: float,
    na: float,
    pxdim: float,
    pxpitch: float,
    m: float | FloatArray,
) -> FloatArray:
    """Calculate a scalar detection point-spread function."""
    psf = scalar_psf(r, wl, na)
    pinhole = rect(r - pxpitch / m, pxdim / m)
    psf_det = convolve(psf, pinhole, mode="same")

    return cast(FloatArray, psf_det)


def rect(r: FloatArray, d: float | FloatArray) -> FloatArray:
    """Evaluate a normalized rectangular function."""
    r = np.where(abs(r) <= d / 2, 1, 0)
    return r / d


def shift_value(
    m: float | FloatArray,
    wl_ex: float,
    wl_em: float,
    pxpitch: float,
    pxdim: float,
    na: float,
) -> np.float64:
    """Calculate the theoretical channel shift for a magnification."""
    airy_unit = 1.22 * wl_em / na  # nm
    pxsize = 0.1  # nm
    range_x = int(airy_unit / pxsize)
    ref = np.arange(-range_x, range_x + 1) * pxsize

    psf_det = scalar_psf_det(ref, wl_em, na, pxdim, pxpitch, m)
    psf_ex = scalar_psf(ref, wl_ex, na)
    psf_conf = psf_det * psf_ex

    shift = cast(np.float64, ref[np.argmax(psf_conf)])

    return shift


def loss_shift(
    x: float | FloatArray,
    shift_exp: float,
    wl_ex: float,
    wl_em: float,
    pxpitch: float,
    pxdim: float,
    na: float,
) -> np.float64:
    """Return squared error between theoretical and measured channel shifts."""
    shift_t = shift_value(x, wl_ex, wl_em, pxpitch, pxdim, na)
    loss_func = np.linalg.norm(shift_t - shift_exp) ** 2
    return cast(np.float64, loss_func)


def loss_minimizer(
    shift_t: float,
    wl_ex: float,
    wl_em: float,
    pxpitch: float,
    pxdim: float,
    na: float,
    m_0: float,
    tol: float,
    opt: Mapping[str, int],
) -> np.float64:
    """Estimate magnification by minimizing channel-shift error."""
    result = minimize(
        loss_shift,
        x0=m_0,
        args=(shift_t, wl_ex, wl_em, pxpitch, pxdim, na),
        options=opt,
        tol=tol,
        method="Nelder-Mead",
    )

    if not result.success:
        print("Minimization did not succeed.")
        print(result.message)

    return cast(np.float64, result.x[0])


def find_mag(
    shift: float,
    wl_ex: float,
    wl_em: float,
    pxpitch: float,
    pxdim: float,
    na: float,
) -> np.float64:
    """Estimate magnification from a measured channel shift."""
    opt = {"maxiter": 10000}
    tol = 1e-6
    m_0 = 500
    m = loss_minimizer(shift, wl_ex, wl_em, pxpitch, pxdim, na, m_0, tol, opt)

    return m
