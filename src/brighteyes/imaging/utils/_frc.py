"""Fourier ring correlation and radial spectrum helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as ft

# Origin: VicidominiLab/BrightEyes-ISM, src/brighteyes_ism/analysis/FRC_lib.py at d799720.
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess

from ..analysis._windows import hann_window_2d

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from numpy.typing import NDArray

    from .._types import Array

    FloatArray = NDArray[np.float64]


def smooth(x: Array, y: Array) -> tuple[Array, Array]:
    """Smooth a curve with LOWESS interpolation."""
    x_interp = np.linspace(x[0], x[-1], num=100 * len(x))
    y_interp = np.interp(x_interp, x, y)

    filtered = lowess(y_interp, x_interp, is_sorted=True, frac=0.05, it=0)
    return x_interp, filtered[:, 1]


def hann2d(*args: int) -> NDArray[np.float64]:
    """Return a two-dimensional Hann window for a square or rectangular image."""
    if len(args) > 1:
        width, height = args
    else:
        (width,) = args
        height = width

    return hann_window_2d((height, width))


def radial_profile(data: Array, center: Sequence[float]) -> tuple[Array, Array]:
    """Calculate the radial profile of a two-dimensional array."""
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), np.real(data).ravel()).astype(np.complex128)
    tbin += 1j * np.bincount(r.ravel(), np.imag(data).ravel())

    nr = np.bincount(r.ravel())
    radialprofile = tbin

    return radialprofile, nr


def FRC(im1: Array, im2: Array) -> Array:
    """Calculate Fourier ring correlation for two images."""
    M, N = np.shape(im1)
    cx = int((N + np.mod(N, 2)) / 2)
    cy = int((M + np.mod(M, 2)) / 2)
    center = [cx, cy]

    im1 = im1 * hann2d(N, M)
    im2 = im2 * hann2d(N, M)

    ft1 = ft.fftshift(ft.fft2(im1))
    ft2 = ft.fftshift(ft.fft2(im2))

    num = np.real(radial_profile(ft1 * np.conj(ft2), center)[0])
    den = np.real(radial_profile(np.abs(ft1) ** 2, center)[0])
    den = den * np.real(radial_profile(np.abs(ft2) ** 2, center)[0])
    den = np.sqrt(den)

    FRC = num / den

    return np.asarray(FRC)


def fixed_threshold(frc: Array, y: float) -> tuple[Array, int | np.integer[Any]]:
    """Return a fixed threshold and its first crossing index."""
    N = len(frc)
    th = np.ones(N) * y
    try:
        idx = np.argwhere(np.diff(np.sign(frc - y))).flatten()[0]
    except IndexError:
        idx = 0
    return th, idx


def nsigma_threshold(
    k: Array, frc: Array, img: Array, sigma: int
) -> tuple[Array, int | np.integer[Any]]:
    """Estimate an FRC threshold from image noise."""
    N, M = np.shape(img)
    cx = int((N + np.mod(N, 2)) / 2)
    cy = int((M + np.mod(M, 2)) / 2)
    center = [cx, cy]

    nr = radial_profile(img, center)[1]

    th = sigma / np.sqrt(nr / 2)

    _, th_interp = smooth(k, th)

    try:
        idx1 = np.argwhere(np.diff(np.sign(frc - th_interp))).flatten()
        idx2 = idx1[1]
    except IndexError:
        idx2 = 0
    return th_interp, idx2


def FRC_resolution(
    I1: Array,
    I2: Array,
    px: float = 1,
    method: str = "fixed",
    smoothing: str = "lowess",
) -> tuple[float, Array, Array, Array, Array, Array]:
    """Estimate image resolution from two independent images."""
    frc = FRC(I1, I2)
    F = len(frc)

    max_kpx = 1 / np.sqrt(2)

    kpx = np.linspace(0, max_kpx, F, endpoint=True)
    k = kpx / px

    if smoothing == "lowess":
        k_interp, frc_smooth = smooth(k, frc)  # FRC smoothing

    elif smoothing == "fit":
        p0 = (1, 1, 10, 0)
        bounds = ((0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf))

        sigmoid_fit: Callable[[Array, float, float, float, float], Array] = (
            lambda x, a, b, c, d: a / (1 + np.exp((x - b) / c)) + d
        )

        popt, _ = curve_fit(
            sigmoid_fit, k[kpx < 0.5], frc[kpx < 0.5], p0, bounds=bounds
        )

        amplitude = popt[0]
        offset = popt[-1]

        k_interp = np.linspace(0, max_kpx, F * 100, endpoint=True) / px
        frc_smooth = (sigmoid_fit(k_interp, *popt) - offset) / amplitude
        frc = (frc - offset) / amplitude

    else:
        raise ValueError('The smoothing parameter has to be "fit" or "lowess".')

    if method == "fixed":
        th, idx = fixed_threshold(frc_smooth, 1 / 7)
    elif method == "3sigma":
        th, idx = nsigma_threshold(k, frc_smooth, I1, 3)
    elif method == "5sigma":
        th, idx = nsigma_threshold(k, frc_smooth, I1, 5)

    res_um = 1 / k_interp[idx]

    return res_um, k, frc, k_interp, frc_smooth, th


def timeFRC(
    dset: Array,
    px: float = 1,
    method: str = "fixed",
    smoothing: str = "lowess",
) -> tuple[float, Array, Array, Array, Array, Array]:
    """Estimate resolution from even and odd temporal samples."""
    if dset.shape[-1] % 2 == 0:
        img_even = dset[:, :, 0::2].sum(axis=-1)
        img_odd = dset[:, :, 1::2].sum(axis=-1)
    else:
        img_even = dset[:, :, 0:-1:2].sum(axis=-1)
        img_odd = dset[:, :, 1::2].sum(axis=-1)

    res_um, k, frc, k_interp, frc_smooth, th = FRC_resolution(
        img_even, img_odd, px=px, method=method, smoothing=smoothing
    )

    return res_um, k, frc, k_interp, frc_smooth, th


def plotFRC(
    res_um: float,
    k: Array,
    frc: Array,
    k_interp: Array,
    frc_smooth: Array,
    th: Array,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot FRC curves with their threshold."""
    if fig == None or ax == None:
        fig, ax = plt.subplots()

    ax.plot(k, frc, ".", label="FRC - raw")
    ax.plot(k_interp, frc_smooth, "-", linewidth=3, label="FRC - smoothed")
    ax.plot(k_interp, th, linewidth=3, label="Threshold")

    ax.legend()

    idx = (np.abs(k_interp - 1 / res_um)).argmin()

    ax.plot(k_interp[idx], frc_smooth[idx], "o", markersize=6)

    k_max = k[-1] * np.sqrt(2) / 2
    ax.set_xlim((0, k_max))
    ax.set_ylim((-0.05, 1.05))

    ax.set_xlabel(r"k ($\mathregular{\mu m ^{-1}}$)")
    ax.set_ylabel("FRC")

    ax.set_title(f"Resolution = {res_um:.3f}" + r" $\mathregular{\mu m}$")

    return fig, ax
