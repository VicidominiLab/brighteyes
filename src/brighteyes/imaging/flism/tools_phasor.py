# Origin: VicidominiLab/BrightEyes-FLISM, src/brighteyes_flism/tools_phasor.py at 85f0d96.
"""Phasor utilities, lifetime estimators, and plotting helpers."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, cast

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb
from matplotlib.pyplot import gca
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray
from scipy.ndimage import median_filter, shift
from skimage.registration import phase_cross_correlation
from tqdm.auto import tqdm

from brighteyes.mcs.dataprep import Alignment
from brighteyes.mcs.dataprep import tools_phasor as dataprep_phasor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

Array = NDArray[Any]
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

try:
    gra: Any
    import brighteyes.imaging.plotting as gra
except ImportError:  # pragma: no cover - optional dependency
    gra = None
try:
    torch: Any
    fftn: Any
    import torch
    from torch.fft import fftn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    fftn = None

__all__ = [
    "align_decays",
    "align_image",
    "apply_lifetime_equalizer",
    "build_lifetime_equalizer",
    "calculate_irf_correction",
    "calculate_m_phi_tau_phi_tau_m",
    "calculate_phasor",
    "calculate_phasor_on_img",
    "calculate_phasor_on_img_ch",
    "calculate_phasor_on_img_pixels",
    "calculate_tau_m",
    "calculate_tau_phi",
    "calibrate_phasor",
    "cmap2d",
    "correct_phasor",
    "correction_phasor",
    "equalized_lifetime_tick_values",
    "estimate_irf",
    "estimate_lifetime_from_birfi",
    "estimate_lifetime_from_circmean",
    "estimate_lifetime_from_log",
    "find_shifts_irf_data",
    "find_temporal_shifts",
    "flatten_and_remove_nan",
    "fourier_shift",
    "g0s0_to_m_phi",
    "linear_shift",
    "m_phi_to_complex",
    "m_phi_to_g0s0",
    "median_phasors",
    "pad_tensor",
    "partial_convolution_fft",
    "periodic_single_exponential_model",
    "phasor",
    "phasor_h5",
    "plot_flim_image",
    "plot_funny_single_phasor",
    "plot_phasor",
    "plot_tau",
    "plot_universal_circle",
    "richardson_lucy_deconvolution",
    "show_flim_equalized",
    "show_lifetime_histogram",
    "sum_adjacent_pixel",
    "threshold_phasor",
    "torch_median_filter",
]


def correct_phasor(
    phasor_data: ComplexArray | complex | None,
    laser_phasor: complex,
    coeff: complex = 1.0,
    laser_correction: bool = True,
) -> ComplexArray | complex:
    """Apply laser and scale corrections to phasor values."""
    if phasor_data is None:
        raise (ValueError("call before calculate_phasor_on_img_ch(...)"))

    if laser_correction:
        return cast(
            ComplexArray | complex,
            phasor_data * coeff * np.exp(1j * (-np.angle(laser_phasor))),
        )
    else:
        return phasor_data * coeff


def sum_adjacent_pixel(data: Array, n: int = 4) -> Array | None:
    """Sum square pixel blocks while preserving the histogram axes."""
    if n <= 1:
        return data
    if len(data.shape) == 2:
        data = data[:, 0 : ((data.shape[0] // n) * n)]
        data = data[0 : ((data.shape[1] // n) * n), :]
        assert data.shape[0] % n == 0
        assert data.shape[1] % n == 0
        d = data[0::n, ::n]
        for i in range(1, n):
            d = d + data[i::n, i::n]
        return d
    elif len(data.shape) == 6:
        data = data[:, :, :, 0 : ((data.shape[2] // n) * n), :, :]
        data = data[:, :, 0 : ((data.shape[3] // n) * n), :, :, :]
        assert data.shape[2] % n == 0
        assert data.shape[3] % n == 0
        d = data[:, :, 0::n, 0::n, :, :]
        for i in range(1, n):
            d = d + data[:, :, i::n, i::n, :, :]
        return d
    return None


def _require_torch() -> None:
    """Raise when a required MCS tensor operation is unavailable."""
    Alignment._require_torch()


def periodic_single_exponential_model(
    t: ArrayLike, C: float, tau: float, period: float
) -> FloatArray:
    """Evaluate a periodic single-exponential decay."""
    t = np.asarray(t, dtype=float)
    offset = (t.max() - t.min()) / 2
    return cast(
        FloatArray,
        C
        * (np.heaviside(t - offset, 1) + 1 / (np.exp(period / tau) - 1))
        * np.exp((-t + offset) / tau),
    )


def pad_tensor(
    x: Any, pad_left: int, pad_right: int, dim: int, mode: str = "reflect"
) -> Any:
    """Pad a tensor along one dimension."""
    return Alignment.pad_tensor(x, pad_left, pad_right, dim, mode=mode)


def torch_median_filter(
    x: Any,
    window_size: int | Sequence[int] = 3,
    dims: Sequence[int] | None = None,
    mode: str = "reflect",
) -> Any:
    """Apply a median filter along selected tensor dimensions."""
    return Alignment.median_filter(x, window_size=window_size, dims=dims, mode=mode)


def partial_convolution_fft(
    volume: Any,
    kernel: Any,
    dim1: str = "ijk",
    dim2: str = "jkl",
    axis: str = "jk",
    fourier: tuple[bool, bool] = (False, False),
) -> Any:
    """Convolve tensors over named axes using FFTs."""
    return Alignment.partial_convolution_fft(
        volume,
        kernel,
        dim1=dim1,
        dim2=dim2,
        axis=axis,
        fourier=fourier,
    )


def richardson_lucy_deconvolution(
    ref_data: Any,
    t: ArrayLike,
    C_R: float,
    tau_R: float,
    T: float,
    iterations: int = 30,
    eps: float = 1e-8,
    regularization: int = 3,
) -> Any:
    """Estimate the instrument response with Richardson-Lucy deconvolution."""
    _require_torch()

    if not torch.is_tensor(ref_data):
        ref_data = torch.as_tensor(ref_data)

    ref_data = ref_data.to(dtype=torch.float64)
    kernel = torch.as_tensor(
        periodic_single_exponential_model(t, C_R, tau_R, T),
        dtype=ref_data.dtype,
        device=ref_data.device,
    )
    kernel_t = kernel.flip(0)

    irf_est = torch.ones_like(ref_data)
    kernel_fft = fftn(kernel, dim=0)
    kernel_t_fft = fftn(kernel_t, dim=0)
    y = torch.clamp(ref_data, min=0)

    for _ in range(iterations):
        conv = partial_convolution_fft(
            irf_est, kernel_fft, dim1="t", dim2="t", axis="t", fourier=(False, True)
        )
        conv = torch.clamp(conv, min=eps)
        relative_blur = y / conv
        correction = partial_convolution_fft(
            relative_blur,
            kernel_t_fft,
            dim1="t",
            dim2="t",
            axis="t",
            fourier=(False, True),
        )
        irf_est = torch.clamp(irf_est * correction, min=0)
        if regularization > 1:
            irf_est = torch_median_filter(
                irf_est, window_size=regularization, dims=[0], mode="replicate"
            )

    return irf_est


def estimate_irf(
    ref_data: Any,
    t: ArrayLike,
    C_R: float,
    tau: float,
    period: float,
    iterations: int = 300,
    eps: float = 1e-8,
    regularization: int = 3,
) -> Any:
    """Estimate an instrument response with the default fit settings."""
    return richardson_lucy_deconvolution(
        ref_data,
        t,
        C_R,
        tau,
        period,
        iterations=iterations,
        eps=eps,
        regularization=regularization,
    )


# ---------------------------------------------------------------------------
# Lifetime estimators
# ---------------------------------------------------------------------------
def estimate_lifetime_from_birfi(
    x: ArrayLike,
    y: ArrayLike,
    window_length: int = 11,
    polyorder: int = 3,
    persistence: int = 5,
    threshold: float = 0.05,
    axis: int = 0,
    return_bounds: bool = False,
) -> Any:
    """Estimate lifetime from the centroid of a detected decay window."""
    return dataprep_phasor.estimate_lifetime_from_birfi(
        x, y, window_length, polyorder, persistence, threshold, axis, return_bounds
    )


def estimate_lifetime_from_log(
    data_hist: ArrayLike,
    t_ns: ArrayLike,
    dt_ns: float,
    nbin: int,
    period_ns: float,
    start_level: float = 0.95,
    end_level: float = 0.25,
) -> tuple[Any, Any, Any, Any, Any]:
    """Estimate lifetime by fitting the logarithm of a decay segment."""
    return cast(
        "tuple[Any, Any, Any, Any, Any]",
        dataprep_phasor.estimate_lifetime_from_log(
            data_hist, t_ns, dt_ns, nbin, period_ns, start_level, end_level
        ),
    )


def estimate_lifetime_from_circmean(
    counts: ArrayLike,
    t0_ns: float,
    repetition_rate_MHz: float = 40.0,
    background_per_bin: float = 0.0,
    truncate_ns: float | None = None,
) -> float:
    """Estimate lifetime from the weighted circular mean of photon counts."""
    return cast(
        "float",
        dataprep_phasor.estimate_lifetime_from_circmean(
            counts, t0_ns, repetition_rate_MHz, background_per_bin, truncate_ns
        ),
    )


# ---------------------------------------------------------------------------
# Phasor analysis and visualization
# ---------------------------------------------------------------------------
def plot_universal_circle(ax: Any = None) -> None:
    """Draw the universal circle and the unit circle on the current axes."""
    if ax is None:
        ax = gca()
    n = np.linspace(0, 2 * np.pi)
    g = 0.5 * (np.cos(n) + 1)
    s = 0.5 * np.sin(n)
    ax.plot(g, s, "--")
    n = np.linspace(0, 2 * np.pi)
    g = np.cos(n)
    s = np.sin(n)
    ax.plot(g, s, "--")
    ax.plot([-1.1, 1.1], [0, 0], "--k")
    ax.plot([0, 0], [-1.1, 1.1], "--k")


def flatten_and_remove_nan(
    a: Array, b: Array | None = None
) -> Array | tuple[Array, Array]:
    """Flatten arrays and drop entries that contain NaNs."""
    a = a.flatten()
    if b is None:
        cond = np.isfinite(a)
        return a[cond]
    if b is not None:
        b = b.flatten()
        cond = np.logical_not(np.logical_or(np.isnan(a), np.isnan(b)))
        return a[cond], b[cond]


def g0s0_to_m_phi(g0: Any, s0: Any) -> tuple[Any, Any]:
    """Convert Cartesian phasor coordinates into modulus and phase."""
    phi = np.arctan2(s0, g0)
    m = np.sqrt(s0**2 + g0**2)
    return m, phi


def m_phi_to_g0s0(m: Any, phi: Any) -> tuple[Any, Any]:
    """Convert modulus and phase into Cartesian phasor coordinates."""
    g0 = np.cos(phi) * m
    s0 = np.sin(phi) * m
    return g0, s0


def phasor(
    data: Array, threshold: float = 0, harmonic: int = 1, time_axis: int = -1
) -> ComplexArray | complex:
    """Calculate a phasor from one or more decay histograms."""
    dset = data.copy()
    flux = dset.sum(time_axis)
    if time_axis != -1:
        dset = np.swapaxes(dset, time_axis, -1)
    transform = np.fft.fft(dset, axis=time_axis)[..., harmonic].conj()
    if transform.ndim == dset.ndim:
        transform = np.swapaxes(transform, -1, time_axis)

    return np.where(flux < threshold, np.nan + 1j * np.nan, transform / flux)


def calculate_phasor(
    data_hist: ArrayLike, threshold: float = 1, harmonic: int = 1
) -> ComplexArray | complex:
    """Calculate the selected phasor harmonic from 1D, 2D, or 3D histograms."""
    data = np.asarray(data_hist, dtype=float).copy()
    if len(data.shape) == 1:
        flux = float(np.sum(data))
        if flux < threshold:
            return np.nan + 1j * np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.fft.fft(data / flux)
        return cast(complex, f[harmonic].conj())
    elif len(data.shape) == 2:
        flux = np.sum(data, axis=0, dtype=float)
        normalized = np.divide(
            data,
            flux,
            out=np.zeros_like(data, dtype=float),
            where=flux > 0,
        )
        f = np.fft.fft(normalized, axis=0)
        out = f[harmonic]
        out[flux < threshold] = np.nan + 1j * np.nan
        return cast(ComplexArray, out.conj())
    elif len(data.shape) == 3:
        flux = data.sum(-1)
        transform = np.fft.fft(data, axis=-1)[..., harmonic].conj()
        out = np.divide(
            transform,
            flux,
            out=np.full(transform.shape, np.nan + 1j * np.nan, dtype=complex),
            where=flux > 0,
        )
        out[flux < threshold] = np.nan + 1j * np.nan
        return cast(ComplexArray, out)
    else:
        raise ValueError("Wrong input dimension")


def calculate_tau_phi(
    g0_or_complex: Array, s0: Array | None = None, dfd_freq: float = 41.48e6
) -> Any:
    """Calculate lifetime from phasor phase."""
    if s0 is None:
        phi = np.angle(g0_or_complex)
    else:
        g0 = g0_or_complex
        phi = np.arctan2(s0, g0)

    tau_phi = np.tan(phi) / (2 * np.pi * dfd_freq)

    return tau_phi


def calculate_tau_m(
    g0_or_complex: Array, s0: Array | None = None, dfd_freq: float = 41.48e6
) -> Any:
    """Calculate lifetime from phasor modulation."""
    if s0 is None:
        m = np.abs(g0_or_complex)
    else:
        g0 = g0_or_complex
        m = np.sqrt(s0**2 + g0**2)

    tau_m = np.sqrt((1.0 / (m**2)) - 1) / (2 * np.pi * dfd_freq)

    return tau_m


def calculate_m_phi_tau_phi_tau_m(
    g0_or_complex: Array, s0: Array | None = None, dfd_freq: float = 41.48e6
) -> tuple[Any, Any, Any, Any]:
    """Calculate phasor magnitude, phase, and both lifetime estimates."""
    if s0 is None:
        phi = np.angle(g0_or_complex)
        m = np.abs(g0_or_complex)
    else:
        g0 = g0_or_complex
        phi = np.arctan2(s0, g0)
        m = np.sqrt(s0**2 + g0**2)

    tau_phi = np.tan(phi) / (2 * np.pi * dfd_freq)
    tau_m = np.sqrt((1.0 / (m**2)) - 1) / (2 * np.pi * dfd_freq)

    return phi, m, tau_phi, tau_m


def calculate_phasor_on_img_ch(
    data_input: Array,
    threshold: float = 1,
    harmonic: int = 1,
    phasor_data_size: int = 100,
) -> None:
    """Write per-channel phasors to an HDF5 file."""
    x_dim, y_dim, channel_dim = (
        data_input.shape[0],
        data_input.shape[1],
        data_input.shape[3],
    )
    with h5py.File("dataset_of_phasor_per_channel", "w") as fi:
        h5_dim = (x_dim, y_dim, channel_dim)
        h5_dataset_p = fi.create_dataset(
            "h5_dataset_p", shape=h5_dim, dtype=np.complex128
        )

        for x_start in range(0, x_dim, phasor_data_size):
            for y_start in range(0, y_dim, phasor_data_size):
                x_stop = min(x_start + phasor_data_size, x_dim)
                y_stop = min(y_start + phasor_data_size, y_dim)

                slice_term_p = np.s_[x_start:x_stop, y_start:y_stop, :, :]
                sub_image_p = data_input[slice_term_p]

                aligned_phasor = np.zeros(
                    sub_image_p.shape[:-2] + (sub_image_p.shape[-1],),
                    dtype=np.complex128,
                )

                for cc in np.arange(sub_image_p.shape[-1]):
                    for yy in np.arange(sub_image_p.shape[-4]):
                        for xx in np.arange(sub_image_p.shape[-3]):
                            aligned_phasor[yy, xx, cc] = phasor(
                                sub_image_p[yy, xx, :, cc], threshold, harmonic
                            )
                h5_dataset_p[x_start:x_stop, y_start:y_stop, :] = aligned_phasor
    fi.close()


def phasor_h5(data_path: str, data_input: Array, harmonic: int = 1) -> Any:
    """Write a pixel phasor matrix to an HDF5 file."""
    data_input_shape = data_input.shape
    if len(data_input_shape) == 4:
        data_input_3d = np.sum(data_input, axis=-1)
    else:
        data_input_3d = data_input.copy()

    directory, filename = os.path.split(data_path)
    filename = os.path.splitext(filename)[0]
    new_filename = filename + "_phasors_matrix.h5"
    new_file_path = os.path.join(directory, new_filename)
    if len(data_input_shape) < 3:
        raise ValueError("data_input must have 3 or more dimensions")
    elif len(data_input_shape) >= 6:
        raise ValueError("use calculate_phasor_on_img_ch() instead")

    with h5py.File(new_file_path, "w") as fil:
        x_dim, y_dim = data_input_3d.shape[0], data_input_3d.shape[1]
        h5_dim = (x_dim, y_dim)
        h5_dataset_phasor_pix = fil.create_dataset(
            "h5_dataset_phasor_pix", shape=h5_dim, dtype=np.complex128
        )

        h5_dataset_phasor_pix[:, :] = calculate_phasor(data_input_3d[:, :, :], harmonic)
        return h5_dataset_phasor_pix

    fil.close()


def calculate_phasor_on_img(
    data_input: Array, threshold: float = 1, harmonic: int = 1
) -> Any:
    """Calculate phasors for each pixel in an image stack."""
    data_input_shape = data_input.shape
    if len(data_input_shape) <= 3:
        raise ValueError("data_input must have 3 or more dimensions")
    elif len(data_input_shape) >= 6:
        raise ValueError("use calculate_phasor_on_img_ch() instead")

    if len(data_input_shape) == 3:
        data_input = data_input[None, None, :, :, :]
    elif len(data_input_shape) == 4:
        data_input = data_input[None, :, :, :, :]

    out = np.zeros(data_input.shape)

    for rr in tqdm(np.arange(data_input.shape[-5])):
        for zz in tqdm(np.arange(data_input.shape[-4])):
            for yy in tqdm(np.arange(data_input.shape[-3])):
                for xx in np.arange(data_input.shape[-2]):
                    out[rr, zz, yy, xx] = phasor(
                        data_input[rr, zz, yy, xx, :], threshold, harmonic
                    )

    if len(data_input_shape) == 3:
        out = out[0, 0, :, :, :]
    elif len(data_input_shape) == 4:
        out = out[0, :, :, :, :]

    return out


def calculate_phasor_on_img_pixels(
    data_path: str,
    data_input: Array,
    threshold: float = 1,
    harmonic: int = 1,
    phasor_pix_data_size: int = 100,
) -> None:
    """Write pixel phasors to an HDF5 file."""
    data_input_3d = np.sum(data_input, axis=-1)
    data_input_shape = data_input_3d.shape

    directory, filename = os.path.split(data_path)
    filename = os.path.splitext(filename)[0]
    new_filename = filename + "_phasors_matrix.h5"
    new_file_path = os.path.join(directory, new_filename)
    if len(data_input_shape) < 3:
        raise ValueError("data_input must have 3 or more dimensions")
    elif len(data_input_shape) >= 6:
        raise ValueError("use calculate_phasor_on_img_ch() instead")

    with h5py.File(new_file_path, "w") as fil:
        x_dim, y_dim = data_input_3d.shape[0], data_input_3d.shape[1]
        h5_dim = (x_dim, y_dim)
        h5_dataset_phasor_pix = fil.create_dataset(
            "h5_dataset_phasor_pix", shape=h5_dim, dtype=np.complex128
        )

        for x_start in range(0, x_dim, phasor_pix_data_size):
            for y_start in range(0, y_dim, phasor_pix_data_size):
                x_stop = min(x_start + phasor_pix_data_size, x_dim)
                y_stop = min(y_start + phasor_pix_data_size, y_dim)

                slice_term_pix = np.s_[x_start:x_stop, y_start:y_stop, :]
                sub_image_pix = data_input_3d[slice_term_pix]

                aligned_phasor_pix = np.zeros(
                    sub_image_pix.shape[:-1], dtype=np.complex128
                )

                for yyy in np.arange(sub_image_pix.shape[-3]):
                    for xxx in np.arange(sub_image_pix.shape[-2]):
                        aligned_phasor_pix[yyy, xxx] = phasor(
                            sub_image_pix[yyy, xxx, :], threshold, harmonic
                        )
                h5_dataset_phasor_pix[x_start:x_stop, y_start:y_stop] = (
                    aligned_phasor_pix
                )
    fil.close()


def plot_tau(list_value: Any = None, dfd_freq: float = 41.48e6, ax: Any = None) -> None:
    """Plot lifetime guides on the universal circle."""
    if ax is None:
        ax = gca()

    if list_value is None:
        list_value = np.array([1, 2, 3, 4, 6, 9, 16]) * 1e-9

    for i in list_value:
        x0 = np.cos(np.arctan(2 * np.pi * i * dfd_freq))
        y0 = np.sin(np.arctan(2 * np.pi * i * dfd_freq))
        x = [0, x0]
        y = [0, y0]
        g0s0_to_m_phi(x0, y0)

        ttt = y0 / x0
        x0 = 1 / (1 + ttt**2)
        y0 = np.sqrt(0.25 - (x0 - 0.5) ** 2)
        x = [0, x0]
        y = [0, y0]

        g0s0_to_m_phi(x0, y0)
        ax.plot(x, y, ".k")

        gamma = np.arctan2(y0, x0 - 0.5)

        ax.text(
            0.57 * np.cos(gamma) + 0.5,
            0.57 * np.sin(gamma),
            f" {int(i * 1e9)} ns",
            horizontalalignment="center",
            verticalalignment="center",
            color="red",
        )


def plot_funny_single_phasor(
    data: Array,
    correction_phi: float = 0,
    correction_m: float = 1.0,
    ax: Any = None,
) -> None:
    """Plot a histogram and its corrected phasor points."""
    N = sum(data)
    l = data.shape[0]
    w = np.linspace(0, 2 * np.pi, l)
    s = np.sin(w - correction_phi) / correction_m
    c = np.cos(w - correction_phi) / correction_m
    g0 = c * data
    s0 = s * data
    ax.plot(len(g0) * g0 / N, len(s0) * s0 / N, ".k")
    ax.plot(sum(g0 / N), sum(s0 / N), "or")
    ax.ylim(-10, 10)
    ax.xlim(-10, 10)
    ax.plot(np.cos(np.linspace(0, 2 * np.pi)), np.sin(np.linspace(0, 2 * np.pi)))


def plot_phasor(
    phasors: Array,
    bins_2dplot: int = 100,
    log_scale: bool = True,
    draw_universal_circle: bool = True,
    tau_labels: bool = True,
    quadrant: str = "all",
    fig: Any = None,
    ax: Any = None,
    dfd_freq: float = 41.48e6,
    cmap: str = "viridis",
) -> tuple[Any, Any]:
    """Plot phasor values and optional lifetime guides."""
    if fig is None:
        fig, ax = plt.subplots()
    elif fig is not None and ax is None:
        ax = gca()
    phasors_flat = flatten_and_remove_nan(phasors)

    if draw_universal_circle == True:
        plot_universal_circle(ax)

    if log_scale:
        im = ax.hist2d(
            np.real(phasors_flat),
            np.imag(phasors_flat),
            range=[[-1, 1], [-1, 1]],
            bins=bins_2dplot,
            norm=colors.LogNorm(),
            cmap=cmap,
        )
    else:
        im = ax.hist2d(
            np.real(phasors_flat),
            np.imag(phasors_flat),
            range=[[-1, 1], [-1, 1]],
            bins=bins_2dplot,
            cmap=cmap,
        )

    if quadrant == "all":
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    elif quadrant == "first":
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 0.6)

    if tau_labels == True:
        plot_tau(ax=ax, dfd_freq=dfd_freq)

    ax.set_xlabel("g")
    ax.set_ylabel("s")

    ax.set_aspect("equal", "box")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im[-1], cax=cax)
    cax.set_ylabel("Pixel counts")

    fig.tight_layout()

    return fig, ax


def _finite_bounds(array: ArrayLike) -> list[float]:
    """Return finite lower and upper bounds for an array."""
    finite = np.asarray(array, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [0.0, 1.0]
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if np.isclose(lower, upper):
        upper = lower + 1.0
    return [lower, upper]


def build_lifetime_equalizer(
    reference_lifetime: ArrayLike,
    lifetime_bounds: Sequence[float],
    weights: ArrayLike | None = None,
    n_quantiles: int = 2048,
    strength: float = 1.0,
    equalization_bins: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Build a lifetime-to-equalized-axis mapping from a reference distribution."""
    low, high = map(float, lifetime_bounds)
    if np.isclose(low, high):
        return np.array([low, high + 1.0]), np.array([0.0, 1.0])

    strength = float(strength)
    if strength <= 0.0:
        return np.array([low, high]), np.array([0.0, 1.0])

    reference_lifetime = np.asarray(reference_lifetime, dtype=float).ravel()
    if weights is None:
        weights = None
    else:
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.size != reference_lifetime.size:
            raise ValueError("weights must match reference_lifetime size")

    mask = np.isfinite(reference_lifetime)
    mask &= (reference_lifetime >= low) & (reference_lifetime <= high)
    if weights is not None:
        mask &= np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.array([low, high]), np.array([0.0, 1.0])

    if equalization_bins is None:
        n_bins = max(int(n_quantiles) - 1, 32)
    else:
        n_bins = max(int(equalization_bins), 32)

    histogram, edges = np.histogram(
        reference_lifetime[mask],
        bins=n_bins,
        range=(low, high),
        weights=None if weights is None else weights[mask],
    )
    histogram = histogram.astype(float)
    histogram = np.power(histogram, strength)
    keep = histogram > 0.0
    if not np.any(keep):
        return np.array([low, high]), np.array([0.0, 1.0])

    support = 0.5 * (edges[:-1] + edges[1:])
    support = support[keep]
    histogram = histogram[keep]
    cumulative = np.cumsum(histogram)
    equalized_axis = (cumulative - 0.5 * histogram) / cumulative[-1]
    support = np.concatenate(([low], support, [high]))
    equalized_axis = np.concatenate(([0.0], equalized_axis, [1.0]))
    return support, equalized_axis


def apply_lifetime_equalizer(
    lifetime: ArrayLike, support: ArrayLike, equalized_axis: ArrayLike
) -> Any:
    """Map lifetime values onto a precomputed equalized axis."""
    lifetime = np.asarray(lifetime, dtype=float)
    support = np.asarray(support, dtype=float)
    equalized_axis = np.asarray(equalized_axis, dtype=float)
    clipped = np.clip(lifetime, support[0], support[-1])
    equalized = np.interp(clipped, support, equalized_axis, left=0.0, right=1.0)
    equalized[~np.isfinite(lifetime)] = np.nan
    return equalized


def equalized_lifetime_tick_values(
    reference_lifetime: ArrayLike,
    lifetime_bounds: Sequence[float],
    weights: ArrayLike | None = None,
    n_ticks: int = 6,
    n_quantiles: int = 2048,
    strength: float = 1.0,
    equalization_bins: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return equalized-axis tick positions together with their lifetime labels."""
    support, equalized_axis = build_lifetime_equalizer(
        reference_lifetime,
        lifetime_bounds,
        weights=weights,
        n_quantiles=n_quantiles,
        strength=strength,
        equalization_bins=equalization_bins,
    )
    tick_pos = np.linspace(0.0, 1.0, int(n_ticks))
    tick_values = np.interp(tick_pos, equalized_axis, support)
    return tick_pos, tick_values


def _equal_luminance_colormap_array(colormap: str) -> FloatArray:
    """Return colormap colors scaled to equal luminance."""
    cmap = mpl.colormaps[colormap]
    colors = cmap(np.linspace(0.0, 1.0, cmap.N))[:, :3].copy()
    luminance = (299 * colors[:, 0] + 587 * colors[:, 1] + 114 * colors[:, 2]) / 1000.0
    max_luminance = np.max(luminance)
    for idx, lum in enumerate(luminance):
        if lum <= 0:
            continue
        scale = min(max_luminance / lum, 1.0 / max(np.max(colors[idx]), 1e-12))
        colors[idx] *= scale
    return colors


def _normalize_intensity(
    image: ArrayLike, intensity_bounds: Sequence[float]
) -> FloatArray:
    """Normalize finite intensity values to the requested range."""
    low, high = map(float, intensity_bounds)
    image = np.asarray(image, dtype=float)
    clipped = np.clip(image, low, high)
    normalized = (clipped - low) / max(high - low, np.finfo(float).eps)
    normalized[~np.isfinite(image)] = 0.0
    return cast(FloatArray, normalized)


def _flim_rgb_from_equalized_axis(
    image: ArrayLike,
    equalized_lifetime: ArrayLike,
    intensity_bounds: Sequence[float],
    colormap: str,
) -> FloatArray:
    """Map intensity and equalized lifetime arrays to RGB values."""
    cmap_array = _equal_luminance_colormap_array(colormap)
    n_colors = cmap_array.shape[0]
    intensity = _normalize_intensity(image, intensity_bounds)
    equalized_lifetime = np.asarray(equalized_lifetime, dtype=float)
    invalid = ~np.isfinite(image) | ~np.isfinite(equalized_lifetime)
    equalized_lifetime = np.clip(np.nan_to_num(equalized_lifetime, nan=0.0), 0.0, 1.0)
    idx = np.clip(
        np.floor(equalized_lifetime * (n_colors - 1)).astype(int), 0, n_colors - 1
    )
    rgb = cmap_array[idx] * intensity[..., None]
    rgb[invalid] = 0.0
    return cast(FloatArray, rgb)


def show_flim_equalized(
    image: ArrayLike,
    lifetime: ArrayLike,
    pxsize: float,
    pxdwelltime: float,
    lifetime_bounds: Sequence[float] | None = None,
    intensity_bounds: Sequence[float] | None = None,
    colormap: str = "gist_rainbow",
    fig: Any = None,
    ax: Any = None,
    equalization_reference: ArrayLike | None = None,
    equalization_weights: ArrayLike | None = None,
    n_quantiles: int = 2048,
    colorbar_ticks: int = 6,
    equalization_strength: float = 1.0,
    equalization_bins: int | None = None,
) -> tuple[Any, Any]:
    """Show a FLIM image using an equalized lifetime axis for the hue mapping."""
    image = np.asarray(image, dtype=float)
    lifetime = np.asarray(lifetime, dtype=float)
    pxsize = float(np.asarray(pxsize).squeeze())
    pxdwelltime = float(np.asarray(pxdwelltime).squeeze())

    if intensity_bounds is None:
        intensity_bounds = _finite_bounds(image)
    if lifetime_bounds is None:
        lifetime_bounds = _finite_bounds(lifetime)
    if equalization_reference is None:
        equalization_reference = lifetime

    support, equalized_axis = build_lifetime_equalizer(
        equalization_reference,
        lifetime_bounds,
        weights=equalization_weights,
        n_quantiles=n_quantiles,
        strength=equalization_strength,
        equalization_bins=equalization_bins,
    )
    lifetime_equalized = apply_lifetime_equalizer(lifetime, support, equalized_axis)
    rgb = _flim_rgb_from_equalized_axis(
        image, lifetime_equalized, intensity_bounds, colormap
    )

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    img_extent = (0, image.shape[1] * pxsize, 0, image.shape[0] * pxsize)
    ax.imshow(rgb, extent=img_extent)
    ax.axis("off")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    n_bar = 256
    intensity_gradient = np.linspace(intensity_bounds[0], intensity_bounds[1], n_bar)
    equalized_gradient = np.linspace(0.0, 1.0, n_bar)
    intensity_grid = np.tile(intensity_gradient, (n_bar, 1)).T
    equalized_grid = np.tile(equalized_gradient, (n_bar, 1))
    rgb_colorbar = _flim_rgb_from_equalized_axis(
        intensity_grid, equalized_grid, intensity_bounds, colormap
    )
    rgb_colorbar = np.moveaxis(rgb_colorbar, 0, 1)
    cax.imshow(
        rgb_colorbar,
        origin="lower",
        aspect="auto",
        extent=(intensity_bounds[0], intensity_bounds[1], 0.0, 1.0),
    )
    cax.set_xticks([int(intensity_bounds[0]), int(intensity_bounds[1])])
    cax.set_xlabel(f"Counts/{pxdwelltime} " + "$\\mathregular{\\mu s}$")
    tick_pos = np.linspace(0.0, 1.0, int(colorbar_ticks))
    tick_values = np.interp(tick_pos, equalized_axis, support)
    cax.set_yticks(tick_pos)
    cax.set_yticklabels([f"{tick:.2f}" for tick in tick_values])
    cax.set_ylabel("Lifetime (ns, equalized axis)")
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")

    scalebar = ScaleBar(1, "um", box_alpha=0, color="w", length_fraction=0.25)
    ax.add_artist(scalebar)
    plt.tight_layout()
    return fig, ax


def fourier_shift(data: Array, shift_angle: float = 0.0) -> FloatArray:
    """Shift a histogram along its final axis in Fourier space."""
    if len(data.shape) == 1:
        w = np.arange(data.shape[0])
    elif len(data.shape) > 1:
        w = np.asarray([np.arange(data.shape[0])] * data.shape[1]).T
        fft_hists = np.fft.fft(data)
    return cast(
        FloatArray,
        np.abs(np.fft.ifft(fft_hists * np.exp(1j * 2 * np.pi * shift_angle * w))),
    )


def cmap2d(intensity: Array, lifetime: Array, params: dict[str, Any]) -> FloatArray:
    """Map intensity and lifetime values to RGB colors."""
    sz = intensity.shape

    Hp = np.minimum(np.maximum(lifetime, params["minTau"]), params["maxTau"])

    Hn = ((Hp - params["minTau"]) / (params["maxTau"] - params["minTau"])) * params[
        "satFactor"
    ]
    Sn = np.ones(Hp.shape)
    Vn = (intensity - params["minInt"]) / (params["maxInt"] - params["minInt"])

    HSV = np.empty((sz[0], sz[1], 3))

    if params["invertColormap"] == True:
        Hn = params["satFactor"] - Hn

    Hn[np.not_equal(lifetime, Hp)] = params["outOfBoundsHue"]

    HSV[:, :, 0] = Hn.astype("float64")
    HSV[:, :, 1] = Sn.astype("float64")
    HSV[:, :, 2] = Vn.astype("float64")

    RGB = hsv_to_rgb(HSV)

    return RGB


def linear_shift(data: Array, shift_value: float, cyclic: bool = True) -> FloatArray:
    """Shift a one-dimensional array by linear interpolation."""
    xp = np.arange(0, data.shape[0])
    fp = data.copy()
    x = np.arange(0, data.shape[0]) - shift_value
    if cyclic:
        x = np.mod(x, data.shape[0])
    return cast(FloatArray, np.interp(x, xp, fp))


def correction_phasor(laser: Array, laser_irf: Array) -> Any:
    """Calculate the phasor correction between laser histograms."""
    phasor_laser = phasor(laser)
    phasor_laser_irf = phasor(laser_irf)

    angle_laser = np.angle(phasor_laser)
    angle_laser_irf = np.angle(phasor_laser_irf)

    return np.exp(-1j * (angle_laser - angle_laser_irf))


def threshold_intensity(intensity_map: Array, threshold: float = 0.15) -> Array:
    """Return intensity values above a fraction of the maximum."""
    max_counts = intensity_map.max()
    idx = intensity_map > threshold * max_counts
    thresholded_intensity = intensity_map[idx]
    return cast(Array, thresholded_intensity)


def threshold_phasor(
    intensity_map: Array, phasor_map: Array, threshold: float = 0.15
) -> Array:
    """Return phasor values whose intensity exceeds a relative threshold."""
    max_counts = intensity_map.max()
    idx = intensity_map > threshold * max_counts
    thresholded_phasor = phasor_map[idx].ravel()

    return cast(Array, thresholded_phasor)


def median_phasors(phasor_map: Array, window: int = 3) -> Array:
    """Apply a median filter to a phasor map."""
    filtered_phasors = median_filter(phasor_map, size=window)

    return cast(Array, filtered_phasors)


def calculate_irf_correction(hist_irf: Array, n_bins: int) -> Any:
    """Calculate the phase correction in histogram bins."""
    phasors_irf = calculate_phasor(hist_irf)
    angle_irf = np.angle(phasors_irf)
    correction_term = (angle_irf / (2 * np.pi)) * n_bins

    return correction_term


def m_phi_to_complex(m: Any, phi: Any) -> Any:
    """Convert phasor magnitude and phase to a complex value."""
    return m * np.exp(-1j * phi)


def find_shifts_irf_data(data_reference: Array, data: Array) -> FloatArray:
    """Estimate per-channel temporal shifts against reference data."""
    data_hist_irf = data_reference.sum((0, 1))
    data_hist_input = data.sum((0, 1))
    nch = data_hist_irf.shape[-1]

    shift_vec = np.empty(nch)
    for i in range(nch):
        shift_vec[i], *_ = cast(Any, phase_cross_correlation)(
            data_hist_irf[:, i],
            data_hist_input[:, i],
            upsample_factor=100,
            normalization=None,
        )

    return shift_vec


def find_temporal_shifts(data_4D: Array) -> FloatArray:
    """Estimate each channel's temporal shift against channel 12."""
    data_hist = data_4D.sum((0, 1))
    nch = data_hist.shape[-1]

    shift_vec = np.empty(nch)

    for i in range(nch):
        shift_vec[i], *_ = cast(Any, phase_cross_correlation)(
            data_hist[:, 12], data_hist[:, i], upsample_factor=100, normalization=None
        )

    return shift_vec


def align_decays(data_4D: Array, shift_vec: Array) -> FloatArray:
    """Align decay histograms after summing spatial dimensions."""
    nch = data_4D.shape[-1]
    data_hist = data_4D.sum((0, 1))
    data_hist_shift = np.empty_like(data_hist)
    for i in range(nch):
        data_hist_shift[:, i] = shift(
            data_hist[:, i], shift_vec[i], order=1, mode="grid-wrap"
        )

    return cast(FloatArray, data_hist_shift)


def align_image(data_4D: Array, shift_vector: Array) -> Array:
    """Shift each channel's histograms by its temporal offset."""
    nch = data_4D.shape[-1]
    image_4D_aligned = np.empty_like(data_4D)
    shift_dset = np.zeros((data_4D.ndim - 1, nch))
    shift_dset[-1] = shift_vector

    for i in range(nch):
        image_4D_aligned[..., i] = shift(
            data_4D[..., i], shift_dset[:, i], order=1, mode="grid-wrap"
        )

    return image_4D_aligned


def calibrate_phasor(
    data: Array,
    data_reference: Array,
    ch: int = 12,
    tau_m_reference: float = 2.5 * 10**-9,
    tau_phi_reference: float = 2.5 * 10**-9,
    laser_MHz: float = 40,
    h: int = 1,
    processing: str = "Open confocal",
) -> Any:
    """Correct measured phasors using a reference image."""
    if processing == "Single channel":
        shift_vec = find_shifts_irf_data(data_reference, data)
        data_aligned = align_image(data, shift_vec)

        data_hist_irf = data_reference.sum((0, 1))
        phasor_calibration = np.zeros(data_hist_irf.shape[-1], dtype=complex)

        for i in range(data_hist_irf.shape[-1]):
            phasor_calibration[i] = phasor(data_hist_irf[:, i])

        nch = data_hist_irf.shape[-1]
        m_phi_calibration = np.zeros((nch, 2))
        for i in range(nch):
            m_phi_calibration[i] = g0s0_to_m_phi(
                np.real(phasor_calibration[i]), np.imag(phasor_calibration[i])
            )

        laser_Hz = laser_MHz * 1e6
        k = 1 / (2 * np.pi * laser_Hz * h)
        phi_ref = math.atan2(tau_phi_reference, k)
        m_irf = 1
        phi_irf: Any
        if np.real(phasor_calibration[12]) < 0 and np.imag(phasor_calibration[12]) > 0:
            phi_irf = -(np.abs(m_phi_calibration[:, 1] - phi_ref))

        if np.real(phasor_calibration[12]) > 0 and np.imag(phasor_calibration[12]) > 0:
            phi_irf = -(np.abs(m_phi_calibration[:, 1] - phi_ref))

        if np.real(phasor_calibration[12]) > 0 and np.imag(phasor_calibration[12]) < 0:
            phi_irf = np.abs(m_phi_calibration[:, 1] - phi_ref)

        if np.real(phasor_calibration[12]) < 0 and np.imag(phasor_calibration[12]) < 0:
            phi_irf = np.abs(m_phi_calibration[:, 1] - phi_ref)

        phasor_irf = np.zeros((nch), dtype=complex)
        for i in range(nch):
            phasor_irf[i] = m_phi_to_complex(m_irf, phi_irf[i])

        phasor_data_pixels = phasor(data_aligned[:, :, :, ch])
        phasor_corrected = phasor_data_pixels / phasor_irf[ch]
        return phasor_corrected

    if processing == "Open confocal":
        shift_vec = find_shifts_irf_data(data_reference, data)
        data_aligned = align_image(data, shift_vec)
        shift_vec_reference = find_temporal_shifts(data_reference)
        data_aligned_reference = align_image(data_reference, shift_vec_reference)
        shift_vec_interchannel = find_temporal_shifts(data_aligned)
        data_aligned_final = align_image(data_aligned, shift_vec_interchannel)

        data_hist_irf_shift_sum = data_aligned_reference.sum((0, 1, -1))
        phasor_calibration_sum = cast(complex, phasor(data_hist_irf_shift_sum))
        m_phi_calibration_sum = g0s0_to_m_phi(
            np.real(phasor_calibration_sum), np.imag(phasor_calibration_sum)
        )
        laser_Hz = laser_MHz * 1e6
        k = 1 / (2 * np.pi * laser_Hz * h)
        phi_ref = math.atan2(tau_phi_reference, k)
        m_irf_sum = 1

        phi_irf_sum: Any
        if np.real(phasor_calibration_sum) < 0 and np.imag(phasor_calibration_sum) > 0:
            phi_irf_sum = -(np.abs(m_phi_calibration_sum[-1] - phi_ref))

        if np.real(phasor_calibration_sum) > 0 and np.imag(phasor_calibration_sum) > 0:
            phi_irf_sum = -(np.abs(m_phi_calibration_sum[-1] - phi_ref))

        if np.real(phasor_calibration_sum) > 0 and np.imag(phasor_calibration_sum) < 0:
            phi_irf_sum = np.abs(m_phi_calibration_sum[-1] - phi_ref)

        if np.real(phasor_calibration_sum) < 0 and np.imag(phasor_calibration_sum) < 0:
            phi_irf_sum = np.abs(m_phi_calibration_sum[-1] - phi_ref)

        phasor_irf_sum = m_phi_to_complex(m_irf_sum, phi_irf_sum)

        data_aligned_3D = np.sum(data_aligned_final, axis=-1)

        phasor_pixels_aligned = phasor(data_aligned_3D)
        phasor_corrected_pix_aligned = phasor_pixels_aligned / phasor_irf_sum
        return phasor_corrected_pix_aligned


def show_lifetime_histogram(
    phasors_calibrated: Array,
    method: str = "tau_phi",
    interval: tuple[float, float] = (-2, 6),
    bin_number: int = 50,
    dfd_freq: float = 40e6,
) -> None:
    """Plot a histogram of pixel lifetimes."""
    if method == "tau_phi":
        tau_phi = calculate_tau_phi(
            np.real(phasors_calibrated), np.imag(phasors_calibrated), dfd_freq=dfd_freq
        )
        tau_data = 1e9 * tau_phi.flatten()
        plt.figure()
        plt.hist(tau_data, range=interval, bins=bin_number)

    if method == "tau_m":
        tau_m = calculate_tau_m(
            np.real(phasors_calibrated), np.imag(phasors_calibrated), dfd_freq=dfd_freq
        )
        tau_data = 1e9 * tau_m.flatten()
        plt.figure()
        plt.hist(tau_data, range=interval, bins=bin_number)


def plot_flim_image(
    data_4D: Array,
    phasors_calibrated: Array,
    method: str = "tau_phi",
    pxsize: float = 0.04,
    pxdwelltime: float = 91,
    lifetime_bounds: tuple[float, float] = (1, 2.5),
    log_scale: bool = False,
    dfd_freq: float = 40e6,
) -> None:
    """Plot a lifetime map and its calibrated phasors."""
    if gra is None:
        raise ImportError("plot_flim_image requires brighteyes.imaging.plotting")

    if method == "tau_phi":
        data_histograms = data_4D.sum((-2, -1))
        tau_phi = calculate_tau_phi(
            np.real(phasors_calibrated), np.imag(phasors_calibrated), dfd_freq=dfd_freq
        )
        fig = plt.figure(figsize=(9, 6))
        gs = fig.add_gridspec(4, 4)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[2:4, 0:2])
        plot_phasor(
            phasors_calibrated,
            bins_2dplot=200,
            log_scale=log_scale,
            quadrant="first",
            fig=fig,
            ax=ax1,
        )
        gra.show_flim(
            data_histograms,
            tau_phi * 1e9,
            pxsize=pxsize,
            pxdwelltime=pxdwelltime,
            lifetime_bounds=lifetime_bounds,
            fig=fig,
            ax=ax2,
        )
        fig.tight_layout()

    if method == "tau_m":
        data_histograms = data_4D.sum((-2, -1))
        tau_m = calculate_tau_m(
            np.real(phasors_calibrated), np.imag(phasors_calibrated), dfd_freq=dfd_freq
        )
        fig = plt.figure(figsize=(9, 6))
        gs = fig.add_gridspec(4, 4)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[2:4, 0:2])
        plot_phasor(
            phasors_calibrated,
            bins_2dplot=200,
            log_scale=log_scale,
            quadrant="first",
            fig=fig,
            ax=ax1,
        )
        gra.show_flim(
            data_histograms,
            tau_m * 1e9,
            pxsize=pxsize,
            pxdwelltime=pxdwelltime,
            lifetime_bounds=lifetime_bounds,
            fig=fig,
            ax=ax2,
        )
        fig.tight_layout()
