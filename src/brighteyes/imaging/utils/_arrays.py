"""Array utilities used by image-scanning microscopy."""

# Origin: VicidominiLab/BrightEyes-ISM, src/brighteyes_ism/analysis/Tools_lib.py at d799720.
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.linalg import eig, inv
from scipy.optimize import curve_fit
from scipy.special import kl_div as kullback_leibler
from scipy.special import rel_entr

from ._frc import radial_profile

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Array = NDArray[np.generic]
    FloatArray = NDArray[np.float64]


def sigmoid(R: FloatArray, T: float, S: float) -> FloatArray:
    """Return a circularly symmetric sigmoid."""
    return 1 / (1 + np.exp((R - T) / S))


def low_pass(img: Array, T: float, S: float, data: str = "real") -> Array:
    """Apply a low-pass sigmoidal filter to an image."""
    if data == "real":
        img_fft: np.ndarray = np.fft.fftn(img, axes=(0, 1))
        img_fft = np.fft.fftshift(img_fft, axes=(0, 1))
    elif data == "fourier":
        img_fft = img
    else:
        raise ValueError("data has to be 'real' or 'fourier'")

    Nx = np.shape(img_fft)[0]
    Ny = np.shape(img_fft)[1]
    cx = int((Nx + np.mod(Nx, 2)) / 2)
    cy = int((Ny + np.mod(Ny, 2)) / 2)

    x = (np.arange(Nx) - cx) / Nx
    y = (np.arange(Ny) - cy) / Ny

    X, Y = np.meshgrid(x, y)
    R: FloatArray = np.sqrt(X**2 + Y**2)

    sig = sigmoid(R, T, S)

    img_filt = cast(Callable[..., np.ndarray], np.einsum)(
        "ij..., ij -> ij...", img_fft, sig
    )

    if data == "real":
        img_filt = np.fft.ifftshift(img_filt, axes=(0, 1))
        img_filt = np.fft.ifftn(img_filt, axes=(0, 1))
        img_filt = np.abs(img_filt)

    return img_filt


# %%


def Reorder(dset: Array, inOrder: str, outOrder: str = "rzxytc") -> Array:
    """Reorder dataset dimensions and add missing dimensions."""
    data = dset.copy()

    Nout = len(outOrder)
    Ndim = len(inOrder)

    if Ndim < Nout:
        # check where the current dimensions are located
        idx = np.empty(Nout)
        for n, c in enumerate(outOrder):
            idx[n] = np.char.find(inOrder, c)
        idx = idx.astype("int")

        # add missing dimensions
        new_axis_slices: list[slice | None] = []
        for i in idx:
            if i == -1:
                new_axis_slices.append(np.newaxis)
            else:
                new_axis_slices.append(np.s_[:])

        data = data[tuple(new_axis_slices)]

        # reorder final dimensions
        idx2 = np.empty(Ndim)
        for n, c in enumerate(inOrder):
            idx2[n] = np.char.find(outOrder, c)
        idx2 = idx2.astype("int")

        order = idx.copy()
        order[np.where(idx != -1)] = idx2
        order[np.where(idx == -1)] = np.argwhere(idx == -1).flatten()

        data = np.moveaxis(data, np.arange(Nout), order)

    else:
        # check where the dimensions are located
        idx = np.empty(Ndim)
        for n, c in enumerate(inOrder):
            idx[n] = np.char.find(outOrder, c)
        idx = idx.astype("int")

        # remove undesired dimensions
        existing_axis_slices: list[slice | int] = []
        for i in idx:
            if i == -1:
                existing_axis_slices.append(np.s_[0])
            else:
                existing_axis_slices.append(np.s_[:])

        data = data[tuple(existing_axis_slices)]

        # reorder remaining dimensions
        order = idx[idx != -1]
        data = np.moveaxis(data, np.arange(Nout), order)

    return data


def CropEdge(
    dset: Array, npx: int = 10, edges: str = "l", order: str = "rzxytc"
) -> Array:
    """Crop dataset edges in the xy plane."""
    default_order = "rzxytc"

    dset_cropped = Reorder(dset, inOrder=order, outOrder=default_order)

    if "l" in edges:
        dset_cropped = dset_cropped[..., npx:, :, :, :]

    if "r" in edges:
        dset_cropped = dset_cropped[..., :-npx, :, :, :]

    if "u" in edges:
        dset_cropped = dset_cropped[..., :, npx:, :, :]

    if "d" in edges:
        dset_cropped = dset_cropped[..., :, :-npx, :, :]

    return Reorder(dset_cropped, inOrder=default_order, outOrder=order)


def DownSample(dset: Array, ds: int = 2, order: str = "rzxytc") -> Array:
    """Downsample a dataset on the xy plane."""
    default_order = "rzxytc"

    dset = Reorder(dset, inOrder=order, outOrder=default_order)

    dset_ds = dset[..., ::ds, ::ds, :, :]

    return Reorder(dset_ds, inOrder=default_order, outOrder=order)


def UpSample(
    dset: Array, us: int = 2, npx: str = "even", order: str = "rzxytc"
) -> Array:
    """Upsample a dataset on the xy plane."""
    default_order = "rzxytc"

    dset = Reorder(dset, inOrder=order, outOrder=default_order)

    sz = dset.shape

    if npx == "even":
        sz_us = np.asarray(sz)
        sz_us[2] = sz_us[2] * us
        sz_us[3] = sz_us[3] * us
    elif npx == "odd":
        sz_us = np.asarray(sz)
        sz_us[2] = sz_us[2] * us - 1
        sz_us[3] = sz_us[3] * us - 1

    dset_us = np.zeros(sz_us)
    dset_us[..., ::us, ::us, :, :] = dset

    return Reorder(dset_us, inOrder=default_order, outOrder=order)


def ArgMaxND(data: Array) -> tuple[tuple[int, ...], np.generic]:
    """Return the maximum value and its indices."""
    idx = np.argmax(data)

    mx = np.array(data).ravel()[idx]

    arg = np.unravel_index(idx, np.array(data).shape)

    return cast(tuple[int, ...], arg), mx


def FWHM(
    x: FloatArray, y: FloatArray, height: float = 0.5
) -> tuple[float, list[float]]:
    """Calculate the full width at half maximum of a curve."""
    height_half_max = np.max(y) * height
    index_max = np.argmax(y)
    x_low = np.interp(height_half_max, y[:index_max], x[:index_max])
    x_high = np.interp(height_half_max, np.flip(y[index_max:]), np.flip(x[index_max:]))
    fwhm = x_high - x_low

    return cast(tuple[float, list[float]], (fwhm, [x_low, x_high]))


def RadialSpectrum(
    img: Array, pxsize: float = 1, normalize: bool = True
) -> tuple[Array, Array]:
    """Calculate the radial frequency spectrum of an image."""
    fft_img = np.fft.fftn(img, axes=[0, 1])
    fft_img = np.abs(np.fft.fftshift(fft_img, axes=[0, 1]))

    sx, sy = fft_img.shape
    c = (sx // 2, sy // 2)

    space_f = np.fft.fftfreq(sx, pxsize)[: c[0]]

    profile, counts = radial_profile(fft_img, c)

    ftR = profile[: c[0]] / counts[: c[0]]

    ftR = np.real(ftR)

    if normalize == True:
        ftR /= np.max(ftR)

    return ftR, space_f


def fingerprint(dset: Array, volumetric: bool = False) -> Array:
    """Sum detector channels into a spatial fingerprint."""
    N = int(np.sqrt(dset.shape[-1]))

    if volumetric == True:
        Nz = dset.shape[0]
        f = np.empty((Nz, N * N))
        axis = tuple(range(1, dset.ndim - 1))
        f = np.sum(dset, axis=axis)
        f = f.reshape(Nz, N, N)
    else:
        axis = tuple(range(dset.ndim - 1))
        f = np.sum(dset, axis=axis)
        f = f.reshape(N, N)
    return cast(np.ndarray, f)


def point_cloud_from_img(dset: Array) -> Array:
    """Convert an image or stack into a point cloud matrix."""
    shape = dset.shape

    N = dset.size

    indices = np.array(np.unravel_index(range(N), shape)).T

    values = dset.flatten()

    point_cloud_matrix = np.column_stack((indices, values))

    return point_cloud_matrix


def kl_divergence(
    ground_truth: Array,
    reconstruction: Array,
    remove_inf: bool = True,
    intensity_offset: bool = False,
    normalize_entries: bool = False,
) -> Array:
    """Calculate divergence for each reconstruction iteration."""
    kl_div = kullback_leibler if intensity_offset else rel_entr

    if normalize_entries is True:
        norm_gt = ground_truth.sum()
        norm_data = reconstruction.sum(axis=(-1, -2))

        gt = ground_truth.copy()
        gt = np.divide(ground_truth, norm_gt, out=gt, where=(norm_gt > 0))

        data = np.moveaxis(reconstruction, 0, -1).copy()
        data = np.divide(data, norm_data, out=data, where=(norm_data > 0))
        data = np.moveaxis(data, -1, 0)
    else:
        gt = ground_truth
        data = reconstruction

    n_iter = reconstruction.shape[0]
    kl = np.empty(n_iter)

    for n in range(n_iter):
        kl_iter = kl_div(gt, data[n])
        if remove_inf is True:
            kl_iter[np.isposinf(kl_iter)] = 0
        kl[n] = kl_iter.sum(axis=(-2, -1))

    return kl


def normalized_absolute_difference(ground_truth: Array, reconstruction: Array) -> Array:
    """Return the normalized absolute difference between two images."""
    tot_ref = ground_truth.sum()
    tot_img = reconstruction.sum()

    nad = np.abs(reconstruction / tot_img - ground_truth / tot_ref)

    return cast(np.ndarray, nad)


def check_saturation(dset: Array, sat_map: Array | None = None) -> None:
    """Print saturation counts for each detector channel."""
    if sat_map is None:
        sat_map = np.ones((5, 5)) * 4
        sat_map[1:-1, 1:-1] = 5
        sat_map[2, 1:-1] = 6
        sat_map[1:-1, 2] = 6
        sat_map[2, 2] = 10
        sat_map = 2**sat_map - 1
        sat_map = sat_map.flatten()

    n_ch = dset.shape[-1]
    n_sat = np.empty(n_ch).astype("int")
    n_tot = np.size(dset[..., 0])

    print("\nSaturated pixels: \n")

    for c in range(n_ch):
        n_sat[c] = np.size(dset[..., c][dset[..., c] == sat_map[c]])
        percent = 100 * n_sat[c] / n_tot
        print(rf"Channel {c:02d}: {n_sat[c]}/{n_tot} ({percent:.2f} %)")


def GaussMultVar(X: FloatArray, Y: FloatArray, M1: FloatArray, M2: FloatArray) -> Array:
    """Evaluate a two-dimensional multivariate Gaussian."""
    S = np.asarray([X, Y])
    S = np.moveaxis(S, 0, 2) - M1

    A = inv(M2)

    B = np.einsum("ij, lmj -> ilm", A, S)
    C = np.einsum("ijk, kij -> ij", S, B)

    g = np.exp(-0.5 * C)

    return cast(np.ndarray, g)


def fit_to_gaussian(
    img: Array,
    pxsize: float,
    baseline: bool = False,
    p0: tuple[float, ...] | None = None,
) -> tuple[Array, Array, Array]:
    """Fit an image to a multivariate Gaussian."""
    sz = img.shape

    y = pxsize * (np.arange(sz[0]) - sz[0] // 2)
    x = pxsize * (np.arange(sz[1]) - sz[1] // 2)

    X, Y = np.meshgrid(x, y)

    if baseline is False:
        if p0 is None:
            p0 = (0, 0, 1000, 0, 1000, 1)
        fit_model: Callable[..., Array] = lambda xdata, a, b, c, d, e, f: (
            f
            * GaussMultVar(
                xdata[0].reshape(sz),
                xdata[1].reshape(sz),
                np.asarray([a, b]),
                np.asarray([[c, d], [d, e]]),
            ).ravel()
        )
    elif baseline is True:
        if p0 is None:
            p0 = (0, 0, 1000, 0, 1000, 1, 0)
        fit_model = lambda xdata, a, b, c, d, e, f, g: (
            g
            + f
            * GaussMultVar(
                xdata[0].reshape(sz),
                xdata[1].reshape(sz),
                np.asarray([a, b]),
                np.asarray([[c, d], [d, e]]),
            ).ravel()
        )

    xdata = np.vstack((X.ravel(), Y.ravel()))

    popt, _ = curve_fit(fit_model, xdata, img.ravel(), p0)

    img_fit = fit_model(xdata, *popt).reshape(sz)

    var_matrix = np.asarray([[popt[2], popt[3]], [popt[3], popt[4]]])
    var_matrix_diag = np.diag(eig(var_matrix)[0])
    sigma_matrix_diag = np.sqrt(var_matrix_diag)

    # D4sigma = 4 * np.sqrt(sigma_matrix_diag) / 1e3

    return img_fit, sigma_matrix_diag, popt
