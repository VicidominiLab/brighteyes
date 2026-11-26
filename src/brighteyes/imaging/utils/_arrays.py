"""Array utilities used by image-scanning microscopy."""

# Origin: VicidominiLab/BrightEyes-ISM, src/brighteyes_ism/analysis/Tools_lib.py at d799720.
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.linalg import inv
from scipy.special import kl_div as kullback_leibler
from scipy.special import rel_entr


if TYPE_CHECKING:
    from numpy.typing import NDArray

    Array = NDArray[np.generic]
    FloatArray = NDArray[np.float64]


def sigmoid(R: FloatArray, T: float, S: float) -> FloatArray:
    """Return a circularly symmetric sigmoid."""
    return 1 / (1 + np.exp((R - T) / S))


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


def GaussMultVar(X: FloatArray, Y: FloatArray, M1: FloatArray, M2: FloatArray) -> Array:
    """Evaluate a two-dimensional multivariate Gaussian."""
    S = np.asarray([X, Y])
    S = np.moveaxis(S, 0, 2) - M1

    A = inv(M2)

    B = np.einsum("ij, lmj -> ilm", A, S)
    C = np.einsum("ijk, kij -> ij", S, B)

    g = np.exp(-0.5 * C)

    return cast(np.ndarray, g)
