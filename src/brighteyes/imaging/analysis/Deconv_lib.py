# Origin: VicidominiLab/BrightEyes-ISM, src/brighteyes_ism/analysis/Deconv_lib.py at d799720.
# ruff: noqa: N999
"""Image deconvolution algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.linalg import inv, toeplitz
from scipy.ndimage import laplace
from scipy.signal import convolve
from tqdm import tqdm

from . import APR_lib as APR
from . import FRC_lib as FRC

if TYPE_CHECKING:
    from .._types import Array


def gauss2d(X: Array, Y: Array, mux: float, muy: float, sigma: float) -> Array:
    """Evaluate a two-dimensional Gaussian."""
    g = np.exp(-((X - mux) ** 2 + (Y - muy) ** 2) / (2 * sigma**2))
    return g


def disk2d(X: Array, Y: Array, mux: float, muy: float, T: float) -> Array:
    """Create a two-dimensional disk mask."""
    R = np.sqrt((X - mux) ** 2 + (Y - muy) ** 2)
    d = np.where(R < T, 1, 0)

    return d


def convolution_matrix(K: Array, I: Array) -> tuple[Array, Array]:
    """Build the matrix representation of a two-dimensional convolution."""
    # Part 1: generate doubly blocked toeplitz matrix

    # calculate sizes
    K_row_num, K_col_num = K.shape
    I_row_num, I_col_num = I.shape
    R_row_num = K_row_num + I_row_num - 1
    R_col_num = K_col_num + I_col_num - 1
    # pad the kernel
    K_pad = np.pad(
        K,
        ((0, R_row_num - K_row_num), (0, R_col_num - K_col_num)),
        "constant",
        constant_values=0,
    )
    # Assemble the list of Toeplitz matrices
    toeplitz_list: list[Array] = []
    for i in range(R_row_num):
        c = K_pad[i, :]
        r = np.r_[c[0], np.zeros(I_col_num - 1)]
        toeplitz_list.append(toeplitz(c, r).copy())
    # make a matrix with the indices of the block
    # of the doubly blocked Toeplitz matrix
    c = np.array(range(R_row_num))
    r = np.r_[c[0], c[-1:1:-1]]
    doubly_indices = np.array(toeplitz(c, r).copy())
    # assemble the doubly blocked toeplitz matrix
    toeplitz_blocks: list[Array] = []
    for i in range(R_row_num):
        row: list[Array] = []
        for j in range(I_row_num):
            row.append(toeplitz_list[doubly_indices[i, j]])
        toeplitz_blocks.append(np.hstack(row))
    toeplitz_m = np.vstack(toeplitz_blocks)

    # Part 2: pad and flatten input image

    dim = np.empty(2).astype(int)

    dim[0] = (K.shape[0] - 1) // 2
    dim[1] = (K.shape[1] - 1) // 2

    N_pad = ((dim[0], dim[0]), (dim[1], dim[1]))
    I_pad = np.pad(I, N_pad, "constant", constant_values=0)

    return toeplitz_m, I_pad.flatten()


def deconv_Wiener(
    h: Array,
    i: Array,
    reg: float = 0,
    regularization: str = "Tikhonov",
) -> Array:
    """Deconvolve an image with matrix-based Wiener filtering."""
    N, M = i.shape

    H, I = convolution_matrix(h, i)
    Ht = np.transpose(H)

    if regularization == "Tikhonov":
        R = np.eye(H.shape[1]) * reg

    elif regularization == "Laplace":
        Z = np.zeros(i.shape)
        Z[N // 2, M // 2] = 1
        l = laplace(Z)
        L, _ = convolution_matrix(l, i)
        Lt = np.transpose(L)
        R = np.matmul(Lt, L) * reg

    A = inv(np.matmul(Ht, H) + R)
    B = np.matmul(Ht, I)
    OUT = np.matmul(A, B)

    out = OUT.reshape(N, M)

    return cast("Array", out)


def deconv_Wiener_FFT(h: Array, i: Array, reg: float = 0) -> Array:
    """Deconvolve an image with FFT-based Wiener filtering."""
    H = np.fft.fft2(h)
    I = np.fft.fft2(i)

    A = H * np.conj(H) + reg
    B = np.conj(H) * I
    OUT = np.real(np.fft.ifft2(B / A))
    out = np.fft.fftshift(OUT)

    return out


def deconv_RL_FFT(
    h: Array,
    i: Array,
    max_iter: int = 50,
    epsilon: float | None = None,
    reg: float = 0,
    out: str = "last",
) -> Array | None:
    """Run Richardson-Lucy deconvolution with FFT-based convolution."""
    if out == "all":
        sz = [max_iter, i.shape[0], i.shape[1]]
        obj_all = np.empty(sz)

    if epsilon is None:
        epsilon = np.finfo(float).eps

    h = h / np.sum(h)  # PSF normalization
    hT = np.flip(h)
    obj = np.ones(i.shape)  # Initialization

    k = 0
    while k < max_iter:
        conv = convolve(obj, h, mode="same")
        A = np.where(conv < epsilon, 0, i / conv)
        B = convolve(hT, A, mode="same")
        C = obj / (1 + reg * obj)
        obj = B * C

        if out == "all":
            obj_all[k] = obj.copy()

        k += 1

    if out == "last":
        return obj
    elif out == "all":
        return obj_all
    return None


def MultiImg_RL_FFT(
    h: Array,
    i: Array,
    bkg: Array | None = None,
    max_iter: int = 50,
    pad: int | None = None,
    epsilon: float | None = None,
    reg: float = 0,
    out: str = "last",
) -> Array | None:
    """Run Richardson-Lucy deconvolution on multiple images."""
    if out == "all":
        sz = [max_iter] + list(i.shape[:-1])
        obj_all = np.empty(sz)

    if bkg is None:
        bkg = np.zeros(i.shape)

    if pad is not None:
        h = PadDataset(h, pad)
        i = PadDataset(i, pad)

    N = i.shape[-1]

    if epsilon is None:
        epsilon = np.finfo(float).eps

    h = h / np.sum(h)  # PSF normalization

    axis_to_flip = range(len(i.shape) - 1)
    hT = np.flip(h, axis=tuple(axis_to_flip))

    obj = np.ones(list(i.shape[:-1]))  # Initialization

    k = 0
    print("Multi-image deconvolution:")
    pbar = tqdm(total=max_iter)
    while k < max_iter:
        tmp = 0

        for n in range(N):
            with np.errstate(invalid="ignore", divide="ignore"):
                conv = convolve(obj, h[..., n], mode="same") + bkg[..., n]
                A = np.where(conv < epsilon, 0, i[..., n] / conv)
                B = convolve(A, hT[..., n], mode="same")
                tmp += B

        obj = obj * tmp / (1 + reg * obj)  # * s[n]

        if out == "all":
            obj_all[k] = obj.copy()

        k += 1
        pbar.update(1)

    if pad is not None:
        if out == "last":
            obj = UnpadDataset(obj, pad)
        elif out == "all":
            for n in range(max_iter):
                obj_all[n] = UnpadDataset(obj_all[n], pad)

    if out == "last":
        return obj
    elif out == "all":
        return obj_all
    return None


def PSF_FRC(i_1: Array, i_2: Array) -> Array:
    """Estimate channel PSFs from two images using Fourier ring correlation."""
    sz = i_1.shape

    ref = sz[-1] // 2

    frc_result = FRC.FRC_resolution(
        i_1[:, :, ref], i_2[:, :, ref], px=1, method="fixed"
    )

    sigma_frc = frc_result[0] / (2 * np.sqrt(2 * np.log(2)))

    usf = 100

    shift_frc, _ = APR.APR(i_1, usf, ref, 1)

    shift_frc_x, shift_frc_y = -shift_frc[:, 1], -shift_frc[:, 0]

    psf_frc = np.empty(i_1.shape)

    x = np.arange(sz[0]) - sz[0] // 2
    y = np.arange(sz[1]) - sz[1] // 2
    X, Y = np.meshgrid(x, y)

    Fingerprint = np.sum(i_1 + i_2, axis=(0, 1)).astype("float64")
    Fingerprint /= np.sum(Fingerprint)

    for i in range(sz[-1]):
        psf_frc[:, :, i] = gauss2d(X, Y, shift_frc_x[i], shift_frc_y[i], sigma_frc)
        psf_frc[:, :, i] = psf_frc[:, :, i] * Fingerprint[i] / np.sum(psf_frc[:, :, i])

    return psf_frc


def FRC_MultiImg_RL_FFT(
    dset: Array,
    max_iter: int = 50,
    pad: int | None = None,
    epsilon: float | None = None,
    reg: float = 0,
) -> tuple[Array | None, Array]:
    """Estimate a PSF, then deconvolve temporal samples."""
    Nt = dset.shape[-2]

    if Nt % 2 == 0:
        img_even = dset[:, :, 0::2, :].sum(axis=-2)
        img_odd = dset[:, :, 1::2, :].sum(axis=-2)
    else:
        img_even = dset[:, :, 0:-1:2, :].sum(axis=-2)
        img_odd = dset[:, :, 1::2, :].sum(axis=-2)

    psf_frc = PSF_FRC(img_even, img_odd)

    img = dset.sum(axis=-2)
    obj = MultiImg_RL_FFT(
        psf_frc, img, max_iter=max_iter, pad=pad, epsilon=epsilon, reg=reg
    )

    return obj, psf_frc


def MultiImg_RL_FFT_2(
    h: Array,
    i: Array,
    max_iter: int = 50,
    pad: int | None = None,
    epsilon: float | None = None,
    reg: float = 0,
) -> Array:
    """Deconvolve each channel with Richardson-Lucy iterations."""
    if pad is not None:
        h = PadDataset(h, pad)
        i = PadDataset(i, pad)

    N = i.shape[-1]

    if epsilon is None:
        epsilon = np.finfo(float).eps

    for n in range(N):  # PSF normalization
        h[:, :, n] /= np.sum(h[:, :, n])

    obj = np.ones(i.shape)  # Initialization

    for n in range(N):
        obj[:, :, n] = deconv_RL_FFT(
            h[:, :, n], i[:, :, n], max_iter=max_iter, epsilon=epsilon, reg=reg
        )

    if pad is not None:
        obj = UnpadDataset(obj, pad)

    return obj


def FRC_MultiImg_RL_FFT_2(
    i_1: Array,
    i_2: Array,
    max_iter: int = 50,
    pad: int | None = None,
    epsilon: float | None = None,
    reg: float = 0,
) -> tuple[Array | None, Array, float, Array]:
    """Estimate a PSF, then deconvolve each channel."""
    # RL each images individually. returns Nd deconvoluted images
    sz = i_1.shape

    ref = sz[-1] // 2

    res, _, _, _, _, _ = FRC.FRC_resolution(
        i_1[:, :, ref], i_2[:, :, ref], px=1, method="fixed"
    )

    sigma_frc = res / (2 * np.sqrt(2 * np.log(2)))

    usf = 100

    shift_frc, _ = APR.APR(i_1, usf, ref, 1)

    shift_frc_x, shift_frc_y = -shift_frc[:, 1], -shift_frc[:, 0]

    psf_frc = np.empty(i_1.shape)

    x = np.arange(sz[0]) - sz[0] // 2
    y = np.arange(sz[1]) - sz[1] // 2
    X, Y = np.meshgrid(x, y)

    for i in range(sz[-1]):
        psf_frc[:, :, i] = gauss2d(X, Y, shift_frc_x[i], shift_frc_y[i], sigma_frc)

    obj = MultiImg_RL_FFT_2(
        psf_frc, i_1, max_iter=max_iter, pad=pad, epsilon=epsilon, reg=reg
    )

    return obj, shift_frc, res, psf_frc


def PadDataset(img: Array, pad_width: int) -> Array:
    """Pad the spatial dimensions of an image dataset."""
    pad = [[pad_width, pad_width]] * (img.ndim - 1)
    pad.append([0, 0])

    img_pad = np.pad(img, pad, mode="constant")

    return img_pad


def UnpadDataset(img: Array, pad_width: int) -> Array:
    """Remove padding from the spatial dimensions of an image dataset."""
    img_unpad = img[pad_width:-pad_width, pad_width:-pad_width, ...]
    return img_unpad
