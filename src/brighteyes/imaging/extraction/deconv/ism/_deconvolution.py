# Origin: VicidominiLab/BrightEyes-ISM, src/brighteyes_ism/analysis/Deconv_lib.py at d799720.
"""Image deconvolution algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import convolve
from tqdm import tqdm

if TYPE_CHECKING:
    from ...._types import Array


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
    if pad is not None:
        h = PadDataset(h, pad)
        i = PadDataset(i, pad)

    if out == "all":
        sz = [max_iter] + list(i.shape[:-1])
        obj_all = np.empty(sz)

    if bkg is None:
        bkg = np.zeros(i.shape)

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
            obj_all = obj_all[:, pad:-pad, pad:-pad, ...]

    if out == "last":
        return obj
    elif out == "all":
        return obj_all
    return None


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
