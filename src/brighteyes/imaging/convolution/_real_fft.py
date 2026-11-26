# Origin: VicidominiLab/s2ISM, src/s2ism/s2ism.py at 10bb5fc.
"""Partial convolution implemented with real FFTs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import einsum, real
from torch.fft import ifftshift, irfftn, rfftn


def partial_convolution_rfft(
    kernel: torch.Tensor,
    volume: torch.Tensor,
    dim1: str = "ijk",
    dim2: str = "jkl",
    axis: str = "jk",
    fourier: tuple[bool | int, bool | int] = (False, False),
    padding: Sequence[int] | None = None,
) -> torch.Tensor:
    """Convolve selected dimensions with real FFTs."""
    dim3 = dim1 + dim2
    dim3 = "".join(sorted(set(dim3), key=dim3.index))

    dims = [dim1, dim2, dim3]
    axis_list = [[d.find(c) for c in axis] for d in dims]

    if padding is None:
        padding = [volume.size(d) for d in axis_list[1]]

    if fourier[0] == False:
        kernel_fft = rfftn(kernel, dim=axis_list[0], s=padding)
    else:
        kernel_fft = kernel

    if fourier[1] == False:
        volume_fft = rfftn(volume, dim=axis_list[1], s=padding)
    else:
        volume_fft = volume

    conv = einsum(f"{dim1},{dim2}->{dim3}", kernel_fft, volume_fft)

    conv = irfftn(conv, dim=axis_list[2], s=padding)  # inverse FFT of the product
    conv = ifftshift(
        conv, dim=axis_list[2]
    )  # Rotation of 180 degrees of the phase of the FFT
    conv = real(conv)  # Clipping to zero the residual imaginary part

    return conv
