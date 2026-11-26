"""Partial convolution implemented with tensor FFTs."""

from __future__ import annotations

import torch
from torch.fft import fftn, ifftn, ifftshift


def partial_convolution(
    volume: torch.Tensor,
    kernel: torch.Tensor,
    dim1: str = "ijk",
    dim2: str = "jkl",
    axis: str = "jk",
    fourier: tuple[bool, bool] = (False, False),
) -> torch.Tensor:
    """Convolve two tensors along selected axes using Fourier transforms.

    Return the real part of the inverse transform after shifting it along
    the convolution axes. Output axes follow `dim1`, then any labels from
    `dim2` that do not already appear in `dim1`.

    Parameters
    ----------
    volume
        Input data to convolve with the kernel.
    kernel
        Convolution kernel. For each convolution axis, use the same length
        as the volume or a length that can broadcast to it.
    dim1
        Axis labels for `volume`, one character per dimension in tensor order.
    dim2
        Axis labels for `kernel`, one character per dimension in tensor order.
        Use the same label for axes that should align.
    axis
        Labels of the axes to convolve along. Each must appear in both `dim1`
        and `dim2`.
    fourier
        Flags for `volume` and `kernel`, in that order. Set a flag to `True`
        if the corresponding input is already Fourier transformed along
        `axis`.
    """
    dim3 = dim1 + dim2
    dim3 = "".join(dict.fromkeys(dim3))

    dims = [dim1, dim2, dim3]
    axis_list = [[d.find(c) for c in axis] for d in dims]

    volume_fft = volume if fourier[0] else fftn(volume, dim=axis_list[0])
    kernel_fft = kernel if fourier[1] else fftn(kernel, dim=axis_list[1])

    conv = torch.einsum(f"{dim1},{dim2}->{dim3}", volume_fft, kernel_fft)
    conv = ifftn(conv, dim=axis_list[2])
    conv = ifftshift(conv, dim=axis_list[2])
    return torch.real(conv)
