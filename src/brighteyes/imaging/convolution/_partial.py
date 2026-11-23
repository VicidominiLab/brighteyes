"""Partial convolution implemented with tensor FFTs."""

from typing import TypeAlias

import torch
from torch.fft import fftn, ifftn, ifftshift

BoolPair: TypeAlias = tuple[bool, bool]


def partial_convolution(
    volume: torch.Tensor,
    kernel: torch.Tensor,
    dim1: str = "ijk",
    dim2: str = "jkl",
    axis: str = "jk",
    fourier: BoolPair = (False, False),
) -> torch.Tensor:
    """Convolve tensors over selected named axes using FFTs."""
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
