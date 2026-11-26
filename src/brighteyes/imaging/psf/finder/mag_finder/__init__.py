"""Estimate magnification from detector shift vectors."""

from ._find import (
    find_mag,
    loss_minimizer,
    loss_shift,
    rect,
    scalar_psf,
    scalar_psf_det,
    shift_value,
)

__all__ = [
    "find_mag",
    "loss_minimizer",
    "loss_shift",
    "rect",
    "scalar_psf",
    "scalar_psf_det",
    "shift_value",
]
