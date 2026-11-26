"""Simulate detector geometry and point spread functions."""

from ._detector import (
    GridParametersLike,
    airy_to_hex,
    circle,
    custom_detector,
    det_coords,
    hex_grid,
    hex_to_airy,
    hexagon,
    pinhole_array,
    rect_grid,
    square,
    transform_detector,
)
from ._psf import (
    SPAD_PSF_2D,
    SPAD_PSF_3D,
    GridParameters,
    simSettings,
    singlePSF,
)

__all__ = [
    "SPAD_PSF_2D",
    "SPAD_PSF_3D",
    "GridParameters",
    "GridParametersLike",
    "airy_to_hex",
    "circle",
    "custom_detector",
    "det_coords",
    "hex_grid",
    "hex_to_airy",
    "hexagon",
    "pinhole_array",
    "rect_grid",
    "simSettings",
    "singlePSF",
    "square",
    "transform_detector",
]
