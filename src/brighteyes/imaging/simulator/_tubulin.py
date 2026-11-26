# Origin: VicidominiLab/BrightEyes-ISM, src/brighteyes_ism/simulation/Tubulin_sim.py at d799720.
"""Tubulin phantom simulation utilities."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

FloatArray = NDArray[np.float64]


class tubSettings:
    """Hold tubulin phantom simulation parameters."""

    def __init__(self) -> None:
        """Initialize default simulation parameters."""
        self.xy_pixel_size: float = 5  # pixel size (nm)
        self.xy_dimension: int = 256  # dimension image simulation (px)
        self.xz_dimension: int = 1  # layer for the sample for 3D image (int)
        self.z_pixel: float = 1  # pixel size for 3D image (nm)
        self.n_filament: int = 10  # number of filaments (but it can be also random with an interval of [10,100] )
        self.radius_filament: float = 10  # radius of filaments (but it can be also random with an interval of [8-12] (nm) )
        self.intensity_filament: list[float] = [
            1,
            1,
        ]  # brightness of filaments [min,max]


def thetaVariation(z: int) -> tuple[float, float, float, float, float, float, float]:
    """Return angular parameters for filament growth."""
    initial_theta_xy = 0
    initial_theta_variation_xy = 2 * np.pi / 20
    theta_variation_xy = 2 * np.pi / 50
    theta_xy_boundaries = np.pi / 4
    initial_theta_variation_xz: float
    theta_variation_xz: float

    if z == 1:
        initial_theta_xz = 0
        initial_theta_variation_xz = 0
        theta_variation_xz = 0
    else:
        initial_theta_xz = 0
        initial_theta_variation_xz = 2 * np.pi / 100
        theta_variation_xz = np.pi / 50

    return (
        initial_theta_xy,
        initial_theta_variation_xy,
        theta_variation_xy,
        theta_xy_boundaries,
        initial_theta_xz,
        initial_theta_variation_xz,
        theta_variation_xz,
    )


def getElipsoid(
    xC: Any,
    yC: Any,
    zC: Any,
    xR: Any,
    yR: Any,
    zR: Any,
    cIn: Any,
    rIn: Any,
    dIn: Any,
) -> NDArray[np.bool_]:
    """Return points inside an ellipsoid."""
    elipsoidVol = (rIn - xC) ** 2 / xR**2 + (cIn - yC) ** 2 / yR**2 + (
        dIn - zC
    ) ** 2 / zR**2 <= 1
    return cast(NDArray[np.bool_], elipsoidVol)


def functionPhTub(tub: tubSettings) -> FloatArray:
    """Generate a two- or three-dimensional tubulin phantom."""
    phTub = np.zeros(
        (tub.xy_dimension, tub.xy_dimension, tub.xz_dimension)
    )  # Phantom initialization
    ps = tub.xy_pixel_size

    xR = tub.radius_filament / ps  # the radius of the filament in pixels
    yR = tub.radius_filament / ps
    zR = tub.radius_filament / (ps * tub.z_pixel)

    cIn, rIn, dIn = np.meshgrid(
        np.arange(0, tub.xy_dimension),
        np.arange(0, tub.xy_dimension),
        np.arange(1, tub.xz_dimension + 1),
    )

    xy_0, xy_0_var, xy_var, xy_bound, xz_0, xz_0_var, xz_var = thetaVariation(
        tub.xz_dimension
    )

    for i in tqdm(range(tub.n_filament)):
        currIntensity: Any
        currIntensity = tub.intensity_filament[0] + (
            tub.intensity_filament[1] - tub.intensity_filament[0]
        ) * np.random.rand(1, 1)
        theta_xy: Any = xy_0 - xy_0_var + 2 * xy_0_var * np.random.rand(1)
        theta_xz: Any = xz_0 - xz_0_var + 2 * xz_0_var * np.random.rand(1)

        xyC: Any = np.random.random_integers(0, tub.xy_dimension, 2)
        yC = xyC[1]
        xC: Any = 1
        zC: Any = np.random.random_integers(0, tub.xz_dimension, 1)
        zC = math.ceil(tub.xz_dimension / 2)

        elipsoid_vol = getElipsoid(xC, yC, zC, xR, yR, zR, cIn, rIn, dIn)

        phTub[elipsoid_vol] = currIntensity[0]
        max_iterations = tub.xy_dimension * 2

        for j in range(max_iterations):
            #       Calculate the next coord
            next_theta_xy = theta_xy - xy_var + 2 * xy_var * np.random.rand(1)
            next_theta_xz = theta_xz - xz_var + 2 * xz_var * np.random.rand(1)
            next_xC = xC + xR * np.cos(next_theta_xy)
            next_yC = yC + yR * np.sin(next_theta_xy)
            next_zC = zC + zR * np.sin(next_theta_xz)

            if (
                next_xC < 0
                or next_xC > tub.xy_dimension
                or next_yC < 0
                or next_yC > tub.xy_dimension
            ):
                print(
                    "tubulin filament out of the boundaries",
                    next_xC,
                    next_yC,
                    tub.xy_dimension,
                )
                break

            if next_zC < 0 or next_zC > tub.xz_dimension:
                theta_xy = next_theta_xy
                theta_xz = next_theta_xz + np.pi
                xC = next_xC
                yC = next_yC

            if next_theta_xy < -xy_bound or next_theta_xy > xy_bound:
                theta_xy = next_theta_xy - xy_var * np.sign(next_theta_xy)
                theta_xz = next_theta_xz
                xC = next_xC
                yC = next_yC
                zC = next_zC

            elipsoid_vol = getElipsoid(
                next_xC, next_yC, next_zC, xR, yR, zR, cIn, rIn, dIn
            )

            # set the volume to the target intensity
            phTub[elipsoid_vol] = currIntensity[0]

            theta_xy = next_theta_xy
            theta_xz = next_theta_xz
            xC = next_xC
            yC = next_yC
            zC = next_zC

    return phTub
