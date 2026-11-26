# Origin: VicidominiLab/BrightEyes-ISM, src/brighteyes_ism/simulation/simulator.py at d799720.
"""Noisy image simulation from sample phantoms and PSFs."""

from __future__ import annotations

import copy as cp
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve

Array = NDArray[Any]


class ImageSimulator:
    """Generate noisy microscope images from phantoms and point spread functions."""

    def __init__(
        self,
        phantom: Array | None = None,
        psf: Array | None = None,
        signal: Any = 1,
        z_projection: bool = False,
    ) -> None:
        """Initialize an image simulator."""
        self.image: Array | None = None
        self.clean_image: Array | None = None
        self.phantom: Array | None = phantom
        self.psf: Array | None = psf
        self.signal: Array = np.array(signal)
        self.z_projection: bool = z_projection

    def blur(self) -> None:
        """Convolve the phantom with the point spread function."""
        gt = self.ground_truth
        psf = cast(Array, self.psf)
        phantom = cast(Array, self.phantom)
        num_ch = np.ndim(psf) - np.ndim(phantom)
        sz = psf.shape

        self.clean_image = np.empty(gt.shape + (psf.shape[-1],))
        clean_image = cast(Array, self.clean_image)

        if num_ch == 1:
            if np.ndim(phantom) == 3 and self.z_projection is True:
                for z in range(sz[0]):
                    for c in range(sz[-1]):
                        clean_image[z, ..., c] = convolve(
                            gt[z], psf[z, ..., c], mode="same"
                        )
                self.clean_image = clean_image.sum(0)
            else:
                for c in range(sz[-1]):
                    clean_image[..., c] = convolve(gt, psf[..., c], mode="same")
        elif num_ch == 0:
            if np.ndim(phantom) == 3 and self.z_projection is True:
                for z in range(sz[0]):
                    self.clean_image = convolve(gt[z], psf[z], mode="same")
                self.clean_image = cast(Array, self.clean_image).sum(0)
            else:
                self.clean_image = convolve(gt, psf, mode="same")
        else:
            raise Exception("The PSF has less dimensions than the phantom.")  # noqa: TRY002

        clean_image = cast(Array, self.clean_image)
        clean_image[clean_image < 0] = 0

    def poisson_noise(self) -> None:
        """Draw a Poisson-noise image from the clean image."""
        self.image = np.random.poisson(cast(Array, self.clean_image))

    def forward(self) -> None:
        """Generate the blurred image and add Poisson noise."""
        self.blur()
        self.poisson_noise()

    @property
    def ground_truth(self) -> Array:
        """Return the phantom scaled by the signal."""
        if np.ndim(self.signal) == 1:
            return cast(
                Array,
                np.einsum("i..., i -> i...", cast(Array, self.phantom), self.signal),
            )
        elif np.ndim(self.signal) == 0:
            return cast(Array, self.phantom) * self.signal
        else:
            raise Exception("The signal should be a scalar or a 1D array.")  # noqa: TRY002

    def copy(self) -> ImageSimulator:
        """Return a shallow copy of the simulator."""
        return cp.copy(self)
