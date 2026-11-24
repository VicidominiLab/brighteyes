"""Regression checks for repaired imaging analysis functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest

from brighteyes.imaging.analysis.Deconv_lib import FRC_MultiImg_RL_FFT_2
from brighteyes.imaging.analysis.Graph_lib import depth_stack

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_frc_multi_image_deconvolution_uses_six_value_resolution_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deconvolve both image stacks after reading the six-value FRC result."""
    frc_result = (0.5, *(np.array([], dtype=np.float64) for _ in range(5)))
    shifts = np.zeros((2, 2), dtype=np.float64)
    monkeypatch.setattr(
        "brighteyes.imaging.analysis.Deconv_lib.FRC.FRC_resolution",
        lambda *_args, **_kwargs: frc_result,
    )
    monkeypatch.setattr(
        "brighteyes.imaging.analysis.Deconv_lib.APR.APR",
        lambda *_args, **_kwargs: (shifts, None),
    )

    images = np.ones((8, 8, 2), dtype=np.float64)
    result = FRC_MultiImg_RL_FFT_2(images, images, max_iter=1)

    deconvolved = result[0]
    psf = result[3]
    assert deconvolved is not None
    assert psf is not None
    assert deconvolved.shape == images.shape
    assert np.isfinite(deconvolved).all()
    np.testing.assert_array_equal(result[1], shifts)
    assert result[2] == 0.5
    assert psf.shape == images.shape


def test_depth_stack_renders_image_and_colorbar_headlessly() -> None:
    """Render a depth stack with an image and color bar on a headless backend."""
    stack: NDArray[np.float64] = np.arange(60, dtype=np.float64).reshape(3, 4, 5)

    figure, axes = depth_stack(stack, pxsize=(0.1, 0.2, 0.3), pxdwelltime=1.0)
    try:
        figure.canvas.draw()

        assert axes is figure.axes[0]
        assert len(figure.axes) == 2
        assert len(axes.images) == 1
        image = axes.images[0].get_array()
        assert image is not None
        assert image.shape == (4, 5, 3)
        assert len(figure.axes[1].images) == 1
    finally:
        plt.close(figure)
