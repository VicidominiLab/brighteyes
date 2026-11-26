"""Regression checks for repaired imaging analysis functions."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from brighteyes.imaging.extraction.deconv.ism import FRC_MultiImg_RL_FFT_2
from brighteyes.imaging.flism import FlimData
from brighteyes.imaging.plotting import depth_stack

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_frc_multi_image_deconvolution_uses_six_value_resolution_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deconvolve both image stacks after reading the six-value FRC result."""
    frc_result = (0.5, *(np.array([], dtype=np.float64) for _ in range(5)))
    shifts = np.zeros((2, 2), dtype=np.float64)
    monkeypatch.setattr(
        "brighteyes.imaging.extraction.deconv.ism._deconvolution.FRC.FRC_resolution",
        lambda *_args, **_kwargs: frc_result,
    )
    monkeypatch.setattr(
        "brighteyes.imaging.extraction.deconv.ism._deconvolution.APR.APR",
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


def test_flmdata_loads_sample_without_irf_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load sample data when no IRF path is supplied."""
    with tempfile.TemporaryDirectory(dir=Path(__file__).parents[2]) as directory:
        sample_path = Path(directory) / "sample.h5"
        with h5py.File(sample_path, "w") as handle:
            handle.create_dataset("data", data=np.ones((1, 1, 8, 5)))
        monkeypatch.setattr(FlimData, "load_data", lambda _self, _path: None)

        sample = FlimData(data_path=str(sample_path))

        assert sample.data_hist.shape == (8, 5)


def test_depth_stack_renders_image_and_colorbar_headlessly() -> None:
    """Render a depth stack with an image and color bar on a headless backend."""
    depth_index = np.indices((4, 5)).sum(axis=0) % 3
    intensity = np.arange(1, 21).reshape(4, 5)
    stack: NDArray[np.float64] = np.stack(
        [np.where(depth_index == depth, intensity, 0) for depth in range(3)]
    ).astype(np.float64)

    figure, axes = depth_stack(stack, pxsize=(0.1, 0.2, 0.3), pxdwelltime=1.0)
    try:
        figure.canvas.draw()

        assert axes is figure.axes[0]
        assert len(figure.axes) == 2
        assert len(axes.images) == 1
        image = axes.images[0].get_array()
        assert image is not None
        assert image.shape == (4, 5, 3)
        assert np.isfinite(image).all()
        assert len(figure.axes[1].images) == 1
    finally:
        plt.close(figure)
