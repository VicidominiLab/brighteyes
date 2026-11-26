"""Regression checks for repaired imaging analysis functions."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from brighteyes.imaging import flism
from brighteyes.imaging.flism import FlimData
from brighteyes.imaging.plotting import depth_stack
from brighteyes.mcs.dataprep import Alignment

if TYPE_CHECKING:
    from numpy.typing import NDArray


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


def test_fit_maps_work_after_importing_flism() -> None:
    """Fit lifetime maps through MCS and FLISM once the FLISM package is imported."""
    time = np.arange(64) * 25 / 64
    irf = np.exp(-((np.arange(64) - 3.0) ** 2) / 1.28) + 1e-4
    data = np.random.default_rng(0).poisson(5, (2, 2, 64)).astype(float)
    for generate_fit_maps in (Alignment.generate_fit_maps, flism.generate_fit_maps):
        maps = generate_fit_maps(data, irf, time, 25.0, show_progress=False)
        assert maps["tau"].shape == (2, 2)
