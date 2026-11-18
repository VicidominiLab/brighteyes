"""Check the reconstruction notebook's per-plane diagnostic curves."""

import json
from importlib import import_module
from pathlib import Path

import numpy as np


def test_kl_diagnostic_preserves_plane_and_iteration_axes() -> None:
    """Calculate each plane's divergence across all reconstruction iterations."""
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (
            root / "examples/imaging/s2ism/Reconstruction_step_by_step_simulation.ipynb"
        ).read_text("utf-8")
    )
    ground_truth = np.ones((2, 2, 2))
    factors = np.array([[1.0, 1.0], [2.0, 4.0], [4.0, 2.0]])
    context = {
        "np": np,
        "tools": import_module("brighteyes.imaging.analysis.Tools_lib"),
        "ground_truth": ground_truth,
        "recon_ph": factors[:, :, None, None] * ground_truth,
    }
    exec("".join(notebook["cells"][29]["source"]), context)  # noqa: S102
    expected = -4 * np.log([[1.0, 2.0, 4.0], [1.0, 4.0, 2.0]])
    np.testing.assert_allclose(context["kullback_leibler"], expected)
