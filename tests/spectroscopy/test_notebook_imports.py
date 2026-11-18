"""Check migrated imports in bundled example notebooks."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

FCS_NOTEBOOKS = (
    "examples/spectroscopy/SPAD-FCS_analysis_example/FCS_Alexa488AB_example.ipynb",
    (
        "examples/spectroscopy/SPAD-FCS_analysis_example/.ipynb_checkpoints/"
        "FCS_Alexa488AB_example-checkpoint.ipynb"
    ),
)


@pytest.mark.parametrize(
    "filename, cell",
    (
        ("examples/imaging/flism/APR_only.ipynb", 4),
        ("examples/imaging/flism/executed/APR_Phasors.ipynb", 4),
        (FCS_NOTEBOOKS[0], 4),
        (FCS_NOTEBOOKS[1], 4),
        (
            (
                "examples/spectroscopy/SPAD-FFS_ttm_analysis_example/"
                "SPAD-FFS_YG_beads_TTM_example.ipynb"
            ),
            2,
        ),
    ),
)
def test_example_imports_without_standalone_repositories(filename, cell):
    """Resolve example imports using the monorepo installation alone."""
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / filename).read_text("utf-8"))
    tree = ast.parse("".join(notebook["cells"][cell]["source"]))
    tree.body = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    exec(compile(tree, filename, "exec"), {})  # noqa: S102


@pytest.mark.parametrize("filename", FCS_NOTEBOOKS)
def test_fcs_notebook_draws_detector_fingerprint(filename):
    """Plot summed photon counts in a 5x5 detector layout."""
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / filename).read_text("utf-8"))
    context = {}
    exec("".join(notebook["cells"][4]["source"]), context)  # noqa: S102
    plt = context["plt"]
    plt.switch_backend("Agg")
    figures = set(plt.get_fignums())
    context["data"] = np.stack((np.arange(1, 26), np.full(25, 2)))
    try:
        exec("".join(notebook["cells"][8]["source"]), context)  # noqa: S102
        expected = np.arange(3, 28)
        np.testing.assert_array_equal(context["dummy"], expected)
        figure = plt.gcf()
        figure.canvas.draw()
        np.testing.assert_array_equal(
            figure.axes[0].images[0].get_array(), expected.reshape(5, 5)
        )
    finally:
        for number in set(plt.get_fignums()) - figures:
            plt.close(number)


def test_ttm_notebook_preserves_in_place_processing_results():
    """Keep arrival data available through histogram alignment and filtering."""
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (
            root / "examples/spectroscopy/SPAD-FFS_ttm_analysis_example/"
            "SPAD-FFS_YG_beads_TTM_example.ipynb"
        ).read_text("utf-8")
    )
    context = {}
    exec("".join(notebook["cells"][2]["source"]), context)  # noqa: S102
    bins = np.arange(32)
    data = SimpleNamespace(
        microtime=1.0,
        microbintime=1.0,
        det0=np.array([[1.0, 2.0], [5.0, 3.0]]),
        hist0=np.column_stack((bins, np.exp(-(((bins - 16) / 3) ** 2)))),
    )
    filters = np.zeros((1, 32, 2))
    filters[0, 2] = [0.8, 0.2]
    filters[0, 3] = [0.6, 0.4]
    context.update(data=data, filtersTheo=filters)
    exec("".join(notebook["cells"][22]["source"]), context)  # noqa: S102
    exec("".join(notebook["cells"][39]["source"]), context)  # noqa: S102
    assert context["data"] is data
    np.testing.assert_array_equal(data.det0[:, 2:], [[0.8, 0.2], [0.6, 0.4]])
    assert data.Ahist0.shape == data.hist0.shape
