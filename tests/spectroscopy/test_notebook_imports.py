"""Check migrated imports in bundled example notebooks."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize(
    "filename, cell",
    (
        ("examples/imaging/flism/APR_only.ipynb", 4),
        ("examples/imaging/flism/executed/APR_Phasors.ipynb", 4),
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
