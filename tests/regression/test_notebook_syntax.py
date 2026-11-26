"""Check Python syntax in the notebooks that download published data."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = (
    "examples/imaging/BrightEyes_ISM_analysis_demo.ipynb",
    "examples/imaging/s2ism/Reconstruction_step_by_step_exp.ipynb",
    "examples/imaging/s2ism/Reconstruction_lifetime.ipynb",
    "examples/imaging/s2ism/Reconstruction_time_lapse.ipynb",
)


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_download_notebook_cells_compile(notebook: str) -> None:
    """Compile every code cell of each download notebook without running it."""
    filename = Path(__file__).resolve().parents[2] / notebook
    cells = json.loads(filename.read_text("utf-8"))["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{notebook}-cell-{index}", "exec")
