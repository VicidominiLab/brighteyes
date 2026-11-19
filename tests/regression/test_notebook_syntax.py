"""Check Python syntax in the experimental reconstruction example."""

import json
from pathlib import Path


def test_experimental_s2ism_notebook_cells_compile() -> None:
    """Compile the reconstruction flow without downloading or processing data."""
    root = Path(__file__).resolve().parents[2]
    filename = root / "examples/imaging/s2ism/Reconstruction_step_by_step_exp.ipynb"
    notebook = json.loads(filename.read_text("utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"s2ism-cell-{index}", "exec")
