"""Check deterministic construction of the radius-sweep phantoms."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_phantom_accumulation_starts_from_zero(monkeypatch) -> None:
    """Build clean phantom backgrounds even when empty arrays contain old data."""
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (
            root
            / "examples/imaging/s2ism/Reconstruction_objects_increasing_radius.ipynb"
        ).read_text("utf-8")
    )
    original_empty = np.empty

    def poisoned_empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        result.fill(7)
        return result

    context = {
        "np": np,
        "grid": SimpleNamespace(pxsizex=40),
        "emPar": SimpleNamespace(airy_unit=575),
    }
    with monkeypatch.context() as patch:
        patch.setattr(np, "empty", poisoned_empty)
        exec("".join(notebook["cells"][7]["source"]), context)  # noqa: S102
        exec("".join(notebook["cells"][8]["source"]), context)  # noqa: S102
    center = context["Nx"] // 2 + 1
    np.testing.assert_array_equal(context["disks_focus"][:, 0, 0], 0)
    np.testing.assert_array_equal(context["disks_bkg"][:, 0, 0], 0)
    np.testing.assert_array_equal(context["disks_focus"][:, center, center], 1)
    np.testing.assert_array_equal(context["disks_bkg"][:, center, center], 0)
