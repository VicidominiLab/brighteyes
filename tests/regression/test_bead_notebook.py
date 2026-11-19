"""Compare measured bead notebook stages with frozen numerical outputs."""

import hashlib
import json
import re
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from brighteyes import datasets
from brighteyes.imaging.psf_distiller import distiller


def test_bead_notebook_reads_coordinates_without_spectroscopy(
    tmp_path, monkeypatch
) -> None:
    """Load tab-separated bead coordinates through the notebook's actual imports."""
    notebook = json.loads(
        Path("examples/imaging/psf_distiller/distiller_test_beads.ipynb").read_text(
            "utf-8"
        )
    )
    coordinates = np.array([[20.5, 31.25], [41.0, 52.5]])
    path = tmp_path / "clicked_points.csv"
    path.write_text("20.5\t31.25\n41.0\t52.5\n")
    monkeypatch.chdir(tmp_path)
    context = {}
    exec("".join(notebook["cells"][3]["source"]), context)  # noqa: S102
    read = next(
        line.strip()
        for line in notebook["cells"][9]["source"]
        if "clicked_points = np.loadtxt(" in line
    )
    exec(read, context)  # noqa: S102
    np.testing.assert_array_equal(context["clicked_points"], coordinates)


@pytest.mark.parametrize(
    "filename, field",
    (
        ("tests/data/deconv3/manifest.json", "raw_input"),
        ("tests/data/imaging/psf-distillation/beads/manifest.json", "measurement"),
    ),
)
def test_private_measurements_publish_only_neutral_ids(
    filename: str, field: str
) -> None:
    """Keep private acquisition names out of published measurement metadata."""
    record = json.loads(Path(filename).read_text("utf-8"))[field]
    assert re.fullmatch(r"measurement-\d{3}", record["measurement_id"])
    assert (
        not {"filename", "source_file", "source_path", "acquisition_date"}
        & record.keys()
    )


def test_text_provenance_preserves_content_across_line_endings(tmp_path) -> None:
    """Ignore checkout line endings while detecting changed source content."""
    capture = runpy.run_path("scripts/capture_bead_notebook.py")
    unix = tmp_path / "unix.py"
    windows = tmp_path / "windows.py"
    unix.write_bytes(b"value = 1\nprint(value)\n")
    windows.write_bytes(b"value = 1\r\nprint(value)\r\n")
    assert capture["text_sha256"](unix) == capture["text_sha256"](windows)
    windows.write_bytes(b"value = 2\r\nprint(value)\r\n")
    assert capture["text_sha256"](unix) != capture["text_sha256"](windows)


def test_bead_notebook_matches_frozen_preprocessing() -> None:
    """Preserve raw bead extraction, registration and calibrated PSF inputs."""
    root = Path(__file__).resolve().parents[2]
    capture = runpy.run_path(str(root / "scripts/capture_bead_notebook.py"))
    reference = root / "tests/data/imaging/psf-distillation/beads"
    manifest = json.loads((reference / "manifest.json").read_text("utf-8"))
    notebook_path = root / manifest["notebook"]
    # Undo only the CSV substitutions when checking the captured source hash.
    source = (
        notebook_path.read_text("utf-8")
        .replace(
            '    "from brighteyes.imaging.s2ism.psf_estimator import GridFinder\\n",\n',
            '    "from brighteyes.imaging.s2ism.psf_estimator import GridFinder\\n",\n'
            '    "from brighteyes.spectroscopy.tools.csv2array import array2csv, csv2array\\n",\n',
        )
        .replace(
            "#np.savetxt('clicked_points.csv', clicked_points, delimiter='\\\\t')",
            "#array2csv(clicked_points, 'clicked_points.csv')",
        )
        .replace(
            "clicked_points = np.loadtxt('clicked_points.csv', delimiter='\\\\t')",
            "clicked_points = csv2array('clicked_points.csv')",
        )
    )
    assert hashlib.sha256(source.encode()).hexdigest() == manifest["notebook_sha256"]
    notebook = json.loads(notebook_path.read_text("utf-8"))
    with np.load(datasets.fetch("beads-measured-small"), allow_pickle=False) as data:
        inputs = {key: data[key] for key in data.files}
    context = capture["namespace"](inputs)
    for stage in manifest["stages"]:
        cell = notebook["cells"][stage["cell"]]
        assert capture["numerical_source"](cell) == stage["executed_source"]
        assert (
            hashlib.sha256("".join(cell["source"]).encode()).hexdigest()
            == stage["source_sha256"]
        )
        path = reference / stage["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == stage["sha256"]
        exec(  # noqa: S102
            compile(stage["executed_source"], f"bead-cell-{stage['cell']}", "exec"),
            context,
        )
        with np.load(path, allow_pickle=False) as expected:
            for key in stage["variables"]:
                actual = context[key]
                if isinstance(actual, torch.Tensor):
                    actual = actual.detach().cpu().numpy()
                np.testing.assert_array_equal(actual, expected[key], strict=True)
        if stage["cell"] == 13:
            np.testing.assert_array_equal(
                inputs["raw_bead_crops"].sum(axis=-2), context["single_beads"]
            )


def test_full_capture_saves_fit_and_masked_wavefronts(
    tmp_path, monkeypatch, reference_arrays
) -> None:
    """Save completed starts and post-fit outputs with the pupil mask intact."""
    root = Path(__file__).resolve().parents[2]
    capture = runpy.run_path(str(root / "scripts/capture_bead_notebook.py"))
    original_fit = distiller.fit_psf

    def short_fit(*args, **kwargs):
        assert kwargs["num_starts"] == 100
        assert kwargs["max_iter"] == 1000
        kwargs["max_iter"] = 2
        return original_fit(*args, **kwargs, verbose=False)

    monkeypatch.setattr(distiller, "fit_psf", short_fit)
    output = tmp_path / "capture"
    capture["full_fit"](output)
    manifest = json.loads((output / "manifest.json").read_text("utf-8"))
    with np.load(output / "full-fit.npz", allow_pickle=False) as fit:
        with np.load(
            root / "tests/data/imaging/psf-distillation/beads/full-fit/full-fit.npz",
            allow_pickle=False,
        ) as frozen:
            reference_arrays(
                fit["history"], frozen["history"][:, :2], "bead-fit/loss-prefix"
            )
        for index in range(100):
            with np.load(
                output / f"start-{index:03d}.npz", allow_pickle=False
            ) as start:
                np.testing.assert_array_equal(start["history"], fit["history"][index])
        assert np.isfinite(fit["result"]).all()
        assert np.isfinite(fit["history"]).all()
    with np.load(output / "wavefronts.npz", allow_pickle=False) as wavefronts:
        coordinates = np.linspace(-1, 1, 101)
        outside_pupil = np.hypot(coordinates[:, None], coordinates[None, :]) >= 1
        for wavefront in wavefronts.values():
            np.testing.assert_array_equal(np.isnan(wavefront), outside_pupil)
            assert np.isfinite(wavefront[~outside_pupil]).all()
    for stage in manifest["post_fit_stages"]:
        path = output / stage["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == stage["sha256"]


def test_bead_full_fit_replays_reviewed_notebook_outputs() -> None:
    """Preserve winner selection and replay post-fit outputs from coefficients."""
    root = Path(__file__).resolve().parents[2]
    capture = runpy.run_path(str(root / "scripts/capture_bead_notebook.py"))
    reference = root / "tests/data/imaging/psf-distillation/beads"
    output = reference / "full-fit"
    preparation = json.loads((reference / "manifest.json").read_text("utf-8"))
    manifest = json.loads((output / "manifest.json").read_text("utf-8"))
    assert (
        capture["text_sha256"](reference / "manifest.json")
        == manifest["input_manifest_sha256"]
    )
    assert manifest["executed_source"] == preparation["full_fit_source"]
    starts_path = output / "starting-points.npz"
    assert (
        hashlib.sha256(starts_path.read_bytes()).hexdigest()
        == manifest["starting_points_sha256"]
    )
    for stage in [manifest, *manifest["post_fit_stages"]]:
        path = output / stage["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == stage["sha256"]
    with np.load(output / "full-fit.npz", allow_pickle=False) as frozen:
        coefficients = frozen["result"]
        history = frozen["history"]
    assert history.shape == (100, 1000)
    assert np.isfinite(history).all()
    assert np.isfinite(coefficients).all()
    for index, losses in enumerate(history):
        with np.load(output / f"start-{index:03d}.npz", allow_pickle=False) as start:
            np.testing.assert_array_equal(start["history"], losses, strict=True)
            assert np.isfinite(start["coefficients"]).all()
            if index == history[:, -1].argmin():
                np.testing.assert_array_equal(
                    distiller.structure_result(
                        torch.from_numpy(start["coefficients"]), 15, True
                    ).numpy(),
                    coefficients,
                    strict=True,
                )
    with np.load(datasets.fetch("beads-measured-small"), allow_pickle=False) as data:
        context = capture["namespace"](
            {key: data[key] for key in ("summed_image", "clicked_points")}
        )
    for stage in preparation["stages"]:
        exec(  # noqa: S102
            compile(stage["executed_source"], f"bead-cell-{stage['cell']}", "exec"),
            context,
        )
    context["result"] = torch.from_numpy(coefficients)
    for stage, original in zip(
        manifest["post_fit_stages"], preparation["post_fit_stages"], strict=True
    ):
        assert stage["executed_source"] == original["executed_source"]
        exec(  # noqa: S102
            compile(stage["executed_source"], f"bead-cell-{stage['cell']}", "exec"),
            context,
        )
        with np.load(output / stage["file"], allow_pickle=False) as expected:
            for name in stage["variables"]:
                actual = context[name]
                if isinstance(actual, torch.Tensor):
                    actual = actual.detach().cpu().numpy()
                np.testing.assert_array_equal(actual, expected[name], strict=True)
