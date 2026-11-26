"""Check that distributions contain imaging code and packaged measurements."""

import hashlib
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path


def test_distributions_include_imaging_without_deferred_modules(tmp_path) -> None:
    """Ship imaging inputs and MCS support without spectroscopy or TTM code."""
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        ["uv", "build", "--out-dir", str(tmp_path)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("*.whl"))
    assert wheel.name.endswith("-py3-none-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert not any(
            name.startswith(("brighteyes/spectroscopy/", "brighteyes/ttm/"))
            or name.endswith((".pyd", ".so"))
            for name in names
        )
        for source in (root / "src/brighteyes").rglob("*.py"):
            name = source.relative_to(root / "src").as_posix()
            assert archive.read(name) == source.read_bytes()
        catalog = json.loads(archive.read("brighteyes/datasets/_catalog.json"))
        for entry in catalog.values():
            if "path" in entry:
                data = archive.read("brighteyes/datasets/data/" + entry["path"])
                assert len(data) == entry["size"]
                assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    with tarfile.open(next(tmp_path.glob("*.tar.gz"))) as archive:
        names = [name.partition("/")[2] for name in archive.getnames()]
        assert not any(
            name.startswith(
                ("src/brighteyes/spectroscopy/", "src/brighteyes/ttm/", "rust/")
            )
            for name in names
        )
        assert "src/brighteyes/imaging/analysis/APR_lib.py" in names
        assert "src/brighteyes/imaging/utils/_frc.py" in names
        assert "src/brighteyes/imaging/plotting/_images.py" in names
        assert "src/brighteyes/imaging/psf/sim/_psf.py" in names
        assert "src/brighteyes/mcs/reader/reader.py" in names
