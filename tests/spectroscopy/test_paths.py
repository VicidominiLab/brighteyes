"""Check directory helpers without machine-specific locations."""

from importlib import import_module
from pathlib import Path


def test_directory_helpers_use_the_requested_location(tmp_path, monkeypatch):
    """Change to a supplied directory and find its files by default."""
    directory = tmp_path / "data"
    directory.mkdir()
    sample = directory / "measurement.bin"
    sample.write_bytes(b"sample")
    monkeypatch.chdir(tmp_path)
    import_module("brighteyes.spectroscopy.tools.goto").goto("data")
    assert Path.cwd() == directory
    assert import_module("brighteyes.spectroscopy.tools.list_files").list_files() == [
        str(sample)
    ]
