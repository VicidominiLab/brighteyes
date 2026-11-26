"""Check packaged data and remote retrieval through pooch."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import pooch
import pytest

from brighteyes import datasets

URL = "https://example.invalid/measurement.bin"
CHECKSUM = "md5:" + "0" * 32


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """Keep packaged inputs and the catalog inside this test's directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    catalog = {}
    monkeypatch.setattr("brighteyes.datasets._fetch.data_dir", data_dir)
    monkeypatch.setattr("brighteyes.datasets._fetch.catalog", catalog)
    return data_dir, catalog


@pytest.fixture
def member_sizes():
    """Map each archive member to the size the fake extraction writes."""
    return {}


@pytest.fixture
def retrieve(tmp_path, monkeypatch, member_sizes):
    """Record pooch.retrieve calls and write extracted members without downloading."""
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        processor = kwargs.get("processor")
        if processor is None:
            return str(tmp_path / kwargs["fname"])
        extracted = []
        for member in processor.members:
            if member in member_sizes:
                path = tmp_path / "archive.zip.unzip" / member
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"x" * member_sizes[member])
                extracted.append(str(path))
        return extracted

    monkeypatch.setattr("pooch.retrieve", fake)
    return calls


def test_fetch_bundled_without_network(storage, retrieve):
    """Return a verified packaged file without retrieving anything."""
    data_dir, catalog = storage
    content = b"real sample"
    (data_dir / "sample.bin").write_bytes(content)
    catalog["sample"] = {
        "path": "sample.bin",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    assert datasets.fetch("sample") == data_dir / "sample.bin"
    assert retrieve == []


def test_fetch_rejects_modified_packaged_data(storage, retrieve):
    """Report corrupted packaged input instead of returning it."""
    data_dir, catalog = storage
    (data_dir / "sample.bin").write_bytes(b"different")
    catalog["sample"] = {
        "path": "sample.bin",
        "size": 9,
        "sha256": hashlib.sha256(b"reference").hexdigest(),
    }
    with pytest.raises(ValueError, match="checksum"):
        datasets.fetch("sample")
    assert retrieve == []


def test_fetch_rejects_unknown_name(storage, retrieve):
    """Reject unknown names without retrieving anything."""
    with pytest.raises(ValueError, match="unknown dataset"):
        datasets.fetch("missing")
    assert retrieve == []


def test_fetch_remote_entry_uses_pooch(storage, retrieve, tmp_path):
    """Retrieve a remote entry with its URL, hash and file name into the cache."""
    _, catalog = storage
    catalog["measurement"] = {
        "url": URL,
        "filename": "measurement.bin",
        "checksum": CHECKSUM,
    }
    result = datasets.fetch("measurement")
    assert retrieve == [
        {
            "url": URL,
            "known_hash": CHECKSUM,
            "fname": "measurement.bin",
            "path": pooch.os_cache("brighteyes"),
            "progressbar": True,
            "processor": None,
        }
    ]
    assert result == tmp_path / "measurement.bin"
    assert isinstance(result, Path)


def test_fetch_member_extracts_only_that_member(
    storage, retrieve, member_sizes, tmp_path
):
    """Retrieve the shared archive and extract only the requested member."""
    _, catalog = storage
    catalog["archive"] = {"url": URL, "filename": "archive.zip", "checksum": CHECKSUM}
    catalog["first"] = {"archive": "archive", "member": "first.h5", "size": 4}
    catalog["second"] = {"archive": "archive", "member": "second.h5", "size": 4}
    member_sizes.update({"first.h5": 4, "second.h5": 4})
    first = datasets.fetch("first")
    datasets.fetch("second")
    for call, member in zip(retrieve, ("first.h5", "second.h5"), strict=True):
        assert (call["url"], call["known_hash"], call["fname"]) == (
            URL,
            CHECKSUM,
            "archive.zip",
        )
        assert isinstance(call["processor"], pooch.Unzip)
        assert call["processor"].members == [member]
    assert first == tmp_path / "archive.zip.unzip" / "first.h5"
    assert isinstance(first, Path)


def test_catalog_member_entries_name_remote_archives():
    """Point every shipped member entry at a remote archive and give its size."""
    catalog = json.loads(
        files("brighteyes.datasets").joinpath("_catalog.json").read_text("utf-8")
    )
    members = [entry for entry in catalog.values() if "archive" in entry]
    assert members
    for entry in members:
        assert {"url", "filename", "checksum"} <= catalog[entry["archive"]].keys()
        assert isinstance(entry["size"], int)


def test_fetch_rejects_truncated_member(storage, retrieve, member_sizes, tmp_path):
    """Extract a short member again, then remove it and raise if it stays short."""
    _, catalog = storage
    catalog["archive"] = {"url": URL, "filename": "archive.zip", "checksum": CHECKSUM}
    catalog["first"] = {"archive": "archive", "member": "first.h5", "size": 4}
    member_sizes["first.h5"] = 2
    with pytest.raises(ValueError, match="size mismatch"):
        datasets.fetch("first")
    assert len(retrieve) == 2
    assert not (tmp_path / "archive.zip.unzip" / "first.h5").exists()


def test_fetch_rejects_member_missing_from_archive(storage, retrieve):
    """Raise ValueError when the archive has no file with the member's name."""
    _, catalog = storage
    catalog["archive"] = {"url": URL, "filename": "archive.zip", "checksum": CHECKSUM}
    catalog["first"] = {"archive": "archive", "member": "first.h5", "size": 4}
    with pytest.raises(ValueError, match="not in archive"):
        datasets.fetch("first")
