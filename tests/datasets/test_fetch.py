"""Check packaged data and remote retrieval through pooch."""

import hashlib
from pathlib import Path

import pooch
import pytest

from brighteyes import datasets
from brighteyes.datasets._fetch import catalog as shipped_catalog

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
def retrieve(tmp_path, monkeypatch):
    """Record pooch.retrieve calls and return paths without downloading."""
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        processor = kwargs.get("processor")
        if processor is None:
            return str(tmp_path / kwargs["fname"])
        return [
            str(tmp_path / "archive.zip.unzip" / member) for member in processor.members
        ]

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
        }
    ]
    assert result == tmp_path / "measurement.bin"
    assert isinstance(result, Path)


def test_fetch_member_extracts_only_that_member(storage, retrieve, tmp_path):
    """Retrieve the shared archive and extract only the requested member."""
    _, catalog = storage
    catalog["archive"] = {"url": URL, "filename": "archive.zip", "checksum": CHECKSUM}
    catalog["first"] = {"archive": "archive", "member": "first.h5"}
    catalog["second"] = {"archive": "archive", "member": "second.h5"}
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
    """Point every shipped member entry at a remote archive entry."""
    members = {name: e for name, e in shipped_catalog.items() if "archive" in e}
    assert members
    for entry in members.values():
        archive = shipped_catalog[entry["archive"]]
        assert {"url", "filename", "checksum"} <= archive.keys()
