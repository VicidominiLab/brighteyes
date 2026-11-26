"""Resolve packaged inputs and retrieve published measurements."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pooch

data_dir = Path(__file__).with_name("data")
catalog = json.loads(Path(__file__).with_name("_catalog.json").read_text("utf-8"))


def verify(path: Path, size: int, checksum: str) -> None:
    """Raise ValueError when a file's size or checksum differs."""
    if path.stat().st_size != size:
        raise ValueError(f"dataset size mismatch: {path.name}")
    algorithm, expected = checksum.split(":", 1)
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"dataset checksum mismatch: {path.name}")


def fetch(name: str) -> Path:
    """Return a verified example input, downloading published files if needed.

    Small inputs are packaged locally. Published measurements download once
    into the user cache and are checked against their hash. A name that refers
    to a file inside a published archive extracts only that file.

    Raises
    ------
    ValueError
        If the name is unknown or a file fails its integrity check.
    requests.RequestException
        If a download fails.
    OSError
        If the input or cache cannot be read or written.
    """
    if name not in catalog:
        raise ValueError(f"unknown dataset: {name}")
    entry = catalog[name]
    if "path" in entry:
        path = data_dir / entry["path"]
        verify(path, entry["size"], "sha256:" + entry["sha256"])
        return path

    if "archive" not in entry:
        return Path(retrieve(entry))
    extracted = retrieve(
        catalog[entry["archive"]], processor=pooch.Unzip(members=[entry["member"]])
    )
    return Path(extracted[0])


def retrieve(entry: dict[str, str], **kwargs: Any) -> Any:
    """Download a published file into the user cache unless it is already there."""
    return pooch.retrieve(
        url=entry["url"],
        known_hash=entry["checksum"],
        fname=entry["filename"],
        path=pooch.os_cache("brighteyes"),
        progressbar=True,
        **kwargs,
    )
