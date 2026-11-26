# Origin: VicidominiLab/s2ISM, src/s2ism/dataio.py at 10bb5fc.
"""BrightEyes-MCS file loading helpers for s2ISM workflows."""

from __future__ import annotations

from importlib import import_module
from os import PathLike
from types import ModuleType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Any

    import h5py
    from numpy.typing import NDArray

    from brighteyes.mcs.reader import DatasetInfo, MCSMetadata, Signal

Pathish = str | PathLike[str]


def _mcs_reader() -> ModuleType:
    try:
        return import_module("brighteyes.mcs.reader")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "s2ism MCS loading requires 'brighteyes-mcs-reader'. "
            "Install it with 'pip install brighteyes-mcs-reader'."
        ) from exc


def is_mcs_file(path: Pathish) -> bool:
    """Return whether *path* is a current-schema BrightEyes-MCS HDF5 file."""
    return cast("bool", _mcs_reader().is_brighteyes_mcs_h5(path))


def list_mcs_datasets(path: Pathish) -> list[DatasetInfo]:
    """Return readable datasets in a BrightEyes-MCS HDF5 file."""
    return cast("list[DatasetInfo]", _mcs_reader().list_datasets(path))


def load_mcs(
    path: Pathish,
    key: str = "data",
    data_format: str = "numpy",
) -> tuple[NDArray[Any] | h5py.Dataset, MCSMetadata]:
    """Load BrightEyes-MCS data using the legacy-compatible reader.

    This is the drop-in replacement for the old
    ``brighteyes.mcs.dataprep.reader_legacy.load`` call used by the original
    examples. It returns ``(data, metadata)``.
    """
    return cast(
        "tuple[NDArray[Any] | h5py.Dataset, MCSMetadata]",
        _mcs_reader().reader_legacy.load(path, key=key, data_format=data_format),
    )


def read_mcs_signal(
    path: Pathish,
    *,
    dataset: str | None = None,
    time: int = 0,
    depth: int = 0,
    channel: int | str | None = 0,
) -> Signal:
    """Read one image or trace from a current-schema BrightEyes-MCS HDF5 file."""
    return cast(
        "Signal",
        _mcs_reader().read_signal(
            path,
            dataset=dataset,
            time=time,
            depth=depth,
            channel=channel,
        ),
    )
