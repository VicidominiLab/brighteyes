"""Check compatibility with saved fluorescence spectroscopy data."""

import builtins
import pickle
from importlib import import_module

import numpy as np
import pytest


@pytest.mark.parametrize("legacy", (False, True))
def test_arrival_times_load_without_standalone_package(tmp_path, monkeypatch, legacy):
    """Read current and legacy arrival-time files without the old package."""
    arrival_type = import_module(
        "brighteyes.spectroscopy.fcs.binfile2arrivaltimes"
    ).ArrivalTimesObject
    arrival_times = arrival_type()
    arrival_times.det0 = np.array([3, 7, 11], dtype=np.int64)
    arrival_times.dwellTime = 1e-6
    content = pickle.dumps(arrival_times, protocol=0)
    if legacy:
        content = content.replace(
            b"brighteyes.spectroscopy.fcs.binfile2arrivaltimes",
            b"brighteyes_ffs.fcs.binfile2arrivaltimes",
        )
        assert b"brighteyes_ffs.fcs.binfile2arrivaltimes" in content
    path = tmp_path / "arrival_times.pickle"
    path.write_bytes(content)
    original_import = builtins.__import__

    def import_without_standalone(name, *args, **kwargs):
        if name == "brighteyes_ffs" or name.startswith("brighteyes_ffs."):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_standalone)
    actual = import_module("brighteyes.spectroscopy.tools.savevar").openvar(str(path))
    assert isinstance(actual, arrival_type)
    np.testing.assert_array_equal(actual.det0, arrival_times.det0, strict=True)
    assert actual.dwellTime == arrival_times.dwellTime
