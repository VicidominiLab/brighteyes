"""Check software provenance in exported MCS measurements."""

from importlib.metadata import PackageNotFoundError, version

import h5py
import numpy as np

from brighteyes.mcs.dataprep import H5OutputProduct, write_h5_output_run


def test_output_records_monorepo_software_version(tmp_path, monkeypatch):
    """Record the installed monorepo version without the standalone package."""
    source_path = tmp_path / "measurement.h5"
    with h5py.File(source_path, "w") as handle:
        handle.attrs["data_format_version"] = "0.0.6"
        handle.create_group("raw").create_dataset(
            "spad", data=np.ones((1, 1, 1, 1, 2, 2), dtype=np.uint16)
        )
        handle.create_group("calibration")

    def installed_version(name):
        if name != "brighteyes":
            raise PackageNotFoundError(name)
        return version(name)

    monkeypatch.setattr(
        "brighteyes.mcs.dataprep.h5_output_writers.version", installed_version
    )
    write_h5_output_run(
        source_path, "result", [H5OutputProduct("image", np.ones((2, 3)))]
    )
    with h5py.File(source_path) as handle:
        attrs = handle["output/result"].attrs
        assert attrs["software_name"] == "brighteyes"
        assert attrs["software_version"] == version("brighteyes")
