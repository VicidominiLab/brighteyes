"""Keep numerical comparisons strict across the MCS package rename."""

import h5py
import numpy as np
import pytest
from _compare import assert_same
from _h5_dump import h5_contents


@pytest.mark.parametrize("change", ("image", "software_name", "software_version"))
def test_package_rename_preserves_other_comparisons(tmp_path, change):
    """Accept only the package rename and reject changed data or other metadata."""
    old = tmp_path / "old.h5"
    new = tmp_path / "new.h5"
    for path, package in ((old, "brighteyes_mcs_dataprep"), (new, "brighteyes")):
        with h5py.File(path, "w") as handle:
            handle.attrs["software_name"] = package
            handle.attrs["created_by_package"] = package
            handle.attrs["software_version"] = "0.1.0"
            handle.create_dataset("image", data=np.arange(4))
    assert_same(h5_contents(old, tmp_path), h5_contents(new, tmp_path))
    with h5py.File(new, "a") as handle:
        if change == "image":
            handle["image"][0] = 10
        else:
            handle.attrs[change] = "different"
    with pytest.raises(AssertionError):
        assert_same(h5_contents(old, tmp_path), h5_contents(new, tmp_path))
