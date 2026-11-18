"""Check HDF5 conversion with the TTM dependency set."""

import numpy as np
import pandas as pd

from brighteyes.ttm import ttp


def test_convert_timing_frame_to_hdf5(tmp_path):
    """Preserve cumulative timing across counter rollover in an HDF5 file."""
    steps = np.arange(65530, 65730, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "step": (steps % 65536).astype(np.uint16),
            "scan_enable": np.ones(200, dtype=np.uint8),
            "line_enable": np.zeros(200, dtype=np.uint8),
            "pixel_enable": np.ones(200, dtype=np.uint8),
            "valid_tdc_L": np.zeros(200, dtype=np.uint8),
        }
    )
    path = tmp_path / "measurement.h5"
    ttp.convertFromPandasDataFrame(
        frame,
        filenameOutputHDF5=str(path),
        list_of_channels=[],
        ignorePixelLineFrame=True,
    )
    saved = pd.read_hdf(path, key="main")
    np.testing.assert_array_equal(saved["cumulative_step"].to_numpy(), steps)
    np.testing.assert_array_equal(
        saved["total_photon"].to_numpy(), np.zeros(200, dtype=np.uint8)
    )
