# Origin: VicidominiLab/BrightEyes-FLISM, src/brighteyes_flism/flism_data.py at 85f0d96.
"""Legacy FLIM data containers."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np
from numpy.typing import NDArray

from brighteyes.mcs.reader import reader_legacy

from .tools_phasor import calculate_phasor

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["FlimData"]

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


class FlimData:
    """Container for global FLIM histograms and phasor-based calibration helpers."""

    SPAD_DATA_KEYS = ("raw/spad", "data")
    AUX_DATA_KEYS = ("raw/aux", "data_channels_extra")
    PRIMARY_DATA_KEYS = SPAD_DATA_KEYS
    EXTRA_DATA_KEYS = AUX_DATA_KEYS

    def __init__(
        self,
        data_path: str | None = None,
        data_path_irf: str | None = None,
        freq_exc: float = 41.48e6,
        correction_coeff: complex
        | Sequence[float]
        | np.ndarray[Any, Any]
        | None = None,
        step_size: int | None = None,
        sub_image_dim: int = 100,
        pre_filter: str | float | None = None,
    ) -> None:
        """Load data, aggregate histograms, and initialize phasor state."""
        self.phasor_laser: complex = 0.0j
        self.phasor_laser_irf: complex = 0.0j
        self.freq_exc: float = freq_exc

        self.sub_image_dim: int = sub_image_dim
        self.data: h5py.File | None = None
        self.data_irf: h5py.File | None = None
        self.sliced_data: FloatArray | None = None
        self.bin_number: int = 81
        self.channel_number: int = 25

        if data_path is not None:
            with h5py.File(data_path, "r") as hf:
                img = self._first_dataset(hf, self.SPAD_DATA_KEYS, "SPAD data")
                self.bin_number = img.shape[-2]
                self.channel_number = img.shape[-1]

        self.data_hist: FloatArray = np.zeros((self.bin_number, self.channel_number))
        self.data_hist_irf: FloatArray = np.zeros(
            (self.bin_number, self.channel_number)
        )
        self.data_laser_hist: FloatArray = np.zeros(self.bin_number)
        self.data_laser_hist_irf: FloatArray = np.zeros(self.bin_number)
        self.data_laser_hist_non_sliced: FloatArray | None = None
        self.metadata: Any = None
        self.metadata_irf: Any = None
        self.calib_ref: Any = None
        self.phasors: ComplexArray | None = None
        self.phasors_global: ComplexArray | None = None
        self.phasors_global_irf: ComplexArray | None = None
        self.shift_term: int = 0
        self.correction_coeff: complex | ComplexArray | None = None
        self.step_size: int | None = step_size

        if data_path is not None:
            self.load_data_irf(data_path_irf)
            self.load_data(data_path)
            if pre_filter is not None:
                if isinstance(pre_filter, str):
                    if pre_filter == "normalize":
                        self.data_hist = (
                            self.data_hist - np.nanmin(self.data_hist.T, axis=1)
                        ) / (
                            np.nanmax(self.data_hist.T, axis=1)
                            - np.nanmin(self.data_hist.T, axis=1)
                        )
                        self.data_hist_irf = (
                            self.data_hist_irf - np.nanmin(self.data_hist_irf.T, axis=1)
                        ) / (
                            np.nanmax(self.data_hist_irf.T, axis=1)
                            - np.nanmin(self.data_hist_irf.T, axis=1)
                        )
                elif isinstance(pre_filter, (int, float)):
                    threshold = pre_filter
                    nnn = self.data_hist / np.max(self.data_hist, axis=0)
                    nnn[nnn < threshold] = 1.0 / np.max(
                        self.data_hist[nnn < threshold], axis=0
                    )
                    self.data_hist = nnn
                    mmm = self.data_hist_irf / np.max(self.data_hist_irf, axis=0)
                    mmm[mmm < threshold] = 1.0 / np.max(
                        self.data_hist_irf[mmm < threshold], axis=0
                    )
                    self.data_hist_irf = mmm
                else:
                    warnings.warn(
                        f"Ignoring unsupported pre_filter value {pre_filter!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            self.calculate_phasor_global_irf()
            self.calculate_phasor_global()
            self.calculate_phasor_laser()
            self.calculate_phasor_laser_irf()

        if correction_coeff is not None:
            if isinstance(correction_coeff, (list, np.ndarray)):
                self.correction_coeff = correction_coeff[0] + 1j * correction_coeff[1]
            if isinstance(correction_coeff, complex):
                self.correction_coeff = correction_coeff

    @staticmethod
    def _first_dataset(
        handle: h5py.File | h5py.Group, candidates: Sequence[str], label: str
    ) -> h5py.Dataset:
        """Find the first matching HDF5 dataset."""
        for key in candidates:
            if key in handle and isinstance(handle[key], h5py.Dataset):
                return handle[key]
        tried = ", ".join(candidates)
        raise KeyError(f"could not find {label} dataset; tried {tried}")

    @staticmethod
    def _sum_to_time_channel(dataset: h5py.Dataset) -> FloatArray:
        """Sum all axes except time and channel."""
        if dataset.ndim < 2:
            raise ValueError(
                f"dataset must have time and channel as final axes, got {dataset.shape}"
            )
        axes = tuple(range(dataset.ndim - 2))
        return cast(
            FloatArray, np.asarray(dataset[...], dtype=np.float64).sum(axis=axes)
        )

    @classmethod
    def _load_aux_laser_histogram(
        cls, handle: h5py.File | h5py.Group, bin_number: int
    ) -> FloatArray:
        """Load and sum the auxiliary laser histogram."""
        try:
            aux_data = cls._first_dataset(handle, cls.AUX_DATA_KEYS, "aux data")
        except KeyError:
            return np.zeros(bin_number, dtype=np.float64)
        if aux_data.shape[-1] <= 1:
            return np.zeros(int(aux_data.shape[-2]), dtype=np.float64)
        aux_hist = cls._sum_to_time_channel(aux_data)
        return aux_hist[:, 1]

    _load_extra_laser_histogram = _load_aux_laser_histogram

    def load_data(self, data_path: str) -> None:
        """Load sample histograms and metadata."""
        self.data = h5py.File(data_path)
        self.metadata = reader_legacy.metadata_load(data_path)
        image = self._first_dataset(self.data, self.SPAD_DATA_KEYS, "SPAD data")
        self.data_hist = self._sum_to_time_channel(image)
        self.data_laser_hist = self._load_aux_laser_histogram(
            self.data, self.data_hist.shape[0]
        )

    def load_data_irf(self, data_path_irf: str | None) -> None:
        """Load reference histograms and metadata."""
        self.data_irf = h5py.File(data_path_irf)
        self.metadata_irf = reader_legacy.metadata_load(data_path_irf)
        image_irf = self._first_dataset(
            self.data_irf, self.SPAD_DATA_KEYS, "SPAD IRF data"
        )
        self.data_hist_irf = self._sum_to_time_channel(image_irf)
        self.data_laser_hist_irf = self._load_aux_laser_histogram(
            self.data_irf,
            self.data_hist_irf.shape[0],
        )

    def calculate_phasor_global(self) -> ComplexArray:
        """Calculate global phasors for the sample data."""
        phasors = cast(ComplexArray, calculate_phasor(data_hist=self.data_hist))
        self.phasors_global = phasors
        return phasors

    def calculate_phasor_global_irf(self) -> ComplexArray:
        """Calculate global phasors for the reference data."""
        phasors = cast(ComplexArray, calculate_phasor(data_hist=self.data_hist_irf))
        self.phasors_global_irf = phasors
        return phasors

    def calculate_phasor_laser(self) -> complex:
        """Calculate the sample laser phasor."""
        self.phasor_laser = cast(complex, calculate_phasor(self.data_laser_hist))
        return self.phasor_laser

    def calculate_phasor_laser_irf(self) -> complex:
        """Calculate the reference laser phasor."""
        self.phasor_laser_irf = cast(
            complex, calculate_phasor(self.data_laser_hist_irf)
        )
        return self.phasor_laser_irf

    def calculate_correction_coeff(
        self, dfd_freq: float = 41.48e6, tau: float = 10e-9
    ) -> ComplexArray:
        """Calculate and store the phasor correction coefficient."""
        dfd_time = 1.0 / dfd_freq

        phasors_global = cast(ComplexArray, self.phasors_global)
        ppp = phasors_global / abs(phasors_global)

        theta = 2 * np.pi * tau / dfd_time
        DeltaM = np.abs(phasors_global) / np.abs(np.cos(theta))

        DeltaPhi = np.exp(1j * theta)
        coeff = cast(ComplexArray, 1.0 / (ppp * DeltaM * DeltaPhi))

        self.correction_coeff = coeff

        return coeff

    def phasor_global_corrected(
        self, coeff: complex = 1.0, laser_correction: bool = True
    ) -> ComplexArray:
        """Apply correction to the sample's global phasors."""
        if laser_correction:
            return cast(
                ComplexArray,
                cast(ComplexArray, self.phasors_global)
                * coeff
                * np.exp(1j * (-np.angle(self.phasor_laser))),
            )
        else:
            return cast(ComplexArray, self.phasors_global) * coeff

    def phasor_global_corrected_irf(
        self, coeff: complex = 1.0, laser_correction: bool = True
    ) -> ComplexArray:
        """Apply correction to the reference's global phasors."""
        if laser_correction:
            return cast(
                ComplexArray,
                cast(ComplexArray, self.phasors_global_irf)
                * coeff
                * np.exp(1j * (-np.angle(self.phasor_laser))),
            )
        else:
            return cast(ComplexArray, self.phasors_global_irf) * coeff
