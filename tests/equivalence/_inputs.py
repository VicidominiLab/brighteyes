from pathlib import Path

import h5py
import numpy as np
import torch

SIZE = 32
CHANNELS = 25


def ism_dataset() -> np.ndarray:
    """Return a small `(x, y, channels)` array of photon counts."""
    return (
        np.random.default_rng(0)
        .poisson(10.0, size=(SIZE, SIZE, CHANNELS))
        .astype(np.float64)
    )


def ism_psf() -> np.ndarray:
    """Return a `(x, y, channels)` array holding one shifted Gaussian per channel."""
    axis = np.arange(SIZE) - SIZE / 2
    x, y = np.meshgrid(axis, axis, indexing="ij")
    shifts = np.random.default_rng(0).uniform(-3, 3, size=(CHANNELS, 2))
    psf = np.stack(
        [np.exp(-((x - dx) ** 2 + (y - dy) ** 2) / (2 * 2.0**2)) for dx, dy in shifts],
        axis=-1,
    )
    return psf / psf.sum(axis=(0, 1), keepdims=True)


def decay_histogram() -> np.ndarray:
    """Return a noisy single-exponential photon arrival histogram."""
    time = np.arange(256)
    return (
        np.random.default_rng(0)
        .poisson(1000.0 * np.exp(-time / 40.0))
        .astype(np.float64)
    )


def ism_series() -> np.ndarray:
    """Return a `(x, y, time, channels)` array of photon counts of a smooth sample."""
    rng = np.random.default_rng(0)
    axis = np.arange(SIZE)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    sample = sum(
        np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 3.0**2))
        for cx, cy in rng.uniform(6, SIZE - 6, size=(4, 2))
    )
    detector = np.arange(CHANNELS)
    radius2 = (detector // 5 - 2) ** 2 + (detector % 5 - 2) ** 2
    weights = np.exp(-radius2 / (2 * 1.2**2)) * rng.uniform(0.9, 1.1, size=CHANNELS)
    mean = 400.0 * sample[:, :, None, None] * weights[None, None, None, :] + 2.0
    return rng.poisson(np.broadcast_to(mean, (SIZE, SIZE, 4, CHANNELS))).astype(
        np.float64
    )


def _flim_decays(seed: int, tau: float, lag: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bins = 64
    time = np.arange(bins)[:, None]
    lags = lag * (np.arange(CHANNELS) % 5)[None, :]
    decay = np.exp(-np.mod(time - lags, bins) / tau)
    amplitude = rng.uniform(5.0, 30.0, size=(8, 8))
    return rng.poisson(amplitude[:, :, None, None] * decay[None, None]).astype(
        np.float64
    )


def flim_stack() -> np.ndarray:
    """Return an `(x, y, time, channels)` array of photon counts with a decay per pixel."""
    return _flim_decays(0, tau=8.0, lag=1.0)


def flim_reference_stack() -> np.ndarray:
    """Return a `flim_stack`-shaped array of a reference sample with another lifetime and delays."""
    return _flim_decays(0, tau=5.0, lag=2.0)


def bead_image() -> np.ndarray:
    """Return a `(x, y, channels)` array holding five Gaussian beads on a dark background."""
    axis = np.arange(64)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    beads = np.array([[16, 16], [16, 44], [32, 30], [46, 14], [48, 48]])
    spot = sum(
        np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2.5**2)) for cx, cy in beads
    )
    weights = np.random.default_rng(0).uniform(0.5, 1.0, size=4)
    return np.round(200 * spot[:, :, None] * weights[None, None, :])


def mcs_h5_file(path: Path) -> Path:
    """Write a small current-schema MCS file with `/raw/spad` and `/raw/aux`, and return its path."""
    rng = np.random.default_rng(0)
    shape = (1, 1, 4, 5, 8)
    spad = rng.poisson(3.0, size=(*shape, 5)).astype(np.uint16)
    aux = rng.poisson(2.0, size=(*shape, 2)).astype(np.uint16)
    time_bin_ns = 0.5
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            data_format_version="0.0.6",
            schema_name="brighteyes_mcs_file",
            default="/raw/spad",
            data_path="/raw",
            metadata_path="/raw/metadata",
            axes_path="/raw/axes",
            legacy_path="/raw/legacy",
            calibration_path="/calibration",
            output_path="/output",
            contains_output=False,
        )
        raw = handle.create_group("raw")
        for name, data, channel_axis in (
            ("spad", spad, "detector_channel"),
            ("aux", aux, "aux_channel"),
        ):
            dataset = raw.create_dataset(name, data=data)
            dataset.attrs.update(
                axis_order=f"repetition,z,y,x,time_bin,{channel_axis}",
                time_axis_path="/raw/axes/digital_time_ns",
                metadata_path="/raw/metadata",
                units="counts",
            )
        metadata = raw.create_group("metadata")
        metadata.attrs.update(
            time_bin_ns=time_bin_ns,
            digital_time_bin_ns=time_bin_ns,
            laser_frequency_mhz=80.0,
            nrep=shape[0],
            nz=shape[1],
            ny=shape[2],
            nx=shape[3],
            time_bins=shape[4],
        )
        axes = raw.create_group("axes")
        axes.create_dataset("digital_time_ns", data=np.arange(shape[4]) * time_bin_ns)
        raw.create_group("legacy")
        handle.create_group("calibration")
    return path


def flus2ism_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Return the central 21 by 21 pixels of the example PSF `(z, x, y, channels)` and phantom `(structures, x, y)`."""
    folder = Path(__file__).parents[2] / "examples" / "imaging" / "flus2ism"
    psf = np.load(folder / "PSFs.npy")[:, 40:61, 40:61]
    with np.load(folder / "simulated_tubulin.npz") as data:
        phantom = data["phantom"][:, 40:61, 40:61]
    return psf, phantom


def mcs_output_h5_file(path: Path) -> Path:
    """Write the `mcs_h5_file` layout plus an `/output` group with a sum run, IRF and reference traces, virtual channels and a calibration lifetime, and return its path."""
    mcs_h5_file(path)
    rng = np.random.default_rng(1)
    with h5py.File(path, "a") as handle:
        handle.attrs["contains_output"] = True
        output = handle.create_group("output")
        output.attrs.update(
            default="/output/sum_channels_001/products/spad",
            default_run="/output/sum_channels_001",
            default_irf_trace_id="sum_irf_001",
            default_ref_trace_id="sum_ref_001",
        )
        run = output.create_group("sum_channels_001")
        run.attrs.update(output_type="image_tool", output_id="sum_channels_001")
        product = run.create_group("products").create_dataset(
            "spad", data=rng.poisson(8.0, size=(1, 1, 4, 5, 8)).astype(np.uint32)
        )
        product.attrs.update(
            axis_order="repetition,z,y,x,time_bin",
            time_axis_path="/output/sum_channels_001/axes/time_ns",
            metadata_path="/raw/metadata",
            product_name="spad",
        )
        run.create_group("axes").create_dataset("time_ns", data=np.arange(8) * 0.5)
        for run_id, kind, shape in (
            ("sum_irf_001", "sum_irf_trace", (8,)),
            ("sum_ref_001", "sum_reference_trace", (8, 2)),
        ):
            trace_run = output.create_group(run_id)
            trace = trace_run.create_group("products").create_dataset(
                "trace", data=rng.poisson(20.0, size=shape).astype(np.float64)
            )
            trace.attrs.update(output_type="trace", trace_kind=kind)
            trace_run.attrs["output_type"] = "trace"
        virtual = output.create_group("virtual_channels")
        for kind, channels in (("spad", 5), ("aux", 2)):
            group = virtual.create_group(kind)
            for channel in range(channels):
                dataset = group.create_dataset(
                    f"channel_{channel}",
                    data=rng.poisson(3.0, size=(1, 1, 4, 5, 8)).astype(np.uint16),
                )
                dataset.attrs.update(
                    virtual_channel_type=kind, axis_order="repetition,z,y,x,time_bin"
                )
        fit = handle["calibration"].create_group("results/spad/fit")
        fit.create_dataset("tau_reference_ns", data=np.array([np.nan, 3.5, 4.0]))
    return path


def mcs_flim_h5_file(
    path: Path,
    tau_ns: float = 2.5,
    seed: int = 0,
    silent_channel: int | None = None,
    circular: bool = False,
) -> Path:
    """Write a legacy-layout MCS file of 64 time bins at 40 MHz holding a decay of `tau_ns`, shifted differently on each of 6 SPAD channels and 2 auxiliary channels, and return its path.

    The SPAD channel `silent_channel` holds no counts. With `circular`, the pixels carry circular repetition and point axes (8 dimensions instead of 6).
    """
    rng = np.random.default_rng(seed)
    bins = 64
    period_ns = 1e3 / 40.0
    time = np.arange(bins) * period_ns / bins

    prefix = (1, 1, 2, 2, 2, 3) if circular else (1, 1, 3, 3)

    def counts(delays: list[float]) -> np.ndarray:
        shape = np.stack(
            [
                np.exp(-np.mod(time - delay * period_ns / bins, period_ns) / tau_ns)
                for delay in delays
            ],
            axis=-1,
        )
        rate = 300.0 * shape / shape.sum(axis=0)
        return rng.poisson(np.broadcast_to(rate, (*prefix, bins, len(delays))))

    with h5py.File(path, "w") as handle:
        handle.attrs.update(data_format_version="0.0.4", comment="flim file")
        spad = counts([0.0, 1.0, 2.0, -1.0, 0.5, 1.5]).astype(np.uint16)
        if silent_channel is not None:
            spad[..., silent_channel] = 0
        handle.create_dataset("data", data=spad)
        handle.create_dataset(
            "data_channels_extra", data=counts([0.0, 3.0]).astype(np.uint16)
        )
        gui = handle.create_group("configurationGUI")
        gui.attrs.update(
            range_x=6.0,
            range_y=6.0,
            range_z=1.0,
            timebin_per_pixel=bins,
            time_resolution=0.01,
            nx=3,
            ny=3,
            nframe=1,
            nrep=1,
            bitFile="C:/bits/DFD_40M64.bit",
        )
        handle.create_group("configurationFPGA").attrs.update(DFD_Activate=True)
    return path


def mcs_legacy_h5_file(path: Path) -> Path:
    """Write a legacy-layout MCS file with root `/data` and `/data_channels_extra` and `configurationGUI` and `configurationFPGA` groups, and return its path."""
    rng = np.random.default_rng(2)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(data_format_version="0.0.4", comment="legacy file")
        handle.create_dataset(
            "data", data=rng.poisson(3.0, size=(1, 2, 4, 5, 8, 5)).astype(np.uint16)
        )
        handle.create_dataset(
            "data_channels_extra",
            data=rng.poisson(2.0, size=(1, 2, 4, 5, 8, 2)).astype(np.uint16),
        )
        gui = handle.create_group("configurationGUI")
        gui.attrs.update(
            calib_x=2.0,
            calib_y=3.0,
            calib_z=4.0,
            range_x=10.0,
            range_y=12.0,
            range_z=6.0,
            offset_x=0.5,
            offset_y=0.25,
            offset_z=0.125,
            timebin_per_pixel=8,
            time_resolution=1.0,
            nx=5,
            ny=4,
            nframe=2,
            nrep=1,
            bitFile="C:/bits/DFD_40M91.bit",
        )
        fpga = handle.create_group("configurationFPGA")
        fpga.attrs.update(
            DFD_Activate=True, DFD_Trig_Selector=2, DFD_LaserSyncDebug=False
        )
    return path


def periodic_decay(
    tau_ns: float = 2.5,
    delay_ns: float = 5.0,
    bins: int = 64,
    period_ns: float = 25.0,
    counts: float = 3000.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the time axis in ns and a Poisson histogram of a decay repeating every `period_ns`, starting at `delay_ns`."""
    time = np.arange(bins) * period_ns / bins
    shape = np.exp(-np.mod(time - delay_ns, period_ns) / tau_ns)
    rate = counts * shape / shape.sum()
    return time, np.random.default_rng(seed).poisson(rate).astype(np.float64)


def shifted_spot_dataset() -> np.ndarray:
    """Return a Poisson `(x, y, channels)` array of 41 by 41 pixels and 9 channels, each channel six Gaussian spots shifted by 1.5 pixels per detector step."""
    rng = np.random.default_rng(0)
    size = 41
    axis = np.arange(size)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    spots = rng.integers(8, 33, (6, 2))
    amplitudes = rng.uniform(1, 3, 6)
    data = np.zeros((size, size, 9))
    for channel, (dx, dy) in enumerate(
        (i * 1.5, j * 1.5) for i in (-1, 0, 1) for j in (-1, 0, 1)
    ):
        for (px, py), amplitude in zip(spots, amplitudes):
            data[..., channel] += amplitude * np.exp(
                -((x - px - dx) ** 2 + (y - py - dy) ** 2) / (2 * 2.0**2)
            )
    return rng.poisson(data * 500).astype(np.float64)


def varreg_counts(channels: int | None = None, k: int = 3) -> np.ndarray:
    """Return 16 by 16 Poisson counts shaped `(x, y, k)`, or `(x, y, channels, k)` when `channels` is given."""
    data = ism_dataset()[:16, :16]
    if channels is None:
        return data[..., :k]
    return data[..., : channels * k].reshape(16, 16, channels, k)


def varreg_psf(layers: int | None = None, channels: int | None = 2) -> np.ndarray:
    """Return a 9 by 9 PSF shaped `(x, y)`, `(x, y, channels)` or, with `layers`, `(layers, x, y, channels)`."""
    psf = ism_psf()[11:20, 11:20]
    if layers is None and channels is None:
        return psf[..., 0]
    if layers is None:
        return psf[..., :channels]
    return np.stack([psf[..., :channels], ism_psf()[11:20, 11:20, 2 : 2 + channels]])


def varreg_layout(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the data and PSF of the `confocal`, `MID` or `s2ISM` layout."""
    match name:
        case "confocal":
            return varreg_counts(), varreg_psf(channels=None)
        case "MID":
            return varreg_counts(2), varreg_psf()
        case "s2ISM":
            return varreg_counts(2), varreg_psf(layers=2)
    raise ValueError(name)


def varreg_canonical() -> tuple[torch.Tensor, torch.Tensor]:
    """Return float32 tensors of the s2ISM data `(x, y, channels, k)` and PSF `(layers, x, y, channels)`."""
    return (
        torch.from_numpy(varreg_counts(2)).float(),
        torch.from_numpy(varreg_psf(layers=2)).float(),
    )


def varreg_confocal_image() -> tuple[torch.Tensor, torch.Tensor]:
    """Return float32 tensors of the confocal data `(x, y, k)` and PSF `(x, y)`."""
    return (
        torch.from_numpy(varreg_counts()).float(),
        torch.from_numpy(varreg_psf(channels=None)).float(),
    )


def positive_tensor(*shape: int, seed: int = 1) -> torch.Tensor:
    """Return a float32 tensor of the given shape with values in 0.5 to 2."""
    return torch.from_numpy(
        np.random.default_rng(seed).uniform(0.5, 2.0, shape)
    ).float()


def unit_tensor(*shape: int, seed: int = 4) -> torch.Tensor:
    """Return a float64 tensor of the given shape with values in 0 to 1."""
    return torch.from_numpy(np.random.default_rng(seed).uniform(0, 1, shape))
