# Origin: VicidominiLab/BrightEyes-FLISM, src/brighteyes_flism/_fit_maps.py at 85f0d96.
"""Fit-map helpers exported by :mod:`brighteyes.imaging.flism`."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from os import cpu_count
from typing import Any

import numpy as np
from joblib import Parallel, delayed, parallel
from numpy.typing import NDArray
from tqdm.auto import tqdm

from brighteyes.mcs.dataprep import Alignment


def _fit_map_pixel_chunk(
    indices: NDArray[Any], histograms: NDArray[Any], **worker_kwargs: Any
) -> list[Any]:
    return [
        Alignment._fit_map_one_pixel(int(idx), hist, **worker_kwargs)
        for idx, hist in zip(indices, histograms)
    ]


def _fit_map_job_chunk_size(
    valid_count: int, n_jobs: int, job_chunk_size: int | None
) -> int:
    if job_chunk_size is not None:
        return max(1, int(job_chunk_size))

    if int(n_jobs) == 1:
        return 1

    if int(n_jobs) > 0:
        worker_count = int(n_jobs)
    else:
        try:
            worker_count = cpu_count() or 1
        except Exception:  # pragma: no cover - extremely defensive fallback  # noqa: BLE001
            worker_count = 1

    target_chunks = max(worker_count * 8, 1)
    return max(1, int(np.ceil(valid_count / target_chunks)))


@contextmanager
def _joblib_completed_pixel_progress(
    progress_bar: Any, pixels_per_task: int
) -> Generator[Any, None, None]:
    """Update tqdm when joblib batches finish instead of when tasks are queued."""
    original_callback = parallel.BatchCompletionCallBack

    def completed_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        remaining = progress_bar.total - progress_bar.n
        if remaining > 0:
            progress_bar.update(min(remaining, self.batch_size * pixels_per_task))
        return original_callback.__call__(self, *args, **kwargs)

    completed_callback = type(
        "CompletedPixelCallback",
        (original_callback,),
        {"__call__": completed_call},
    )

    parallel.BatchCompletionCallBack = completed_callback
    try:
        yield progress_bar
    finally:
        parallel.BatchCompletionCallBack = original_callback
        progress_bar.close()


def generate_fit_maps(
    data: Any,
    irf: Any,
    t: Any,
    period: float,
    initial_tau: float | None = None,
    initial_dT: float | None = None,
    initial_C: float | None = None,
    mode: str = "irf_shift",
    fit_type: str = "likelihood",
    force_C_normalized: bool = True,
    min_counts: float = 0.0,
    min_peak_counts: float = 0.0,
    min_nonzero_bins: int = 1,
    valid_mask: Any = None,
    n_jobs: int = 1,
    backend: str = "loky",
    job_chunk_size: int | None = None,
    show_progress: bool = True,
    catch_exceptions: bool = True,
) -> dict[str, NDArray[np.float64]]:
    """Fit each pixel histogram and return maps of the fitted parameters."""
    data_array = np.asarray(data, dtype=float)
    irf_hist = Alignment.to_numpy_1d(irf, dtype=float)
    t_ns = Alignment.to_numpy_1d(t, dtype=float)

    if data_array.ndim != 3:
        raise ValueError(
            f"data must have shape (ny, nx, nbins), got {data_array.shape}"
        )

    ny, nx, nbins = data_array.shape
    if irf_hist.shape != (nbins,):
        raise ValueError(f"irf must have shape ({nbins},), got {irf_hist.shape}")
    if t_ns.shape != (nbins,):
        raise ValueError(f"t must have shape ({nbins},), got {t_ns.shape}")
    if not np.all(np.isfinite(t_ns)):
        raise ValueError("t contains non-finite values")
    if not np.all(np.isfinite(irf_hist)) or np.sum(irf_hist) <= 0:
        raise ValueError("irf contains non-finite values or has non-positive sum")

    data_2d = data_array.reshape(-1, nbins)
    data_sums = np.sum(data_2d, axis=1)
    pixel_is_valid = (
        np.all(np.isfinite(data_2d), axis=1)
        & (data_sums > float(min_counts))
        & (np.max(data_2d, axis=1) >= float(min_peak_counts))
        & (np.count_nonzero(data_2d > 0, axis=1) >= int(min_nonzero_bins))
    )

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape == (ny, nx):
            valid_mask = valid_mask.ravel()
        elif valid_mask.shape != (ny * nx,):
            raise ValueError(
                f"valid_mask must have shape {(ny, nx)} or {(ny * nx,)}, got {valid_mask.shape}"
            )
        pixel_is_valid &= valid_mask

    valid_indices = np.flatnonzero(pixel_is_valid)

    fit_maps = {
        "C": np.full((ny, nx), np.nan, dtype=float),
        "dT": np.full((ny, nx), np.nan, dtype=float),
        "tau": np.full((ny, nx), np.nan, dtype=float),
        "C_err": np.full((ny, nx), np.nan, dtype=float),
        "dT_err": np.full((ny, nx), np.nan, dtype=float),
        "tau_err": np.full((ny, nx), np.nan, dtype=float),
    }

    if valid_indices.size == 0:
        return fit_maps

    worker_kwargs: dict[str, Any] = {
        "nx": nx,
        "t": t_ns,
        "irf": irf_hist,
        "period": float(period),
        "initial_tau": initial_tau,
        "initial_dT": initial_dT,
        "initial_C": initial_C,
        "mode": mode,
        "fit_type": fit_type,
        "force_C_normalized": force_C_normalized,
        "catch_exceptions": catch_exceptions,
    }

    if int(n_jobs) == 1:
        iterator = valid_indices
        if show_progress:
            iterator = tqdm(
                iterator,
                total=int(valid_indices.size),
                desc="Fitting pixels",
                unit="px",
            )
        results = [
            Alignment._fit_map_one_pixel(int(idx), data_2d[int(idx)], **worker_kwargs)
            for idx in iterator
        ]
    else:
        chunk_size = _fit_map_job_chunk_size(
            valid_indices.size,
            n_jobs=n_jobs,
            job_chunk_size=job_chunk_size,
        )
        index_chunks = [
            valid_indices[start : start + chunk_size]
            for start in range(0, valid_indices.size, chunk_size)
        ]
        parallel_call = (
            delayed(_fit_map_pixel_chunk)(
                chunk,
                data_2d[chunk],
                **worker_kwargs,
            )
            for chunk in index_chunks
        )

        if show_progress:
            progress_bar = tqdm(
                total=int(valid_indices.size),
                desc="Fitting pixels",
                unit="px",
            )
            with _joblib_completed_pixel_progress(
                progress_bar, pixels_per_task=int(chunk_size)
            ):
                chunk_results = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
                    parallel_call
                )
        else:
            chunk_results = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
                parallel_call
            )

        results = [row for chunk_result in chunk_results for row in chunk_result]

    for y, x, C, dT, tau, C_err, dT_err, tau_err in results:
        fit_maps["C"][y, x] = C
        fit_maps["dT"][y, x] = dT
        fit_maps["tau"][y, x] = tau
        fit_maps["C_err"][y, x] = C_err
        fit_maps["dT_err"][y, x] = dT_err
        fit_maps["tau_err"][y, x] = tau_err

    return fit_maps
