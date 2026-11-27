"""Capture BIRFI intermediate results for comparison across CPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from brighteyes import datasets
from brighteyes.imaging.birfi.birfi import Birfi


def capture(output: Path) -> None:
    """Save CPU details and each stage of the fixed BIRFI example."""
    root = Path(__file__).resolve().parents[1]
    reference = root / "tests/data/imaging"
    manifest = json.loads((reference / "birfi-simulation.json").read_text("utf-8"))
    options = manifest["parameters"]

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    with (
        np.load(datasets.fetch(manifest["dataset"]), allow_pickle=False) as inputs,
        threadpool_limits(limits=1),
    ):
        decays = inputs["decays"]
        estimator = Birfi(torch.from_numpy(decays), dt=options["dt"], device="cpu")
        estimator.find_t0_t1()
        estimator.fit_exponential(lr=options["lr"], steps=options["steps"])
        estimator.generate_data_fit()
        estimator.generate_kernel()
        estimator.richardson_lucy_deconvolution(
            iterations=options["rl_iterations"],
            regularization=options["regularization"],
        )

    assert estimator.t0 is not None and estimator.t1 is not None
    assert estimator.params is not None and estimator.data_fit is not None
    assert estimator.kernel is not None and estimator.irf is not None
    arrays = {
        "decays": decays,
        "window_start": estimator.t0.numpy(),
        "window_end": estimator.t1.numpy(),
        "fit_amplitude": estimator.params["A"].numpy(),
        "fit_offset": estimator.params["C"].numpy(),
        "fit_decay_rate": np.asarray(estimator.params["k"]),
        "fit_curves": estimator.data_fit.numpy(),
        "kernel": estimator.kernel.numpy(),
        "response": estimator.irf.numpy(),
    }
    with np.load(reference / "birfi-simulation.npz", allow_pickle=False) as frozen:
        error = np.abs(arrays["response"] - frozen["result"])
    report = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "system": platform.system(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_threads": torch.get_num_threads(),
        "packages": {
            name: version(name) for name in ("numpy", "scipy", "torch", "threadpoolctl")
        },
        "max_error_vs_frozen_response": float(error.max()),
        "stages": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
            for name, array in arrays.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "arrays.npz", **arrays)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    """Read the output directory and run the probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    capture(parser.parse_args().output)


if __name__ == "__main__":
    main()
