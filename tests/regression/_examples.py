"""Run numerical example stages with fixed CPU settings."""

import json
import os
import random
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

original_modules = {
    "apr": "brighteyes.imaging.analysis.APR_lib",
    "frc": "brighteyes.imaging.analysis.FRC_lib",
    "birfi": "brighteyes.imaging.birfi.birfi",
}
current_modules = {
    **original_modules,
    "apr": "brighteyes.imaging.extraction.apr",
    "frc": "brighteyes.imaging.utils",
}


def numerical_arrays(value: object, prefix: str = "result") -> dict[str, np.ndarray]:
    """Flatten numerical results into named arrays without pickled objects."""
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu().numpy()}
    if isinstance(value, (tuple, list)):
        return {
            key: array
            for index, item in enumerate(value)
            for key, array in numerical_arrays(item, f"{prefix}_{index}").items()
        }
    if value is None:
        return {}
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"non-numerical result at {prefix}")
    return {prefix: array}


def execute(
    example: str,
    inputs: dict[str, np.ndarray],
    parameters: dict,
    original: bool = False,
) -> dict[str, np.ndarray]:
    """Compute a registered example stage using its recorded parameters."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    modules = original_modules if original else current_modules
    match example:
        case "ism-apr":
            image = inputs["raw"].sum(axis=(0, 1, 4))
            result = import_module(modules["apr"]).APR(image, **parameters)
        case "ism-frc":
            options = dict(parameters)
            reassignment = options.pop("apr")
            frames = inputs["raw"].sum(axis=(0, 1))
            images = [frames[:, :, offset::2, :].sum(axis=-2) for offset in (0, 1)]
            reassigned = [
                import_module(modules["apr"]).APR(image, **reassignment)[1]
                for image in images
            ]
            result = (
                import_module(modules["frc"]).FRC_resolution(
                    images[0].sum(-1), images[1].sum(-1), **options
                ),
                import_module(modules["frc"]).FRC_resolution(
                    reassigned[0].sum(-1), reassigned[1].sum(-1), **options
                ),
            )
        case "birfi-simulation":
            options = dict(parameters)
            dt = options.pop("dt")
            estimator = import_module(modules["birfi"]).Birfi(
                torch.from_numpy(inputs["decays"]), dt=dt, device=torch.device("cpu")
            )
            result = estimator.run(**options)
        case _:
            raise ValueError(f"unknown example stage: {example}")
    return numerical_arrays(result)


def main() -> None:
    """Save results only after verifying the reference source location."""
    example, input_file, output_file, parameters = sys.argv[1:]
    expected = Path(os.environ["BRIGHTEYES_REFERENCE_ROOT"]).resolve()
    actual = Path(import_module("brighteyes").__file__).resolve()
    if not actual.is_relative_to(expected):
        raise RuntimeError("reference source was not imported")
    torch.set_num_threads(1)
    with np.load(input_file, allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}
    np.savez_compressed(
        output_file, **execute(example, inputs, json.loads(parameters), original=True)
    )


if __name__ == "__main__":
    main()
