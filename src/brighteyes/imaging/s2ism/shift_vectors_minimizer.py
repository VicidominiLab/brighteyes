# Origin: VicidominiLab/s2ISM, src/s2ism/shift_vectors_minimizer.py at 10bb5fc.
"""Fit geometric transforms to detector shift vectors."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from brighteyes.imaging.simulation.detector import airy_to_hex, det_coords

FloatArray = NDArray[np.float64]


def shift_matrix(geometry: str = "rect") -> FloatArray:
    """Return theoretical shifts for a detector layout."""
    coordinates = -det_coords(3, geometry)
    coordinates = np.swapaxes(coordinates, 0, 1)

    return coordinates


def rotation_matrix(theta: float) -> FloatArray:
    """Build a two-dimensional rotation matrix."""
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    rot = np.squeeze(rot_matrix)

    return rot


def mirror_matrix(alpha: float) -> FloatArray:
    """Build a two-dimensional mirroring matrix."""
    mirror = np.array([[1, 0], [0, alpha]])

    return mirror


def crop_shift(shift_exp: FloatArray, geometry: str = "rect") -> FloatArray:
    """Select central shifts for a three-by-three detector region."""
    n_crop = 3
    shift_cropped: FloatArray
    nch = shift_exp.shape[0]
    n = int(np.ceil(np.sqrt(nch)))

    if geometry == "rect":
        shift_exp = shift_exp.reshape(n, n, -1)
        shift_cropped = np.zeros((n_crop, n_crop, 2))

        for i, l in enumerate(np.arange(-1, 2)):
            for j, k in enumerate(np.arange(-1, 2)):
                shift_cropped[i, j, :] = shift_exp[n // 2 + l, n // 2 + k, :]

        shift_cropped = shift_cropped.reshape(n_crop**2, 2)

    elif geometry == "hex":
        c = nch // 2
        idx_crop = np.sort(
            np.asarray([c, c - 1, c + 1, c - n, c - n + 1, c + n, c + n - 1], dtype=int)
        )

        shift_cropped = shift_exp[idx_crop]

    else:
        raise Exception(  # noqa: TRY002
            "Detector geometry not valid. Select 'rect' or 'hex'."
        )

    return shift_cropped


def transform_shift_vectors(param: Sequence[float], shift: FloatArray) -> FloatArray:
    """Apply scaling, rotation, and mirroring to shift vectors."""
    a = param[0]
    r = rotation_matrix(param[1])
    m = mirror_matrix(param[2])
    transform_matrix = a * r @ m
    shift_transf = cast(Callable[..., FloatArray], np.einsum)(
        "ij,kj -> ki", transform_matrix, shift
    )

    return shift_transf


def loss_shifts(
    x0: Sequence[float] | FloatArray,
    shift_exp: FloatArray,
    shift_theor: FloatArray,
    mirror: float,
) -> float:
    """Return squared error between measured and theoretical shifts."""
    parameters = [*x0, mirror]
    shift_fin = transform_shift_vectors(parameters, shift_theor)
    loss_func = np.linalg.norm(shift_exp - shift_fin) ** 2

    return cast(float, loss_func)


def loss_minimizer(
    shift_m: FloatArray,
    shift_t: FloatArray,
    alpha_0: float,
    theta_0: float,
    tol: float,
    opt: Mapping[str, int],
    mirror: int,
) -> tuple[float, float, int]:
    """Fit magnification and rotation to measured shifts."""
    results = minimize(
        loss_shifts,
        x0=(alpha_0, theta_0),
        args=(shift_m, shift_t, mirror),
        options=opt,
        tol=tol,
        method="Nelder-Mead",
    )
    if results.success:
        alpha = results.x[0]
        theta = results.x[1]

        if alpha < 0:
            alpha = abs(alpha)
            theta += np.pi

        return cast(tuple[float, float, int], (alpha, theta, mirror))

    else:
        print("Minimization did not succeed.")
        print(results.message)

        alpha = results.x[0]
        theta = results.x[1]

        if alpha < 0:
            alpha = abs(alpha)
            theta += np.pi

        return cast(tuple[float, float, int], (alpha, theta, mirror))


def find_parameters(
    shift_exp: FloatArray,
    geometry: str = "rect",
    name: str | None = None,
    alpha_0: float = 2,
    theta_0: float = 0.5,
) -> tuple[float, float, int]:
    """Fit detector magnification, rotation, and mirroring."""
    if name == "airyscan":
        shift_vectors = airy_to_hex(shift_exp)
    else:
        shift_vectors = shift_exp

    shift_crop = crop_shift(shift_vectors, geometry)
    shift_theor = shift_matrix(geometry)
    tol = 1e-6
    opt = {"maxiter": 10000}
    params = loss_minimizer(
        shift_crop, shift_theor, alpha_0, theta_0, tol, opt, mirror=1
    )
    params_mirror = loss_minimizer(
        shift_crop, shift_theor, alpha_0, theta_0, tol, opt, mirror=-1
    )

    Loss_0 = loss_shifts(params, shift_crop, shift_theor, 1)
    Loss_1 = loss_shifts(params_mirror, shift_crop, shift_theor, -1)

    if Loss_0 < Loss_1:
        alpha = params[0]
        theta = params[1]
        mirror = 1
    else:
        alpha = params_mirror[0]
        theta = params_mirror[1]
        mirror = -1

    return alpha, theta, mirror


def calc_shift_vectors(
    parameters: Sequence[float], geometry: str = "rect"
) -> FloatArray:
    """Return shift vectors after geometric transforms."""
    shift_theor = shift_matrix(geometry)
    shift_array = transform_shift_vectors(parameters, shift_theor)

    return shift_array
