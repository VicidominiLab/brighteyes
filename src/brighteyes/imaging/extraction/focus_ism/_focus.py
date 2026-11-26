# Origin: VicidominiLab/BrightEyes-ISM, src/brighteyes_ism/analysis/FocusISM_lib.py at d799720.
"""Focused signal reconstruction for image scanning microscopy data."""

from __future__ import annotations

import multiprocessing
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from joblib import Parallel, delayed
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from sklearn.metrics import r2_score
from tqdm import trange

from .. import apr as APR

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.figure import Figure

    from ..._types import Array

# %%


class Selector:
    """Select a rectangular region from an image."""

    fig: Figure
    ax: Axes
    coord: Array | None
    rs: RectangleSelector | None

    def __init__(self, img: Array) -> None:
        """Create a selector and wait for a region."""
        self.fig, self.ax = plt.subplots()
        m = self.ax.imshow(img)
        plt.colorbar(m, ax=self.ax)
        plt.axis("off")
        self.ax.set_title("Drag a region with mouse")

        self.coord = None
        self.rs = None

        self.select()

        while self.coord is None:
            plt.pause(1)

    def line_select_callback(self, eclick: MouseEvent, erelease: MouseEvent) -> None:
        """Store the selected image region."""
        x1, y1 = cast(float, eclick.xdata), cast(float, eclick.ydata)
        x2, y2 = cast(float, erelease.xdata), cast(float, erelease.ydata)

        self.coord = np.asarray([[x1, y1], [x2, y2]]).astype(int)

        rect = Rectangle(
            (min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), fill=False
        )
        self.ax.add_patch(rect)

        assert self.rs is not None
        self.rs.set_visible(False)
        self.rs.set_active(False)

    def select(self) -> None:
        """Create the rectangle selector."""
        self.rs = RectangleSelector(
            self.ax,
            self.line_select_callback,
            useblit=False,
            button=[MouseButton.LEFT],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )


# %%


def fingerprint_model(xdata: Array, A: float, sigma: float, BKG: float = 0) -> Array:
    """Evaluate the Gaussian fingerprint model."""
    x = xdata[0, :]
    y = xdata[1, :]
    return cast("Array", A * np.exp(-((x) ** 2 + (y) ** 2) / (2 * sigma**2)) + BKG)


def gaussian(x: Array, y: Array, x0: float, y0: float, A: float, sigma: float) -> Array:
    """Evaluate a normalized two-dimensional Gaussian."""
    gauss = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    return cast("Array", A * gauss / np.sum(gauss))


def gaussian_model(xdata: Array, sigma_A: float, B: float, sigma_B: float) -> Array:
    """Evaluate the two-component Gaussian model."""
    x = xdata[0, :]
    y = xdata[1, :]
    # sigma_A = 1.3*( 0.83 / np.sqrt(2*np.log(2)) )
    A = 1 - B
    return gaussian(x, y, 0, 0, A, sigma_A) + gaussian(x, y, 0, 0, B, sigma_B)


# %%


def FitFingerprint(
    sub_img: Array,
    p0: Sequence[float] = (1, 1),
    Ndet: int = 5,
) -> tuple[Array, Array, Array]:
    """Fit a Gaussian model to a detector fingerprint."""
    # Calculate fingerprint

    fingerprint = np.sum(sub_img, axis=(0, 1)).reshape(Ndet, Ndet)
    fingerprint = fingerprint / np.max(fingerprint)

    # The two-dimensional domain of the fit.

    x = np.linspace(-(Ndet // 2), Ndet // 2, Ndet)
    X, Y = np.meshgrid(x, x)

    xdata = np.vstack((X.ravel(), Y.ravel()))

    # Fit fingerprint with a single gaussian function

    popt, _ = opt.curve_fit(fingerprint_model, xdata, fingerprint.ravel(), p0)
    popt = cast("Array", popt)

    A = cast(float, popt[0])
    sigma_fing = cast(float, popt[1])
    BKG = 0

    fit_fingerprint = fingerprint_model(xdata, A, sigma_fing, BKG).reshape(Ndet, Ndet)

    return fingerprint, fit_fingerprint, popt


# %%


def pixel_fit_2(
    F: Array,
    sigma_A: float,
    sigma_B_bound: float | None = None,
    threshold: float = 0,
) -> tuple[Array, Array, float, float]:
    """Split a micro-image into focused and background components."""
    sigma_B: float
    r2: float
    """Split a micro-image into focused and background components."""

    if sigma_B_bound is None:
        sigma_B_bound = 3 * sigma_A

    N = np.sum(F)  # Normalization factor - sum of all pixels

    if N > threshold:  # Thresholding
        F = F / N

        # The two-dimensional domain of the fit.

        xmin, xmax, nx = -2, 2, 5
        ymin, ymax, ny = -2, 2, 5
        x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)

        # Ravel the meshgrids of X, Y points to a pair of 1-D arrays.

        xdata = np.vstack((X.ravel(), Y.ravel()))

        # Perform fitting

        p0 = [0.1, 4 * sigma_A]
        fit_model = lambda xdata, B, sigma_B: gaussian_model(xdata, sigma_A, B, sigma_B)

        try:
            popt, _ = opt.curve_fit(
                fit_model, xdata, F, p0, bounds=([0, sigma_B_bound], [1, np.inf])
            )
            popt = cast("Array", popt)
            A = 1 - cast(float, popt[0])
            B = cast(float, popt[0])
            sigma_B = cast(float, popt[1])
        except Exception:  # noqa: BLE001
            bkg = F
            sig = np.zeros(F.shape)
            sigma_B = 0
            return bkg, sig, sigma_B, 0

        sig = N * gaussian(X, Y, 0, 0, A, sigma_A).ravel()
        bkg = N * gaussian(X, Y, 0, 0, B, sigma_B).ravel()

        r2 = cast(float, r2_score(F * N, sig + bkg))
    else:
        bkg = F
        sig = np.zeros(F.shape)
        sigma_B = 0
        r2 = 0

    return bkg, sig, sigma_B, r2


def pixel_fit_1(
    F: Array,
    sigma_A: float,
    sigma_B: float,
    threshold: float = 0,
) -> tuple[Array, Array, float]:
    """Split a micro-image using a fixed background width."""
    r2: float
    """Split a micro-image using a fixed background width."""

    N = np.sum(F)  # Normalization factor - sum of all pixels

    if N > threshold:  # Thresholding
        F = F / N

        # The two-dimensional domain of the fit.

        xmin, xmax, nx = -2, 2, 5
        ymin, ymax, ny = -2, 2, 5
        x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)

        # Ravel the meshgrids of X, Y points to a pair of 1-D arrays.

        xdata = np.vstack((X.ravel(), Y.ravel()))

        # Perform fitting

        p0 = [0.5]
        fit_model = lambda xdata, B: gaussian_model(xdata, sigma_A, B, sigma_B)
        popt, _ = opt.curve_fit(fit_model, xdata, F, p0, bounds=(0, [1]))

        # Classify signal and background

        A = 1 - cast(float, popt[0])
        B = cast(float, popt[0])

        sig = N * gaussian(X, Y, 0, 0, A, sigma_A).ravel()
        bkg = N * gaussian(X, Y, 0, 0, B, sigma_B).ravel()

        r2 = cast(float, r2_score(F * N, sig + bkg))
    else:
        bkg = F
        sig = np.zeros(F.shape)
        r2 = 0

    return bkg, sig, r2


# %%


def focusISM(
    img: Array,
    sigma_B_bound: float | None = None,
    threshold: float = 0,
    apr: bool = True,
    calibration: str | Array = "manual",
    sum_results: bool = True,
    parallelize: bool = True,
) -> tuple[Array, Array, Array]:
    """Reconstruct focused and background signal from an ISM image."""
    """Reconstruct focused and background signal from an ISM image."""

    sz = img.shape
    Ndet = int(np.sqrt(sz[-1]))

    usf = 10
    ref = Ndet**2 // 2

    # Adaptive pixel reassignment

    if apr == True:
        _, img_ism = APR.APR(img, usf, ref, filter_sigma=1)
        img_ism[img_ism < 0] = 0
    else:
        img_ism = img

    # Calibrate algorithm with fingerprint

    ism_sum = np.sum(img_ism, axis=2)

    subimg: Array
    if isinstance(calibration, str) and calibration == "manual":
        xy = Selector(ism_sum).coord
        assert xy is not None
        subimg = img[xy[0, 1] : xy[1, 1], xy[0, 0] : xy[1, 0], :]
    else:
        subimg = cast("Array", calibration)

    _, _, popt = FitFingerprint(subimg)
    sigma_fing = cast(float, popt[1])

    # Fit each micro-image with two Gaussians

    sz = img.shape
    img_reshaped = img_ism.reshape(sz[0] * sz[1], sz[2])

    N = multiprocessing.cpu_count() - 1

    sigma_A = sigma_fing

    if sigma_B_bound is None:
        sigma_B_bound = 2 * sigma_A
    else:
        sigma_B_bound *= sigma_A

    results: list[tuple[Array, Array, float, float]]
    if parallelize == True:
        print("Focus-ISM - parallel:")
        results = cast(
            "list[tuple[Array, Array, float, float]]",
            Parallel(n_jobs=N, backend="threading")(
                delayed(pixel_fit_2)(
                    img_reshaped[i, :],
                    sigma_A,
                    sigma_B_bound=sigma_B_bound,
                    threshold=threshold,
                )
                for i in trange(sz[0] * sz[1])
            ),
        )
    else:
        results = []
        print("Focus-ISM:")
        for i in trange(sz[0] * sz[1]):
            results.append(
                pixel_fit_2(
                    img_reshaped[i, :],
                    sigma_A,
                    sigma_B_bound=sigma_B_bound,
                    threshold=threshold,
                )
            )

    # Reshape

    bkg = np.asarray([results[j][0] for j in range(len(results))]).reshape(
        sz[0], sz[1], sz[2]
    )

    sig = np.asarray([results[j][1] for j in range(len(results))]).reshape(
        sz[0], sz[1], sz[2]
    )

    if sum_results == True:
        bkg_sum = np.sum(bkg, axis=2)

        focus_sum = np.sum(sig, axis=2)

        return focus_sum, bkg_sum, ism_sum

    else:
        return sig, bkg, img_ism
