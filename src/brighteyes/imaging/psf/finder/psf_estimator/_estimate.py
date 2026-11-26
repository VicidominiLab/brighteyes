# Origin: VicidominiLab/s2ISM, src/s2ism/psf_estimator.py at 10bb5fc.
"""Estimate point-spread functions from imaging measurements."""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelmax, argrelmin
from scipy.special import kl_div as kl
from scipy.stats import pearsonr as pearson

import brighteyes.imaging.utils as tool
from brighteyes.imaging._types import Array
from brighteyes.imaging.extraction.apr import ShiftVectors
from brighteyes.imaging.psf import sim

from .. import mag_finder as mag
from .. import misalignment as mis
from .. import shift_vectors_minimizer as svm


class GridFinder(sim.GridParameters):
    """Find detector settings that match measured channel shifts."""

    shift: float | None

    def __init__(self, grid_par: sim.GridParameters | None = None) -> None:
        """Initialize detector grid parameters."""
        sim.GridParameters.__init__(self)
        if grid_par is not None:
            vars(self).update(vars(grid_par))
        self.shift = None  # um

    def estimate(self, dset: Array, wl_ex: float, wl_em: float, na: float) -> None:
        """Estimate detector rotation, mirroring, and magnification."""
        ref = dset.shape[-1] // 2
        usf = 50
        # assuming a dataset with shape (Ny x Nx x Nch), we crop the outer frame of 5 pixels

        shift, _ = ShiftVectors(dset[5:-5, 5:-5, :], usf, ref, filter_sigma=1)
        par = svm.find_parameters(shift, self.geometry, self.name)
        shift_value = par[0] * self.pxsizex
        self.shift = shift_value  # um
        self.rotation = par[1]  # rad
        self.mirroring = par[2]  # +/- 1
        self.M = np.round(
            mag.find_mag(
                shift_value,
                wl_ex=wl_ex,
                wl_em=wl_em,
                pxpitch=self.pxpitch,
                pxdim=self.pxdim,
                na=na,
            )
        )


def psf_width(
    pxsizex: float,
    pxsizez: float,
    Nz: int,
    simPar: sim.simSettings,
    spad_size: float,
    stack: str = "positive",
) -> int:
    """Estimate the field width needed to contain a point-spread function."""
    if stack == "positive" or stack == "negative":
        z = pxsizez * Nz
    else:
        z = pxsizez * (Nz // 2)

    M2 = 3

    w0 = simPar.airy_unit / 2

    z_r = (np.pi * w0**2 * simPar.n) / simPar.wl
    w_z = w0 * np.sqrt(1 + (M2 * z / z_r) ** 2)

    Nx = int(np.round(2 * (w_z + spad_size) / pxsizex))

    if Nx % 2 == 0:
        Nx += 1

    return Nx


def psf_estimator_from_data(
    data: Array,
    exPar: sim.simSettings,
    emPar: sim.simSettings,
    grid: sim.GridParameters,
    downsample: bool = True,
    stedPar: sim.simSettings | None = None,
    z_out_of_focus: str | float = "ToFind",
    n_photon_excitation: int = 1,
    stack: str = "symmetrical",
    check_alignment: bool = False,
) -> tuple[Array, Array, Array]:
    """Generate point-spread functions from an imaging dataset."""
    N = int(np.sqrt(data.shape[-1]))

    # if the depth of the background plane is not set, those lines calculates it by minimizing the correlation between
    # focal PSF and the PSFs along the z-axis.
    if isinstance(z_out_of_focus, str) and z_out_of_focus == "ToFind":
        # generating a stack of PSFs along the z-axis and calculating the correlation curve to minimize
        pxsizez, _ = find_out_of_focus_from_param(
            grid.pxsizex, exPar, emPar, mode="KL", stack="positive"
        )
    else:
        pxsizez = float(z_out_of_focus)

    # find rotation, mirroring, and magnification parameters from the data
    grid_simul = GridFinder(grid)
    grid_simul.estimate(data, exPar.wl, emPar.wl, emPar.na)

    grid_simul.Print()

    if downsample is True:
        ups = int(
            find_upsampling(grid_simul.pxsizex, pxsize_sim=int(emPar.airy_unit / 100))
        )
    else:
        ups = 1

    # calculate optimal simulation range
    Nx_simul = psf_width(
        grid_simul.pxsizex,
        grid_simul.pxsizez,
        grid_simul.Nz,
        exPar,
        cast(float, grid_simul.spad_size()),
        stack=stack,
    )

    # calculate upsampled pixel size and number of pixels to have a more precise simulation of the PSFs
    pxsize_simul = grid_simul.pxsizex / ups
    Nx_up = Nx_simul * ups

    grid_simul.Nx = Nx_up
    grid_simul.pxsizex = pxsize_simul
    grid_simul.pxsizez = pxsizez
    grid_simul.N = N

    if check_alignment is True:
        tip, tilt = mis.find_misalignment(
            data, grid.pxpitch, grid.M, exPar.na, exPar.wl
        )
        exPar.abe_index = [1, 2]
        exPar.abe_ampli = [tip, tilt]

    Psf, detPsf, exPsf = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray],
        sim.SPAD_PSF_3D(
            grid_simul,
            exPar,
            emPar,
            n_photon_excitation=n_photon_excitation,
            stedPar=stedPar,
            spad=None,
            stack=stack,
        ),
    )

    if check_alignment is True:
        Psf = mis.realign_psf(Psf)

    # downsampling the PSFs to the original pixel size
    if downsample:
        Psf_ds = tool.DownSample(Psf, ups, order="zxyc")
        detPsf_ds = tool.DownSample(detPsf, ups, order="zxyc")
        exPsf_ds = tool.DownSample(exPsf, ups, order="zxy")
    else:
        Psf_ds = Psf
        detPsf_ds = detPsf
        exPsf_ds = exPsf

    return Psf_ds, detPsf_ds, exPsf_ds


def find_upsampling(pxsize_exp: float, pxsize_sim: float = 4) -> np.int64:
    """Choose an upsampling factor for point-spread function simulation."""
    # generating an array containing a set of valid values as up sampling factors
    ups = np.arange(1, np.floor(pxsize_exp)).astype(int)

    l = len(ups)

    res = np.empty(l)

    for i in range(l):
        res[i] = (
            pxsize_exp / ups[i] - pxsize_sim
        ) ** 2  # norm function to find the upsampling value that minimizes
        # the difference between the experimental and simulated pixel size (passed as default in this function)

    index = np.argmin(res)

    ups_opt = ups[index]  # optimal upsampling value retrieved

    return cast(np.int64, ups_opt)


def find_out_of_focus(
    gridPar: sim.GridParameters,
    input_psf: list[Array] | None,
    mode: str = "Pearson",
    graph: bool = True,
) -> tuple[float, list[Array]]:
    """Find the best out-of-focus plane in a supplied PSF stack."""
    if input_psf is None:
        raise Exception("PSF is not an input.")  # noqa: TRY002

    correlation, PSF = conditioning(gridPar=gridPar, input_psf=input_psf, mode=mode)

    optimal_depth = find_max_discrepancy(
        correlation=correlation, gridpar=gridPar, mode=mode, graph=graph
    )

    return optimal_depth, PSF


def find_out_of_focus_from_param(
    pxsizex: float | None = None,
    exPar: sim.simSettings | None = None,
    emPar: sim.simSettings | None = None,
    grid: sim.GridParameters | None = None,
    stedPar: sim.simSettings | None = None,
    mode: str = "Pearson",
    stack: str = "symmetrical",
    graph: bool = False,
) -> tuple[float, list[Array]]:
    """Generate a PSF stack and find its best out-of-focus plane."""
    if exPar is None:
        raise Exception("PSF parameters are needed.")  # noqa: TRY002

    if emPar is None:
        raise Exception("PSF parameters are needed.")  # noqa: TRY002

    if pxsizex is None and grid is None:
        raise Exception("Pixel size is needed as input.")  # noqa: TRY002

    if grid is None:
        range_z = 1.5 * exPar.depth_of_field
        nz = 60

        gridPar = sim.GridParameters()
        gridPar.Nz = nz
        gridPar.pxsizez = np.round(range_z / nz)
        gridPar.pxsizex = cast(float, pxsizex)
    else:
        gridPar = grid.copy()

    # finding the minimum number of pixels needed to have the simulated PSFs contained in the FOV (in order to
    # minimize the time complexity of the simulation process)

    Nx_temp = psf_width(
        pxsizex if pxsizex is not None else gridPar.pxsizex,
        gridPar.pxsizez,
        gridPar.Nz,
        exPar,
        cast(float, gridPar.spad_size()),
    )
    gridPar.Nx = Nx_temp

    # function calculating the correlation curve with Pearson correlation or Kullback-Leibler divergence
    correlation, PSF = conditioning(
        gridPar=gridPar,
        emPar=emPar,
        exPar=exPar,
        stedPar=stedPar,
        mode=mode,
        stack=stack,
    )

    optimal_depth = find_max_discrepancy(
        correlation=correlation, gridpar=gridPar, mode=mode, graph=graph
    )

    return optimal_depth, PSF


def find_max_discrepancy(
    correlation: Array, gridpar: sim.GridParameters, mode: str, graph: bool
) -> float:
    """Find the axial plane with the greatest discrepancy."""
    if mode == "KL":
        idx = np.asarray(argrelmax(correlation)).ravel()[
            0
        ]  # find maximum of the KL divergence curve
    elif mode == "Pearson":
        idx = np.asarray(argrelmin(correlation)).ravel()[
            0
        ]  # find the minimum of the Pearson correlation curve
    else:
        raise Exception("Discrepancy method unknown.")  # noqa: TRY002

    optimal_depth = cast(np.float64, idx * gridpar.pxsizez)

    # plotting the correlation/divergence curve and the optimal out-of-focus depth
    if graph:
        z = np.arange(0, gridpar.Nz * gridpar.pxsizez, gridpar.pxsizez)
        plt.figure()
        plt.plot(z, correlation)
        plt.plot(optimal_depth, correlation[idx], "ro")
        plt.vlines(
            optimal_depth,
            0,
            correlation[idx],
            linestyles="dashed",
            label="optimal_depth",
        )

    return optimal_depth


def conditioning(
    gridPar: sim.GridParameters,
    exPar: sim.simSettings | None = None,
    emPar: sim.simSettings | None = None,
    stedPar: sim.simSettings | None = None,
    mode: str = "Pearson",
    stack: str = "positive",
    input_psf: tuple[Array, Array, Array] | list[Array] | None = None,
) -> tuple[Array, list[Array]]:
    """Calculate divergence or correlation across a PSF stack."""
    if input_psf is None:
        if exPar is None or emPar is None:
            raise Exception("PSF is not an input. PSF parameters are needed.")  # noqa: TRY002

        gridPar.Nx = psf_width(
            gridPar.pxsizex,
            gridPar.pxsizez,
            gridPar.Nz,
            exPar,
            cast(float, gridPar.spad_size()),
        )
        # finding the minimum number of pixels needed to have the simulated PSFs contained in the FOV
        # (in order to minimize the time complexity of the simulation process)

        # Simulating PSFs stack
        PSF, detPSF, exPSF = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray],
            sim.SPAD_PSF_3D(
                gridPar, exPar, emPar, stedPar=stedPar, spad=None, stack=stack
            ),
        )

        # calculating crop value taking into account that PSFs inherently contain a shift as we move apart from the
        # central pixel of the detector array

        npx = int(
            np.round(
                ((cast(int, gridPar.N) // 2) * gridPar.pxpitch + gridPar.pxdim / 2)
                / gridPar.M
                / gridPar.pxsizex
            )
        )

        PSF = tool.CropEdge(PSF, npx, edges="l,r,u,d", order="zxyc")
        detPSF = tool.CropEdge(detPSF, npx, edges="l,r,u,d", order="zxyc")
        exPSF = tool.CropEdge(exPSF, npx, edges="l,r,u,d", order="zxy")
    else:
        PSF, detPSF, exPSF = input_psf

    # Normalizing PSF of each axial plane with respect to the total flux of each axial plane
    for i in range(gridPar.Nz):
        PSF[i] /= np.sum(PSF[i])

    corr = np.empty(gridPar.Nz)
    # calculating the correlation/divergence between the in-focus PSF and the PSFs at different planes
    if mode == "KL":
        for i in range(gridPar.Nz):
            corr[i] = kl(PSF[0, ...].flatten(), PSF[i, ...].flatten()).sum()

    elif mode == "Pearson":
        for i in range(gridPar.Nz):
            corr[i] = pearson(PSF[0, ...].flatten(), PSF[i, ...].flatten())[0]

    return corr, [PSF, detPSF, exPSF]


def combine_psf_irf(psf: Array, irf: Array) -> Array:
    """Combine spatial point-spread functions with temporal responses."""
    shape_in = psf.shape
    nbin = irf.shape[0]
    shape_out = shape_in + (nbin,)

    psf_rep = np.repeat(psf, nbin).reshape(shape_out)
    psf_rep = np.swapaxes(psf_rep, -2, -1)

    psf_irf = np.einsum("...lm , lm -> ...lm", psf_rep, irf)

    return cast(np.ndarray, psf_irf)
