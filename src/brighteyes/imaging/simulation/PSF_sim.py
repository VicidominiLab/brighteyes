# Origin: VicidominiLab/BrightEyes-ISM, src/brighteyes_ism/simulation/PSF_sim.py at d799720.
"""Point-spread function simulation parameters and algorithms."""

from __future__ import annotations

import copy as cp

# ruff: noqa: N999
from collections.abc import Sequence
from numbers import Number
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from psf_generator.propagators import VectorialCartesianPropagator
from psf_generator.utils.zernike import create_special_pupil, create_zernike_aberrations
from zernikepy.zernike_polynomials import lookup_table

from .detector import custom_detector
from .utils import partial_convolution

Array = NDArray[np.float64]

# %% functions


class GridParameters:
    """Hold detector geometry and simulation dimensions."""

    # __slots__ = ['pxsizex', 'pxsizez', 'Nx', 'Nz', 'pxpitch', 'pxdim', 'N', 'M']

    pxsizex: float
    pxsizez: float
    Nx: int
    Nz: int
    pxpitch: float
    pxdim: float
    pinhole_shape: str
    geometry: str
    N: int | tuple[int, int]
    M: float
    rotation: float
    mirroring: int
    name: str | None

    def __init__(
        self,
        pxsizex: float = 40,
        pxsizez: float = 50,
        Nx: int = 100,
        Nz: int = 1,
        pxpitch: float = 75e3,
        pxdim: float = 50e3,
        N: int | tuple[int, int] = 5,
        M: float = 450,
    ) -> None:
        self.pxsizex = pxsizex  # nm - lateral pixel size of the images
        self.pxsizez = pxsizez  # nm - distance of the axial planes
        self.Nx = Nx  # number of samples along the X and Y axis
        self.Nz = Nz  # number of axial planes
        self.pxpitch = pxpitch  # nm - spad array pixel pitch (real space)
        self.pxdim = pxdim  # nm - spad pixel size (real space) 57.3e-3 for cooled spad
        self.pinhole_shape = "square"  # 'square', 'cirle', or 'hexagon'
        self.geometry = "rect"  # Pinholes arrangement: 'rect' or 'hex'
        self.N = N  # number of pixels in the detector in each dimension (5x5 typically)
        self.M = M  # overall magnification of the system
        self.rotation = 0  # rototion angle of the detector array (rad)
        self.mirroring = 1  # flip of the x_d axis (+/- 1)
        self.name = None  # None or 'airyscan'

    @property
    def rangex(self) -> float:
        """Return the lateral field of view."""
        return self.Nx * self.pxsizex

    @property
    def rangez(self) -> float:
        """Return the axial field of view."""
        return self.Nz * self.pxsizez

    @property
    def Nch(self) -> int | float:
        """Return the number of detector channels."""
        Ntot: int | float
        if np.ndim(self.N) == 0:
            n_sides = cast(int, self.N)
            if self.geometry == "rect":
                Ntot = n_sides**2
            elif self.geometry == "hex":
                Ntot = cast(
                    float,
                    n_sides**2 - n_sides // 2 - (1 + (-1) ** ((n_sides + 1) / 2)) * 0.5,
                )
            else:
                raise ValueError("Detector geometry not valid. Select 'rect' or 'hex'.")
        elif np.ndim(self.N) == 1:
            sizes = cast(tuple[int, int], self.N)
            Ntot = sizes[1] * sizes[0]
        else:
            raise ValueError("N has to a be a single or a couple of positive integers.")
        return Ntot

    def spad_size(
        self, mode: str = "magnified", simPar: simSettings | None = None
    ) -> float | None:
        """Return the detector size in the selected coordinate space."""
        size = self.pxpitch * (cast(int, self.N) - 1) + self.pxdim
        if simPar is not None:
            return size / self.M / simPar.airy_unit
        elif mode == "magnified":
            return size / self.M
        elif mode == "real":
            return size
        return None

    def copy(self) -> GridParameters:
        """Return a shallow copy of the grid parameters."""
        return cp.copy(self)

    def Print(self) -> None:
        """Print the grid parameters."""
        dic = self.__dict__
        names = list(dic)
        values = list(dic.values())
        for n, name in enumerate(names):
            print(name, end="")
            print(" " * int(14 - len(name)), end="")
            if values[n] is None:
                print()
            elif isinstance(values[n], Number):
                print(f"{values[n]:.2f}")
            else:
                print(values[n])


class simSettings:
    """Hold optical settings used to calculate point-spread functions."""

    # __slots__ = ['na', 'n', 'wl', 'h', 'gamma', 'beta', 'w0', 'I0', 'field', 'mask', 'mask_sampl', 'sted_sat', 'sted_pulse', 'sted_tau', 'abe_index', 'abe_ampli']

    na: float
    n: float
    wl: float
    h: float
    gamma: float
    beta: float
    w0: float
    I0: float
    field: str
    mask: str | None
    mask_sampl: int
    sted_sat: float
    sted_pulse: float
    sted_tau: float
    abe_index: int | Sequence[int] | None
    abe_ampli: float | Sequence[float] | Array | None

    def __init__(
        self,
        na: float = 1.4,
        n: float = 1.5,
        wl: float = 485.0,
        h: float = 2.8,
        gamma: float = 45.0,
        beta: float = 90.0,
        w0: float = 100.0,
        I0: float = 1,
        field: str = "PlaneWave",
        mask: str | None = None,
        mask_sampl: int = 200,
        sted_sat: float = 50,
        sted_pulse: float = 1,
        sted_tau: float = 3.5,
        abe_index: int | Sequence[int] | None = None,
        abe_ampli: float | Sequence[float] | Array | None = None,
    ) -> None:

        self.na = na  # numerical aperture
        self.n = n  # sample refractive index
        self.wl = wl  # wavelength [nm]
        self.h = h  # radius of aperture of the objective lens [mm]
        self.w0 = w0  # radius of the incident gaussian beam [mm]
        self.gamma = gamma  # parameter describing the light polarization (amplitude)
        self.beta = beta  # parameter describing the light polarization (phase)
        self.I0 = I0  # Intensity of the entrance field
        self.field = field  # entrance field at the pupil plane
        #   'PlaneWave' = Flat field
        #   'Gaussian' = Gaussian beam with waist w0
        self.mask = mask  # phase mask
        #   None = no mask
        #   'VP' = vortex phase plate
        self.mask_sampl = mask_sampl  # phase mask sampling
        self.sted_sat = sted_sat  # STED maximum saturation factor
        self.sted_pulse = sted_pulse  # STED pulse duration [ns]
        self.sted_tau = sted_tau  # fluorescence lifetime [ns]
        self.abe_index = abe_index  # aberration index (int or array)
        self.abe_ampli = abe_ampli  # aberration amplitude in rad (float or array)

    @property
    def f(self) -> float:  # focal length of the objective lens [mm]
        """Return the objective focal length."""
        return self.h * self.n / self.na

    @property
    def alpha(self) -> float:  # semiangular aperture of the objective [rad]
        """Return the objective semi-angle."""
        return cast(float, np.arcsin(self.na / self.n))

    @property
    def airy_unit(self) -> float:
        """Return the Airy unit size."""
        au = 1.22 * self.wl / self.na
        return au

    @property
    def depth_of_field(self) -> float:
        """Return the depth of field."""
        dof = 2 * self.n * self.wl / (self.na**2)
        return dof

    @property
    def aberration(self) -> str | list[str]:
        """Return names for the configured Zernike aberrations."""
        if self.abe_index is None:
            return "None"

        zernike_index: Sequence[int]
        if np.isscalar(self.abe_index):
            zernike_index = [cast(int, self.abe_index)]
        else:
            zernike_index = cast(Sequence[int], self.abe_index)

        names = []
        for name, index in lookup_table.items():
            if index in zernike_index:
                names.append(name)

        return names

    @property
    def wavefront(self) -> Array:
        """Return the pupil wavefront."""
        if self.abe_index is not None:
            zernike = torch.zeros(np.max(self.abe_index) + 1)
            zernike[self.abe_index] = torch.as_tensor(self.abe_ampli).float()
        else:
            zernike = torch.zeros(1)

        zer_wf = create_zernike_aberrations(zernike, self.mask_sampl, "cartesian")

        if self.mask is not None:
            mask_wf = create_special_pupil(self.mask_sampl, self.mask)
        else:
            mask_wf = 1

        wf = torch.angle(zer_wf * mask_wf).cpu().detach().numpy()

        x = np.linspace(-1, 1, num=self.mask_sampl)
        xx, yy = np.meshgrid(x, x)
        r = np.sqrt(xx**2 + yy**2)
        pupil = np.where(r < 1, 1, np.nan)

        return wf * pupil

    def copy(self) -> simSettings:
        """Return a shallow copy of the optical settings."""
        return cp.copy(self)

    def Print(self) -> None:
        """Print the optical settings."""
        dic = self.__dict__
        names = list(dic)
        values = list(dic.values())
        for n, name in enumerate(names):
            print(name, end="")
            print(" " * int(14 - len(name)), end="")
            print(str(values[n]))


def singlePSF(
    par: simSettings,
    pxsizex: float,
    Nx: int,
    rangez: Sequence[float],
    nz: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate an excitation point-spread function and its electric field."""
    kwargs = {
        "apod_factor": True,
        "defocus_min": rangez[0],
        "defocus_max": rangez[1],
        "n_defocus": nz,
        "n_pix_psf": Nx,
        "fov": Nx * pxsizex,
        "n_pix_pupil": par.mask_sampl,
        "na": par.na,
        "n_i0": par.n,
        "wavelength": par.wl / par.n,
        "e0x": np.cos(np.deg2rad(par.gamma)),
        "e0y": np.sin(np.deg2rad(par.gamma)) * np.exp(1j * np.deg2rad(par.beta)),
    }

    # Amplitude envelope

    if par.field == "Gaussian":
        kwargs.update({"envelope": getattr(par, "wo")})  # noqa: B009

    # Phase Mask

    if par.mask is not None:
        kwargs.update({"special_phase_mask": par.mask})

    if par.abe_index is not None:
        zernike = torch.zeros(np.max(par.abe_index) + 1)
        zernike[par.abe_index] = torch.as_tensor(par.abe_ampli).float()
        kwargs.update({"zernike_coefficients": zernike})

    propagator = VectorialCartesianPropagator(**kwargs, device=device)

    fields = propagator.compute_focus_field()

    psf = torch.sum(torch.abs(fields) ** 2, 1)

    psf = psf * par.I0 / torch.sum(psf)

    return psf, fields


def SPAD_PSF_3D(
    gridPar: GridParameters,
    exPar: simSettings,
    emPar: simSettings,
    stedPar: simSettings | None = None,
    spad: torch.Tensor | Array | None = None,
    n_photon_excitation: int = 1,
    stack: str | Sequence[float] = "symmetrical",
    normalize: bool = True,
    process: str = "gpu",
    output: str = "numpy",
) -> tuple[Array | torch.Tensor, Array | torch.Tensor, Array | torch.Tensor]:
    """Simulate excitation and detection point-spread functions over a z-stack."""
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and process == "gpu" else "cpu"
    )

    zeta: Sequence[float] | Array
    if stack == "symmetrical":
        zeta = cast(
            Array,
            (np.arange(gridPar.Nz) - gridPar.Nz // 2) * gridPar.pxsizez,
        )
    elif stack == "positive":
        zeta = cast(Array, np.arange(gridPar.Nz) * gridPar.pxsizez)
    elif stack == "negative":
        zeta = cast(Array, -np.arange(gridPar.Nz) * gridPar.pxsizez)
    else:
        zeta = cast(Sequence[float] | Array, stack)

    # simulate detector array

    pinholes: torch.Tensor
    if spad is None:
        pinholes = custom_detector(gridPar, device)
    else:
        pinholes = cast(torch.Tensor, spad)

    # Simulate ism psfs

    exPSF, _ = singlePSF(
        exPar, gridPar.pxsizex, gridPar.Nx, [zeta[0], zeta[-1]], gridPar.Nz, device
    )
    emPSF, _ = singlePSF(
        emPar, gridPar.pxsizex, gridPar.Nx, [zeta[0], zeta[-1]], gridPar.Nz, device
    )

    detPSF = partial_convolution(emPSF, pinholes, "zxy", "xyc", "xy")

    # Apply non-linearity to excitation

    if n_photon_excitation > 1:
        exPSF = exPSF**n_photon_excitation

    # Simulate donut

    if type(stedPar) == simSettings:
        stedPar.mask = "VP"
        donut = singlePSF(
            stedPar, gridPar.pxsizex, gridPar.Nx, [zeta[0], zeta[-1]], gridPar.Nz
        )
        donut *= stedPar.sted_sat / np.max(donut)
        stedPSF = np.exp(-donut * stedPar.sted_pulse / stedPar.sted_tau)
        exPSF *= stedPSF

    # Calculate total PSF

    PSF = torch.einsum("zxyc, zxy -> zxyc", detPSF, exPSF)

    if normalize is True:
        idx = np.argwhere(zeta == 0).item()
        focal_flux = PSF[idx].sum()
        for i, z in enumerate(zeta):
            PSF[i] = PSF[i] / focal_flux

    if output == "numpy":
        return (
            PSF.cpu().detach().numpy(),
            detPSF.cpu().detach().numpy(),
            exPSF.cpu().detach().numpy(),
        )
    elif output == "tensor":
        return PSF, detPSF, exPSF
    else:
        raise Exception("Output unknown")  # noqa: TRY002


def SPAD_PSF_2D(
    gridPar: GridParameters,
    exPar: simSettings,
    emPar: simSettings,
    n_photon_excitation: int = 1,
    stedPar: simSettings | None = None,
    z_shift: float = 0,
    spad: torch.Tensor | Array | None = None,
    normalize: bool = True,
    process: str = "gpu",
    output: str = "numpy",
) -> tuple[Array | torch.Tensor, Array | torch.Tensor, Array | torch.Tensor]:
    """Simulate excitation and detection point-spread functions in one plane."""
    # simulate detector array

    grid = gridPar.copy()
    grid.Nz = 1
    stack = [z_shift, z_shift]

    PSF, detPSFrot, exPSF = SPAD_PSF_3D(
        grid,
        exPar,
        emPar,
        stedPar,
        spad,
        n_photon_excitation,
        stack,
        False,
        process,
        output=output,
    )

    PSF = np.squeeze(PSF)
    detPSFrot = np.squeeze(detPSFrot)
    exPSF = np.squeeze(exPSF)

    if normalize is True:
        PSF = PSF / PSF.sum()
        detPSFrot = detPSFrot / detPSFrot.sum()
        exPSF = exPSF / exPSF.sum()

    return PSF, detPSFrot, exPSF
