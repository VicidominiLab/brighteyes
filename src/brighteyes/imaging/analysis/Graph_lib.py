# Origin: VicidominiLab/BrightEyes-ISM, src/brighteyes_ism/analysis/Graph_lib.py at d799720.
# ruff: noqa: N999
"""Plotting helpers for imaging data."""

from __future__ import annotations

import numbers
from numbers import Number
from typing import TYPE_CHECKING, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.widgets import Slider
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..simulation.detector import airy_to_hex, det_coords

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure

    from .._types import Array

_hot = mpl.colormaps["hot"].resampled(256)
_hot_array = _hot(np.linspace(0, 1, 256))

_bluehot = _hot_array.copy()
_bluehot[:, 0] = _hot_array[:, 2]
_bluehot[:, 2] = _hot_array[:, 0]

bluehot = ListedColormap(_bluehot)

_greenhot = _hot_array.copy()
_greenhot[:, 0] = _hot_array[:, 1]
_greenhot[:, 1] = _hot_array[:, 0]

greenhot = ListedColormap(_greenhot)


# %%
def ShowImg(
    image: Array,
    pxsize_x: float,
    clabel: str | Number | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    colorbar: bool = True,
    cmap: str = "hot",
    integer_values: bool = True,
    log_scale: bool = False,
) -> tuple[Figure, Axes]:
    """Show an image with a scale bar and color bar."""
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Nx, Ny = image.shape
    # rangex = Nx * pxsize_x
    # rangey = Nx * pxsize_x
    # extent = (-rangex/2, rangex/2, -rangey/2, rangey/2)
    if log_scale == True:
        im = ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap, norm=LogNorm())
    else:
        im = ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)  # , extent=extent)
    ax.axis("off")

    if vmax is None:
        vmax = np.nanmax(image)
    if vmin is None:
        vmin = np.nanmin(image)

    if colorbar == True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=[])

        if integer_values == True:
            vmax_text = int(np.floor(vmax))
            vmin_text = int(np.floor(vmin))
        else:
            vmax_text = np.round(vmax, 3)
            vmin_text = np.round(vmin, 3)

        if isinstance(clabel, Number):
            clabel_text = f"Counts / {clabel:.0f} " + r"$\mathregular{\mu s}$"
        else:
            clabel_text = cast(str, clabel)

        cbar.ax.text(
            0.6,
            0.5,
            clabel_text,
            horizontalalignment="center",
            verticalalignment="center",
            rotation="vertical",
            transform=cax.transAxes,
        )
        cbar.ax.text(
            0.6,
            0.98,
            f"{vmax_text}",
            horizontalalignment="center",
            verticalalignment="top",
            rotation="vertical",
            transform=cax.transAxes,
        )
        cbar.ax.text(
            0.6,
            0.02,
            f"{vmin_text}",
            horizontalalignment="center",
            verticalalignment="bottom",
            rotation="vertical",
            transform=cax.transAxes,
            color="white",
        )

    scalebar = ScaleBar(
        pxsize_x,
        "um",  # default, extent is calibrated in meters
        box_alpha=0,
        color="w",
        length_fraction=0.25,
    )

    ax.add_artist(scalebar)

    return fig, ax


def ShowStack(
    image: Array,
    pxsize_x: float,
    pxsize_z: float,
    clabel: str | Number | None = None,
    planes: tuple[int, int, int] | None = None,
    projection: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "hot",
    figsize: tuple[float, float] = (10, 10),
) -> Figure:
    """Show three orthogonal views of an image stack."""
    Nz, Ny, Nx = image.shape

    # find slice coordinates

    if planes is None:
        x0 = Nx // 2
        y0 = Ny // 2
        z0 = Nz // 2
    else:
        z0, y0, x0 = planes

    rangez = Nz * pxsize_z
    rangex = Nx * pxsize_x
    rangey = Ny * pxsize_x

    # define extents

    extent_xy = (-rangex / 2, rangex / 2, -rangey / 2, rangey / 2)
    extent_xz = (-rangex / 2, rangex / 2, -rangez / 2, rangez / 2)
    extent_zy = (-rangez / 2, rangez / 2, -rangey / 2, rangey / 2)

    # generate projection images

    if projection == "mip":
        image_z = image.max(axis=0)
        image_y = image.max(axis=1)
        image_x = image.max(axis=2)
    else:
        image_z = image[z0, :, :]
        image_y = image[:, y0, :]
        image_x = image[:, :, x0]

    # find vmax

    if vmax is None:
        max3d = np.empty(image.ndim)

        max3d[0] = np.max(image_z)
        max3d[1] = np.max(image_y)
        max3d[2] = np.max(image_x)

        vmax = np.max(max3d)

    # find vmin
    if vmin is None:
        min3d = np.empty(image.ndim)

        min3d[0] = np.min(image_z)
        min3d[1] = np.min(image_y)
        min3d[2] = np.min(image_x)

        vmin = np.max(min3d)

    # plot figure
    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[rangez, rangex],
        height_ratios=[rangey, rangez],
        wspace=0.02,
        hspace=0.02,
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.95,
    )
    ax = np.asarray([plt.subplot(gs[i]) for i in range(4)]).reshape((2, 2))

    ax[0, 0].imshow(
        image_x[::-1, :].T, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent_zy
    )
    ax[0, 0].axis("off")

    im = ax[0, 1].imshow(image_z, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent_xy)
    ax[0, 1].axis("off")

    ax[1, 1].imshow(image_y, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent_xz)
    ax[1, 1].axis("off")

    ax[1, 0].axis("off")

    # share axes

    ax[0, 0].sharey(ax[0, 1])
    ax[0, 1].sharex(ax[1, 1])

    # add colorbar

    vmax_text = int(np.floor(vmax))
    vmin_text = int(np.floor(vmin))

    if isinstance(clabel, Number):
        clabel_text = f"Counts / {clabel:.0f} " + r"$\mathregular{\mu s}$"
    else:
        clabel_text = cast(str, clabel)

    y0 = ax[-1, -1].get_position().y0
    y1 = ax[0, 0].get_position().y1
    height = y1 - y0

    cax = fig.add_axes((0.96, y0, 0.03, height))
    cbar = fig.colorbar(im, cax=cax, ticks=[])

    cbar.ax.text(
        0.6,
        0.5,
        clabel_text,
        horizontalalignment="center",
        verticalalignment="center",
        rotation="vertical",
        transform=cax.transAxes,
    )
    cbar.ax.text(
        0.6,
        0.98,
        f"{vmax_text}",
        horizontalalignment="center",
        verticalalignment="top",
        rotation="vertical",
        transform=cax.transAxes,
    )
    cbar.ax.text(
        0.6,
        0.02,
        f"{vmin_text}",
        horizontalalignment="center",
        verticalalignment="bottom",
        rotation="vertical",
        transform=cax.transAxes,
        color="white",
    )

    # add scalebar

    scalebar = ScaleBar(
        1,
        "um",  # default, extent is calibrated in meters
        box_alpha=0,
        color="w",
        length_fraction=0.25,
    )

    ax[0, 1].add_artist(scalebar)

    return fig


def StackSlider(
    image: Array,
    pxsize_x: float,
    pxsize_z: float,
    clabel: str | Number | None = None,
    cmap: str = "hot",
    figsize: tuple[float, float] = (10, 10),
) -> tuple[Slider, Slider, Slider]:
    """Add sliders for the x, y, and z planes of a stack."""
    Nz, Ny, Nx = image.shape

    rangez = Nz * pxsize_z
    rangex = Nx * pxsize_x
    rangey = Ny * pxsize_x

    # Find color limits

    vmax = np.max(image)
    vmin = np.min(image)

    # Make figure

    fig = ShowStack(
        image=image,
        pxsize_x=pxsize_x,
        pxsize_z=pxsize_z,
        clabel=clabel,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
    )
    ax = fig.axes

    line_y = ax[1].vlines(
        0, -rangey / 2, rangey / 2, colors="white", linestyles="dashed"
    )
    line_x = ax[1].hlines(
        0, -rangex / 2, rangex / 2, colors="white", linestyles="dashed"
    )

    line_zx = ax[0].vlines(
        0, -rangey / 2, rangey / 2, colors="white", linestyles="dashed"
    )
    line_zy = ax[3].hlines(
        0, -rangex / 2, rangex / 2, colors="white", linestyles="dashed"
    )

    y0 = ax[1].get_position().y0
    y1 = ax[1].get_position().y1
    height = y1 - y0

    x0 = ax[1].get_position().x0
    x1 = ax[1].get_position().x1
    width = x1 - x0

    ax_x = fig.add_axes((x0, 0.01, width, 0.03))
    x_slider = Slider(
        ax=ax_x,
        label="X",
        valmin=0,
        valmax=Nx - 1,
        valinit=Nx // 2,
        valstep=1,
        initcolor="none",
        facecolor="grey",
        track_color="grey",
    )

    ax_y = fig.add_axes((0.01, y0, 0.0225, height))
    y_slider = Slider(
        ax=ax_y,
        label="Y",
        valmin=-Ny + 1,
        valmax=0,
        valinit=-Ny // 2,
        valstep=1,
        initcolor="none",
        orientation="vertical",
        facecolor="grey",
        track_color="grey",
    )
    y_slider.valtext.set_text(str(-y_slider.valinit))

    ax_z = fig.add_axes((x0, y1, width, 0.03))
    z_slider = Slider(
        ax=ax_z,
        label="Z",
        valmin=0,
        valmax=Nz - 1,
        valinit=Nz // 2,
        valstep=1,
        initcolor="none",
        facecolor="grey",
        track_color="grey",
    )

    # The function to be called anytime a slider's value changes
    def update_y(val: float) -> None:
        y = int(-val)
        im = ax[3].get_images()[0]
        im.set_data(image[:, y, :])

        y_units = -y * pxsize_x + rangey / 2

        seg_x = [np.array([[-rangex / 2, y_units], [rangex / 2, y_units]])]

        line_x.set_segments(seg_x)

        y_slider.valtext.set_text(str(y))

        # fig.canvas.draw_idle()

    def update_x(val: float) -> None:
        x = int(val)
        im = ax[0].get_images()[0]
        im.set_data(image[::-1, :, x].T)

        x_units = x * pxsize_x - rangex / 2

        seg_y = [np.array([[x_units, -rangey / 2], [x_units, rangey / 2]])]

        line_y.set_segments(seg_y)

        # fig.canvas.draw_idle()

    def update_z(val: float) -> None:
        z = int(val)
        im = ax[1].get_images()[0]
        im.set_data(image[z])

        z_units = -z * pxsize_z + rangez / 2

        seg_zx = [np.array([[z_units, -rangey / 2], [z_units, rangey / 2]])]

        line_zx.set_segments(seg_zx)

        seg_zy = [np.array([[-rangex / 2, z_units], [rangex / 2, z_units]])]

        line_zy.set_segments(seg_zy)

        # fig.canvas.draw_idle()

    # register the update function with each slider
    y_slider.on_changed(update_y)
    x_slider.on_changed(update_x)
    z_slider.on_changed(update_z)

    return x_slider, y_slider, z_slider


def ShowDataset(
    dset: Array,
    cmap: str = "hot",
    pxsize: float | None = None,
    normalize: bool = False,
    colorbar: bool = False,
    xlims: Sequence[float | None] | None = None,
    ylims: Sequence[float | None] | None = None,
    extent: tuple[float, float, float, float] | None = None,
    figsize: tuple[float, float] = (6, 6),
    gridshape: tuple[int, int] | None = None,
) -> Figure:
    """Display detector images in a grid."""
    if xlims is None:
        xlims = [None, None]
    if ylims is None:
        ylims = [None, None]

    if gridshape is None:
        nx = int(np.sqrt(dset.shape[-1]))
        ny = int(np.sqrt(dset.shape[-1]))
    else:
        nx = gridshape[0]
        ny = gridshape[1]

    if normalize == True:
        vmin = np.min(dset)
        vmax = np.max(dset)
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(nx, ny, sharex=True, sharey=True, figsize=figsize)
    ax = cast("Array", ax)
    for i in range(nx * ny):
        if np.min([nx, ny]) > 1:
            idx: int | tuple[int, int] = cast(
                "tuple[int, int]", np.unravel_index(i, [nx, ny])
            )
        else:
            idx = i
        if normalize == True:
            im = ax[idx].imshow(dset[:, :, i], norm=norm, cmap=cmap, extent=extent)
        else:
            im = ax[idx].imshow(dset[:, :, i], cmap=cmap, extent=extent)
        ax[idx].set_xlim(xlims)
        ax[idx].set_ylim(ylims)
        ax[idx].axis("off")

    if isinstance(pxsize, numbers.Number):
        scalebar = ScaleBar(
            pxsize,
            "um",  # default, extent is calibrated in meters
            box_alpha=0,
            color="w",
            location="lower right",
            length_fraction=0.5,
        )

        ax[-1, -1].add_artist(scalebar)

    fig.tight_layout()
    if colorbar == True and normalize == True:
        y0 = ax[-1, -1].get_position().y0
        y1 = ax[0, 0].get_position().y1
        height = y1 - y0

        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes((0.94, y0, 0.05, height))
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=[])

        cbar.ax.text(
            0.3,
            0.95,
            f"{int(np.floor(vmax))}",
            rotation=90,
            transform=cbar_ax.transAxes,
        )

        cbar.ax.text(
            0.3,
            0.02,
            f"{int(np.floor(vmin))}",
            rotation=90,
            transform=cbar_ax.transAxes,
            color="white",
        )

    return fig


def PlotShiftVectors(
    shift_vectors: Array,
    pxsize: float = 1,
    labels: bool = True,
    color: Array | str | None = None,
    cmap: str = "summer_r",
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot detector-channel shifts."""
    if fig == None or ax == None:
        fig, ax = plt.subplots()

    shift = shift_vectors * pxsize

    Nch = shift.shape[0]

    if isinstance(color, str) and color == "auto":
        N = int(np.sqrt(Nch))
        x = np.arange(-(N // 2), N // 2 + 1)
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        color = R

    ax.scatter(shift[:, 0], shift[:, 1], s=80, c=color, edgecolors="black", cmap=cmap)
    ax.set_aspect("equal", "box")

    if labels == True:
        for n in range(Nch):
            ax.annotate(str(n), shift[n], xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(r"Shift$_x$ (nm)")
    ax.set_ylabel(r"Shift$_y$ (nm)")
    ax.set_title("Shift vectors")

    ax.set_aspect("equal")

    return fig, ax


def ShowFingerprint(
    dset: Array,
    cmap: str = "hot",
    colorbar: bool = False,
    clabel: str | None = None,
    normalize: bool = False,
    fig: Figure | None = None,
    ax: Axes | None = None,
    hex_grid: bool = False,
    name: str | None = None,
) -> tuple[Figure, Axes]:
    """Display the summed detector fingerprint."""
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    fingerprint = dset.sum(axis=(0, 1))

    if name == "airyscan":
        N = 7
        hex_grid = True
        fingerprint = airy_to_hex(fingerprint)
    else:
        N = int(np.ceil(np.sqrt(dset.shape[-1])))

    if normalize is True:
        max_counts = np.nanmax(fingerprint)
        fingerprint = fingerprint / max_counts

    if hex_grid is True:
        k = (N - 1) / 2
        grid_width = int(np.real(N - (1 + (-1) ** k) / 2))
        if name == "airyscan":
            grid_width -= 1

        s = -det_coords(N, "hex")
        s = np.flip(s, axis=0)

        mappable: ScalarMappable = ax.hexbin(
            s[0], s[1], fingerprint, gridsize=(grid_width, N // 2), cmap=cmap
        )
        w = N // 2 + 1
        ax.set_xlim((-w, w))
        ax.set_ylim((-w, w))
        ax.set_aspect("equal")
    else:
        fingerprint = fingerprint.reshape(N, N)
        mappable = ax.imshow(fingerprint, cmap=cmap)
        ax.set_aspect("equal")

    ax.axis("off")
    fig.tight_layout()

    if colorbar is True:
        vmax = int(np.floor(np.max(fingerprint)))
        vmin = int(np.floor(np.min(fingerprint)))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax, ticks=[])

        cbar.ax.text(
            0.6,
            0.5,
            cast(str, clabel),
            horizontalalignment="center",
            verticalalignment="center",
            rotation="vertical",
            transform=cax.transAxes,
        )
        cbar.ax.text(
            0.6,
            0.98,
            f"{vmax}",
            horizontalalignment="center",
            verticalalignment="top",
            rotation="vertical",
            transform=cax.transAxes,
        )
        cbar.ax.text(
            0.6,
            0.02,
            f"{vmin}",
            horizontalalignment="center",
            verticalalignment="bottom",
            rotation="vertical",
            transform=cax.transAxes,
            color="white",
        )

    return fig, ax


class ColorMap2D:
    """Map intensity and a second value to RGB colors."""

    var_bounds: Sequence[float]
    int_bounds: Sequence[float]
    equalize: bool
    colormap: str

    def __init__(self) -> None:
        """Initialize the color-map settings."""
        self.var_bounds = [0, 1]  # minimum and maximum variable value of the colorbar
        self.int_bounds = [0, 1]  # minimum and maximum intensity value of the colorbar
        self.equalize = True
        self.colormap = "rainbow"

    def image(self, intensity_map: Array, variable_map: Array) -> Array:
        """Map intensity and a second variable to RGB values."""
        output = np.zeros(intensity_map.shape + (3,))

        intensity = np.clip(intensity_map, self.int_bounds[0], self.int_bounds[1])
        intensity = (intensity - self.int_bounds[0]) / (
            self.int_bounds[1] - self.int_bounds[0]
        )

        cmap = mpl.colormaps[self.colormap]
        N = cmap.N
        cmapArray = np.zeros((N, 3))

        for i in range(N):
            cmapArray[i, :] = cmap(i)[0:3]

        # equalize luminescence
        if self.equalize:
            lmax = np.max(
                (299 * cmapArray[:, 0] + 587 * cmapArray[:, 1] + 114 * cmapArray[:, 2])
                / 1000
            )
            for i in range(N):
                lum = (
                    299 * cmapArray[i, 0]
                    + 587 * cmapArray[i, 1]
                    + 114 * cmapArray[i, 2]
                ) / 1000
                lumfactor = np.min((lmax / lum, 1 / np.max(cmapArray[i, :])))
                cmapArray[i, :] *= lumfactor

        # get index of color map for lifetime
        variable = np.clip(variable_map, self.var_bounds[0], self.var_bounds[1])
        idx = (
            np.floor(
                (variable - self.var_bounds[0])
                / (self.var_bounds[1] - self.var_bounds[0])
                * N
            )
        ).astype(int)
        idx = np.clip(idx, 0, N - 1)
        for k in range(3):
            output[:, :, k] = np.take(cmapArray[:, k], idx) * intensity

        return output

    def colorbar(self, n: int) -> Array:
        """Build an RGB color bar."""
        LG = np.linspace(self.var_bounds[0], self.var_bounds[1], num=n)
        VarGradient = np.tile(LG, (n, 1))

        IG = np.linspace(self.int_bounds[0], self.int_bounds[1], num=n)
        IntensityGradient = np.transpose(np.tile(IG, (n, 1)))

        RGB_colormap = self.image(IntensityGradient, VarGradient)
        RGB_colormap = np.moveaxis(RGB_colormap, 0, 1)

        return RGB_colormap


def show_flim(
    image: Array,
    lifetime: Array,
    pxsize: float,
    pxdwelltime: float,
    lifetime_bounds: Sequence[float] | None = None,
    intensity_bounds: Sequence[float] | None = None,
    colormap: str = "gist_rainbow",
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Display intensity and lifetime in a shared color map."""
    # params settings

    cmap = ColorMap2D()
    cmap.colormap = colormap

    if intensity_bounds is None:
        cmap.int_bounds = [np.min(image), np.max(image)]
    else:
        cmap.int_bounds = intensity_bounds

    if lifetime_bounds is None:
        cmap.var_bounds = [np.min(lifetime), np.max(lifetime)]
    else:
        cmap.var_bounds = lifetime_bounds

    # Image

    RGB = cmap.image(image, lifetime)

    # Colorbar

    N = 256
    RGB_colormap = cmap.colorbar(N)

    # Define extents

    sz = image.shape

    img_extent = (0, sz[1] * pxsize, 0, sz[0] * pxsize)
    cmap_extent = (
        cmap.int_bounds[0],
        cmap.int_bounds[1],
        cmap.var_bounds[0],
        cmap.var_bounds[1],
    )

    # Show image with colorbar

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    ax.imshow(RGB, extent=img_extent)
    ax.axis("off")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cax.imshow(RGB_colormap, origin="lower", aspect="auto", extent=cmap_extent)
    cax.set_xticks([int(cmap.int_bounds[0]), int(cmap.int_bounds[1])])
    cax.set_xlabel(f"Counts/{pxdwelltime} " + r"$\mathregular{\mu s}$")
    cax.set_ylabel(r"Lifetime (ns)")
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")

    # Add scalebar

    scalebar = ScaleBar(
        1,
        "um",  # default, extent is calibrated in meters
        box_alpha=0,
        color="w",
        length_fraction=0.25,
    )

    ax.add_artist(scalebar)

    plt.tight_layout()

    return fig, ax


def depth_stack(
    stack: Array,
    pxsize: Sequence[float],
    pxdwelltime: float,
    axis: int = 0,
    colormap: str = "rainbow",
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Display maximum intensity and depth projections."""
    image_mip = stack.max(axis=axis)
    depth_mip = stack.argmax(axis=axis) * pxsize[axis]

    # params settings

    cmap = ColorMap2D()
    cmap.colormap = colormap

    cmap.int_bounds = [np.min(image_mip), np.max(image_mip)]
    cmap.var_bounds = [np.min(depth_mip), np.max(depth_mip)]

    # Image

    RGB = cmap.image(image_mip, depth_mip)

    # Colorbar

    N = stack.shape[axis]
    RGB_colormap = cmap.colorbar(N)

    # Define extents

    sz = image_mip.shape
    px = np.delete(pxsize, axis)

    img_extent = (0, sz[1] * px[1], 0, sz[0] * px[0])
    cmap_extent = (
        cmap.int_bounds[0],
        cmap.int_bounds[1],
        cmap.var_bounds[0],
        cmap.var_bounds[1],
    )

    # Show image with colorbar

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    ax.imshow(RGB, extent=img_extent)
    ax.axis("off")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cax.imshow(RGB_colormap, origin="lower", aspect="auto", extent=cmap_extent)
    cax.set_xticks([int(cmap.int_bounds[0]), int(cmap.int_bounds[1])])
    cax.set_xlabel(f"Counts/{pxdwelltime} " + r"$\mathregular{\mu s}$")
    cax.set_ylabel(r"Depth ($\mathregular{\mu m}$)")
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")

    # Add scalebar

    scalebar = ScaleBar(
        1,
        "um",  # default, extent is calibrated in meters
        box_alpha=0,
        color="w",
        length_fraction=0.25,
    )

    ax.add_artist(scalebar)

    plt.tight_layout()

    return fig, ax
