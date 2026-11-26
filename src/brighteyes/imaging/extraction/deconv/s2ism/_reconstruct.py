# Origin: VicidominiLab/s2ISM, src/s2ism/s2ism.py at 10bb5fc.
"""s2ISM deconvolution algorithms."""

from __future__ import annotations

import gc
import string
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as torchpad
from torch import einsum, real
from torch.fft import ifftshift, irfftn, rfftn
from tqdm import tqdm

from ....psf.finder import psf_estimator as svr

if TYPE_CHECKING:
    from .... import _types
    from ....psf import sim

    Array = _types.Array


def partial_convolution_rfft(
    kernel: torch.Tensor,
    volume: torch.Tensor,
    dim1: str = "ijk",
    dim2: str = "jkl",
    axis: str = "jk",
    fourier: tuple[bool | int, bool | int] = (False, False),
    padding: Sequence[int] | None = None,
) -> torch.Tensor:
    """Convolve selected dimensions with real FFTs."""
    dim3 = dim1 + dim2
    dim3 = "".join(sorted(set(dim3), key=dim3.index))

    dims = [dim1, dim2, dim3]
    axis_list = [[d.find(c) for c in axis] for d in dims]

    if padding is None:
        padding = [volume.size(d) for d in axis_list[1]]

    if fourier[0] == False:
        kernel_fft = rfftn(kernel, dim=axis_list[0], s=padding)
    else:
        kernel_fft = kernel

    if fourier[1] == False:
        volume_fft = rfftn(volume, dim=axis_list[1], s=padding)
    else:
        volume_fft = volume

    conv = einsum(f"{dim1},{dim2}->{dim3}", kernel_fft, volume_fft)

    conv = irfftn(conv, dim=axis_list[2], s=padding)  # inverse FFT of the product
    conv = ifftshift(
        conv, dim=axis_list[2]
    )  # Rotation of 180 degrees of the phase of the FFT
    conv = real(conv)  # Clipping to zero the residual imaginary part

    return conv


def amd_update_fft(
    img: torch.Tensor,
    obj: torch.Tensor,
    psf_fft: torch.Tensor,
    psf_m_fft: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply one expectation-maximization update in Fourier space."""
    # Variables initialization

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    alphabet = string.ascii_lowercase  # Letters from 'a' to 'z'
    n_dim = psf_fft.ndim  # dimension of the PSF

    str_psf = alphabet[:n_dim]  # Takes the first n_dim letters of alphabet Es. 'abcde'
    str_first = alphabet[
        : n_dim - 1
    ]  # Takes the first n_dim-1 letters of alphabet Es. 'abcd'
    str_second = alphabet[
        1:n_dim
    ]  # Takes the letters from the second to n_dim of alphabet Es. 'bcde'
    axis_str = alphabet[
        1 : n_dim - 1
    ]  # Common axis on which to perform convolution Es. 'bcd'

    # Update
    img_estimate = partial_convolution_rfft(
        psf_fft, obj, dim1=str_psf, dim2=str_first, axis=axis_str, fourier=(1, 0)
    ).sum(0)
    fraction = torch.where(img_estimate < eps, 0, img / img_estimate)
    del img_estimate
    update = partial_convolution_rfft(
        psf_m_fft,
        fraction,
        dim1=str_psf,
        dim2=str_second,
        axis=axis_str,
        fourier=(1, 0),
    ).sum(-1)
    del fraction
    obj_new = obj * update

    return obj_new


def amd_stop(
    o_old: torch.Tensor,
    o_new: torch.Tensor,
    pre_flag: bool,
    flag: bool,
    stop: str,
    max_iter: int,
    threshold: float,
    tot: torch.Tensor,
    nz: int,
    k: int,
) -> tuple[bool, bool, torch.Tensor, torch.Tensor]:
    """Update convergence flags and photon statistics."""
    # calculating photon flux in the focal plane reconstruction at the previous iteration
    int_f_old = (o_old[nz // 2]).sum()

    # calculating the photon flux in the focal plane reconstruction at the current iteration
    int_f_new = (o_new[nz // 2]).sum()

    # calculating the derivative of the photon count function in the focal plane
    d_int_f = (int_f_new - int_f_old) / tot

    # calculating the photon flux in the out-of-focus planes reconstruction at the previous iteration
    int_bkg_old = o_old.sum() - int_f_old

    # calculating the photon flux in the out-of-focus planes reconstruction at the current iteration
    int_bkg_new = o_new.sum() - int_f_new

    # calculating the derivative of the photon count function in the out-of-focus planes
    d_int_bkg = (int_bkg_new - int_bkg_old) / tot

    # controlling if the derivative value is under the threshold. The algorithm derivative has to lye under the
    # threshold for two consecutive iterations to stop.
    if isinstance(stop, str) and stop == "auto":
        if torch.abs(d_int_f) < threshold:
            if not pre_flag:
                flag = False
            else:
                pre_flag = False
        elif k == max_iter:
            flag = False
            print("Reached maximum number of iterations.")
    # if the iteration stop rule il claimed to be fixed, the algorithm stop when the maximum number of iterations is
    # reached, default value is 100.
    elif stop == "fixed" and k == max_iter:
        flag = False

    return (
        pre_flag,
        flag,
        torch.Tensor([int_f_new, int_bkg_new]),
        torch.Tensor([d_int_f, d_int_bkg]),
    )


def batch_reconstruction(
    dset: Array,
    psf: Array,
    batch_size: Sequence[int],
    overlap: int,
    stop: str = "fixed",
    max_iter: int = 100,
    threshold: float = 1e-3,
    rep_to_save: str | Sequence[int] | Array = "last",
    initialization: str = "flat",
    process: str = "gpu",
) -> Array:
    """Reconstruct an image in overlapping spatial batches."""
    wx, wy = batch_size
    reconstruction = np.zeros((psf.shape[0],) + dset.shape[:-1])

    nx = dset.shape[0] // (wx - overlap) + int(dset.shape[0] % (wx - overlap) > 0)
    ny = dset.shape[1] // (wy - overlap) + int(dset.shape[1] % (wy - overlap) > 0)

    k_iter = 1
    n_batch = nx * ny

    rx = np.mod(dset.shape[0], wx - overlap)
    ry = np.mod(dset.shape[1], wy - overlap)
    thx = psf.shape[1]
    thy = psf.shape[2]

    if np.logical_and(rx > 0, rx < thx) or np.logical_and(ry > 0, ry < thy):
        raise ValueError("The PSF is bigger than the last batch size.")

    for i in range(nx):
        for j in range(ny):
            print(f"Batch {k_iter}/{n_batch}")

            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            overlap_x = overlap if i > 0 else 0
            overlap_y = overlap if j > 0 else 0

            indx = np.s_[i * (wx - overlap_x) : (i + 1) * wx - i * overlap_x]
            indy = np.s_[j * (wy - overlap_y) : (j + 1) * wy - j * overlap_y]

            crop_dset = dset[indx, indy]

            recon, *_ = max_likelihood_reconstruction(
                dset=crop_dset,
                psf=psf,
                stop=stop,
                max_iter=max_iter,
                threshold=threshold,
                rep_to_save=rep_to_save,
                initialization=initialization,
                process=process,
            )

            indx = np.s_[
                i * (wx - overlap_x) + overlap_x // 2 : (i + 1) * wx - i * overlap_x
            ]
            indy = np.s_[
                j * (wy - overlap_y) + overlap_y // 2 : (j + 1) * wy - j * overlap_y
            ]

            reconstruction[:, indx, indy] = recon[:, overlap_x // 2 :, overlap_y // 2 :]
            k_iter += 1

    return reconstruction


def max_likelihood_reconstruction(
    dset: Array,
    psf: Array,
    stop: str = "fixed",
    max_iter: int = 100,
    threshold: float = 1e-3,
    rep_to_save: str | Sequence[int] | Array = "last",
    initialization: str = "flat",
    process: str = "gpu",
) -> tuple[Array, Array, Array, int]:
    """Reconstruct an object with maximum-likelihood iterations."""
    # Variables initialization taking into account if the data is spread along the axial dimension or not
    cpu_device = torch.device("cpu")
    data = torch.from_numpy(dset * 1.0).type(torch.float32).to(cpu_device)
    h = torch.from_numpy(psf * 1.0).type(torch.float32).to(cpu_device)

    oddeven_check_x = data.shape[0] % 2
    oddeven_check_y = data.shape[1] % 2
    check_x = False
    check_y = False

    data_check = data
    if oddeven_check_x == 0:
        check_x = True
        data_check = data_check[1:]
    if oddeven_check_y == 0:
        check_y = True
        data_check = data_check[:, 1:]

    Nz = h.shape[0]
    shape_data = data_check.shape
    Nx = shape_data[0]
    Ny = shape_data[1]
    shape_init = (Nz,) + shape_data[:-1]

    crop_pad_x = int((shape_data[0] - h.shape[1]) / 2)
    crop_pad_y = int((shape_data[1] - h.shape[2]) / 2)

    if crop_pad_x > 0 or crop_pad_y > 0:
        pad_array = np.zeros(2 * h.ndim, dtype="int")
        pad_array[2:4] = crop_pad_x
        pad_array[4:6] = crop_pad_y
        pad_widths = tuple(int(value) for value in np.flip(pad_array))

        h = torchpad.pad(h, pad_widths, "constant")
    elif crop_pad_x < 0 or crop_pad_y < 0:
        raise ValueError("The PSF is bigger than the image. Warning.")

    flip_ax = list(range(1, data_check.ndim))
    norm_ax = tuple(range(1, h.ndim))

    h = h.type(torch.float64).to(cpu_device)
    h = h / h.sum(keepdim=True, dim=norm_ax)
    ht = torch.flip(h, flip_ax)

    O = torch.ones(shape_init).to(cpu_device)
    b = torch.finfo(torch.float).eps  # assigning the error machine value

    # user can decide how to initialize the object, either with the photon flux of the input image or with
    # a flat initialization

    if initialization == "sum":
        S = data_check.sum(-1) / Nz
        for z in range(Nz):
            O[z, ...] = S
    elif initialization == "flat":
        O *= data_check.sum() / Nz / Nx / Ny
    else:
        raise ValueError("Initialization mode unknown.")

    k = 0

    counts = torch.zeros([2, max_iter + 1]).to(cpu_device)
    diff = torch.zeros([2, max_iter + 1]).to(cpu_device)
    tot = data.sum()

    if isinstance(rep_to_save, str) and rep_to_save == "all":
        size = [max_iter + 1] + list(O.shape)
        O_all = torch.empty(size).to(cpu_device)
    elif isinstance(rep_to_save, Iterable):
        l = len(rep_to_save)
        size_b = [l] + list(O.shape)
        O_all = torch.empty(size_b).to(cpu_device)

    pre_flag = True
    flag = True

    # Iterative reconstruction process
    if stop != "auto":
        total = max_iter
    else:
        total = None
    cont = 0

    padding = [h.size(d) for d in flip_ax]

    # PSF normalization axial plane wise, with respect to the flux of each plane
    h_fft = rfftn(h, dim=flip_ax, s=padding)
    del h
    ht_fft = rfftn(ht, dim=flip_ax, s=padding)
    del ht

    # Try to run on gpu
    if process == "gpu" and torch.cuda.is_available():
        try:
            gpu_device = torch.device("cuda:0")
            h_fft = h_fft.to(gpu_device)
            ht_fft = ht_fft.to(gpu_device)
            data_check = data_check.to(gpu_device)
            O = O.to(gpu_device)
            _ = amd_update_fft(data_check, O, h_fft, ht_fft, b)
            print("The algorithm will run on the GPU.")
        except RuntimeError as e:
            if "CUDA out of memory. " in str(e):
                print(
                    "Warning: The algorithms goes in Out Of Memory with CUDA. The algorithm will run on the CPU."
                )
                h_fft = h_fft.to(cpu_device)
                ht_fft = ht_fft.to(cpu_device)
                data_check = data_check.to(cpu_device)
                O = O.to(cpu_device)
            else:
                raise

    pbar = tqdm(total=total, desc="Progress", position=0)

    while flag:
        O_new = amd_update_fft(data_check, O, h_fft, ht_fft, b)
        pre_flag, flag, counts[:, k], diff[:, k] = amd_stop(
            O, O_new, pre_flag, flag, stop, max_iter, threshold, tot, Nz, k
        )

        if isinstance(rep_to_save, Iterable) and not isinstance(rep_to_save, str):
            if k in rep_to_save:
                O_all[cont, ...] = O.clone()
                cont += 1
        elif isinstance(rep_to_save, str) and rep_to_save == "all":
            O_all[k, ...] = O.clone()

        O = O_new.clone()

        k += 1
        pbar.update(1)
    pbar.close()

    if check_x:
        if rep_to_save == "last":
            pad_arr = np.zeros(2 * len(O.shape), dtype="int")
            pad_arr[-4] = 1
            O = torchpad.pad(O, tuple(pad_arr), "constant")
        else:
            pad_arr = np.zeros(2 * len(O_all.shape), dtype="int")
            pad_arr[-6] = 1
            O_all = torchpad.pad(O_all, tuple(pad_arr), "constant")
    if check_y:
        if rep_to_save == "last":
            pad_arr = np.zeros(2 * len(O.shape), dtype="int")
            pad_arr[-6] = 1
            O = torchpad.pad(O, tuple(pad_arr), "constant")
        else:
            pad_arr = np.zeros(2 * len(O_all.shape), dtype="int")
            pad_arr[-8] = 1
            O_all = torchpad.pad(O_all, tuple(pad_arr), "constant")

    if isinstance(rep_to_save, str) and rep_to_save == "last":
        obj = O.detach().cpu().numpy()
    else:
        obj = O_all.detach().cpu().numpy()

    counts_array = counts[:, :k].detach().cpu().numpy()
    diff_array = diff[:, :k].detach().cpu().numpy()

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    return obj, counts_array, diff_array, k


def data_driven_reconstruction(
    dset: Array,
    gridPar: sim.GridParameters,
    exPar: sim.simSettings,
    emPar: sim.simSettings,
    rep_to_save: str = "last",
    initialization: str = "flat",
    max_iter: int = 100,
    stop: str = "fixed",
    threshold: float = 1e-3,
    downsample: bool = True,
    z_out_of_focus: str | float = "ToFind",
) -> tuple[Array, Array, Array, int, Array]:
    """Estimate a PSF and reconstruct the measured image."""
    psf, _, _ = svr.psf_estimator_from_data(
        dset,
        exPar,
        emPar,
        gridPar,
        downsample=downsample,
        z_out_of_focus=z_out_of_focus,
    )

    obj, counts, diff, k = max_likelihood_reconstruction(
        dset,
        psf,
        stop=stop,
        max_iter=max_iter,
        threshold=threshold,
        rep_to_save=rep_to_save,
        initialization=initialization,
    )

    return obj, counts, diff, k, psf
