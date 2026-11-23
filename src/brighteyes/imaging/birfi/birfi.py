# Origin: VicidominiLab/birfi, src/birfi/birfi.py at 4ea6c4d.
"""Estimate instrument response functions from time-series data."""

from __future__ import annotations

from typing import TypedDict, cast

import torch
from scipy.signal import savgol_filter
from torch.fft import fftn

from .utils import (
    estimate_lifetime,
    generate_truncated_exponential,
    median_filter,
    partial_convolution,
    plot_dataset,
)


class _FitParameters(TypedDict):
    A: torch.Tensor
    C: torch.Tensor
    k: float


class Birfi:
    """Estimate instrument response functions from time-series data."""

    def __init__(
        self,
        data: torch.Tensor,
        dt: float = 1.0,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the estimator with time-series data."""
        self.device: torch.device | str
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.data: torch.Tensor
        data = torch.as_tensor(data, dtype=torch.float32)
        if data.dim() == 1:
            self.data = data.unsqueeze(1).clone().to(self.device)
        elif data.dim() == 2:
            self.data = data.clone().to(self.device)
        elif data.dim() > 2:
            raise ValueError("data must be 1D or 2D tensor")

        self.dt = dt
        self.time = torch.arange(self.data.shape[0], device=self.device) * dt

        self.num_samples, self.num_channels = self.data.shape
        self.t0: torch.Tensor | None = None
        self.t1: torch.Tensor | None = None
        self.params: _FitParameters | None = None
        self.data_fit: torch.Tensor | None = None
        self.kernel: torch.Tensor | None = None
        self.irf: torch.Tensor | None = None

    def find_t0_t1(
        self,
        window_length: int = 11,
        polyorder: int = 3,
        persistence: int = 5,
        threshold: float = 0.05,
    ) -> None:
        """Estimate decay start and end indices for each channel."""
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd.")

        t0s, t1s = [], []
        y_range = self.data.max(dim=0).values - self.data.min(dim=0).values

        for c in range(self.num_channels):
            # Move data to CPU for scipy (savgol_filter), then back to device
            y = self.data[:, c].detach().cpu().numpy()
            dy = savgol_filter(
                y,
                window_length=window_length,
                polyorder=polyorder,
                deriv=1,
                delta=self.dt,
            )
            dy = torch.tensor(dy, dtype=self.data.dtype, device=self.device)

            # t0: global minimum of derivative
            t0 = int(torch.argmin(dy))

            # t1: first point after t0 with persistent positive derivative
            t1 = len(dy) - 1  # fallback to end
            for i in range(t0 + 1, len(dy) - persistence):
                avg_diff = dy[i : i + persistence].mean()
                amplitude = torch.clamp(
                    self.data[i + persistence, c] - self.data[..., c].min(), 0
                ).item()
                if avg_diff > 0 and amplitude > threshold * y_range[c].item():
                    t1 = i
                    break

            t0s.append(t0)
            t1s.append(t1)

        self.t0 = torch.tensor(t0s, device=self.device)
        self.t1 = torch.tensor(t1s, device=self.device)

    def fit_exponential(
        self, offset: int = 0, lr: float = 1e-2, steps: int = 1000
    ) -> None:
        """Fit truncated exponential curves to the channel data."""
        if self.t0 is None or self.t1 is None:
            raise RuntimeError("Run find_t0_t1 first.")

        device, dtype = self.device, self.data.dtype
        A = (
            (self.data.max(dim=0).values - self.data.min(dim=0).values)
            .clone()
            .detach()
            .to(device)
            .requires_grad_(True)
        )
        Cparam = (
            self.data.min(dim=0).values.clone().detach().to(device).requires_grad_(True)
        )

        cc = self.num_channels // 2  # central channel index
        tau = estimate_lifetime(self.time, self.data[..., cc], self.t0[cc], self.t1[cc])
        k = (1.0 / tau).clone().detach().to(device).requires_grad_(True)

        opt = torch.optim.Adam([A, Cparam, k], lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            loss = torch.tensor(0.0, device=device, dtype=dtype)
            for c in range(self.num_channels):
                y = self.data[self.t0[c] + offset : self.t1[c] + 1, c]
                x = torch.arange(len(y), device=device, dtype=dtype) * self.dt
                y_pred = A[c] * torch.exp(-k * x) + Cparam[c]
                loss = loss + torch.mean((y - y_pred) ** 2)
            loss = loss / self.num_channels
            torch.autograd.backward(loss)
            opt.step()

        self.params = {"A": A.detach(), "C": Cparam.detach(), "k": k.detach().item()}

    def generate_data_fit(self) -> None:
        """Generate fitted curves for all channels."""
        if self.params is None:
            raise RuntimeError("Run fit_exponential first.")

        exp_curves = torch.zeros_like(self.data, device=self.device)
        t0 = cast(torch.Tensor, self.t0)

        for c in range(self.num_channels):
            params: dict[str, float | torch.Tensor] = {
                "A": self.params["A"][c],
                "C": self.params["C"][c],
                "k": self.params["k"],
                "t0": int(t0[c]) * self.dt,
            }
            exp_curves[:, c] = generate_truncated_exponential(self.time, params).to(
                self.device
            )

        self.data_fit = exp_curves

    def generate_kernel(self) -> None:
        """Build a normalized kernel from the fitted decay rate."""
        if self.params is None:
            raise RuntimeError("Run fit_exponential first.")

        params = {
            "A": 1,
            "C": 0,
            "k": self.params["k"],
            "t0": 0,
        }

        exp_curve = generate_truncated_exponential(self.time, params).to(self.device)
        exp_curve = torch.clamp(exp_curve, min=0)  # enforce positivity
        exp_curve /= exp_curve.sum()  # normalize kernel
        self.kernel = exp_curve

    def richardson_lucy_deconvolution(
        self, iterations: int = 30, eps: float = 1e-4, regularization: int = 3
    ) -> None:
        """Estimate channel responses with Richardson-Lucy deconvolution."""
        if self.kernel is None:
            raise RuntimeError(
                "Run generate_kernel() first or provide a convolution kernel manually."
            )

        # initialize output tensor
        x_est = torch.ones_like(self.data, device=self.device)  # shape (time, channel)

        # load deconvolution kernel
        kernel = self.kernel.clone().to(self.device)  # shape (time,)
        kernel_t = (
            self.kernel.clone().flip(0).to(self.device)
        )  # time-reversed kernel, shape (time,)

        kernel = fftn(kernel, dim=0)  # FT of kernel, shape (time,)
        kernel_t = fftn(kernel_t, dim=0)  # FT of time-reversed kernel, shape (time,)

        # Subtract offset
        y = self.data.detach().clone().to(
            dtype=torch.float32, device=self.device
        ) - cast(_FitParameters, self.params)["C"].unsqueeze(0).to(self.device)
        y = torch.clamp(y, min=0)

        # RL deconvolution
        for _ in range(iterations):
            conv = partial_convolution(
                x_est, kernel, dim1="xc", dim2="x", axis="x", fourier=(False, True)
            )
            conv = torch.clamp(conv, min=eps)  # avoid div by 0
            relative_blur = y / conv
            correction = partial_convolution(
                relative_blur,
                kernel_t,
                dim1="xc",
                dim2="x",
                axis="x",
                fourier=(False, True),
            )
            x_est = x_est * correction
            x_est = torch.clamp(x_est, min=0)  # enforce positivity
            if regularization > 1:
                x_est = median_filter(
                    x_est, window_size=regularization, dims=[0], mode="replicate"
                )  # temporal median filter

        self.irf = x_est

    def run(
        self,
        lr: float = 1e-2,
        steps: int = 1000,
        rl_iterations: int = 30,
        regularization: int = 3,
        window_length: int = 11,
        polyorder: int = 3,
        persistence: int = 5,
        threshold: float = 0.05,
    ) -> torch.Tensor | None:
        """Run the instrument response estimation pipeline."""
        self.find_t0_t1(
            window_length=window_length,
            polyorder=polyorder,
            persistence=persistence,
            threshold=threshold,
        )
        self.fit_exponential(lr=lr, steps=steps)
        self.generate_data_fit()
        self.generate_kernel()
        self.richardson_lucy_deconvolution(
            iterations=rl_iterations, regularization=regularization
        )

        return self.irf

    def plot_raw_and_fit(self) -> None:
        """Plot measured data, fitted curves, and decay bounds."""
        if self.data_fit is None:
            raise RuntimeError("Run generate_data_fit() first.")

        # Move tensors to CPU for plotting
        time = self.time.cpu().numpy()
        raw = self.data.cpu().numpy()
        fit = self.data_fit.cpu().numpy()
        t0 = cast(torch.Tensor, self.t0)
        t1 = cast(torch.Tensor, self.t1)

        # First, plot raw data as scatter (points)
        fig, ax = plot_dataset(time, raw, color="k", linestyle="none", marker=".")

        # Overlay full fit curve
        fig, ax = plot_dataset(time, fit, color="r", linestyle="-", fig=fig, ax=ax)

        # Draw vertical dashed lines at t0 and t1
        for c in range(self.num_channels):
            t0_val = self.time[int(t0[c])].item()
            t1_val = self.time[int(t1[c])].item()
            ax[c].axvline(t0_val, color="grey", linestyle="--", alpha=0.5)
            ax[c].axvline(t1_val, color="grey", linestyle="--", alpha=0.5)

        fig.legend(
            ["Raw", "Fit", "Fitting interval"],
            loc="upper right",
            bbox_to_anchor=(0.98, 0.95),
        )

    def plot_forward_model(self) -> None:
        """Plot measured data against the convolved response model."""
        if self.irf is None:
            raise RuntimeError("Run richardson_lucy_deconvolution() first.")

        time = self.time.cpu().numpy()
        raw = self.data.cpu().numpy()

        forward = partial_convolution(
            self.irf,
            cast(torch.Tensor, self.kernel),
            dim1="xc",
            dim2="x",
            axis="x",
            fourier=(False, False),
        )
        forward += (
            cast(_FitParameters, self.params)["C"].unsqueeze(0).to(forward.device)
        )  # Add offset on same device
        forward_array = forward.cpu().numpy()  # Move to CPU only for plotting

        # First, plot raw data as scatter
        fig, ax = plot_dataset(time, raw, color="k", linestyle="none", marker=".")
        # Then add forward model as line
        fig, ax = plot_dataset(
            time, forward_array, color="g", linestyle="-", fig=fig, ax=ax
        )

        fig.legend(
            ["Measured", "IRF ⊗ Exp"], loc="upper right", bbox_to_anchor=(0.95, 0.95)
        )
