from __future__ import annotations

import warnings
from os import PathLike

import numpy as np
import torch


def autocorrelation(
    series: torch.Tensor | np.ndarray,
    max_lag: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate the autocorrelation of one or more independent chains.

    A one-dimensional input is treated as a single chain.  For a two-dimensional
    input, rows are independent chains and the last axis is Monte Carlo time.
    Each chain is centred separately before their autocovariances are pooled.

    The FFT estimator uses the full available history by default.  In
    particular, it does not impose a fixed lag cutoff, since that can severely
    underestimate the autocorrelation time of a slowly mixing chain.
    """
    values = torch.as_tensor(series)
    if values.is_complex():
        raise ValueError("series must be real-valued")
    values = values.to(dtype=torch.float64)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    elif values.ndim != 2:
        raise ValueError(
            "series must be a non-empty one- or two-dimensional array")
    if values.shape[-1] == 0 or values.shape[0] == 0:
        raise ValueError(
            "series must be a non-empty one- or two-dimensional array")
    if not torch.all(torch.isfinite(values)):
        raise ValueError("series must contain only finite values")

    if max_lag is not None and (
        isinstance(max_lag, bool) or not isinstance(max_lag, (int, np.integer))
    ):
        raise TypeError("max_lag must be an integer or None")

    values = values - values.mean(dim=-1, keepdim=True)
    n = values.shape[-1]
    if max_lag is None:
        max_lag = n - 1
    max_lag = min(max_lag, n - 1)
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")

    # Zero-padding prevents the FFT's circular convolution from wrapping the
    # end of a chain onto its beginning.  Pool sums, rather than normalized
    # per-chain ACFs, so chains with negligible variance do not dominate.
    fft_size = 1 << (2 * n - 1).bit_length()
    transform = torch.fft.rfft(values, n=fft_size, dim=-1)
    autocovariance = torch.fft.irfft(
        transform * transform.conj(), n=fft_size, dim=-1
    )[..., : max_lag + 1]
    autocovariance = autocovariance.sum(dim=0) / (values.shape[0] * n)

    if autocovariance[0] <= 0:
        return (
            torch.arange(max_lag + 1, device=values.device),
            torch.full(
                (max_lag + 1,),
                torch.nan,
                dtype=values.dtype,
                device=values.device,
            ),
        )
    acf = autocovariance / autocovariance[0]
    return torch.arange(max_lag + 1, device=values.device), acf


def integrated_autocorrelation_time(acf: torch.Tensor | np.ndarray) -> float:
    """Estimate ``1/2 + sum(t > 0, rho[t])`` with a stable window.

    Geyer's initial positive, monotone sequence is used: adjacent ACF values
    are paired, integration stops at the first nonpositive pair, and the
    retained pair sums are made monotonically decreasing.  Pairing avoids the
    large downward bias caused by stopping at one noisy negative ACF value.

    The returned convention is the one commonly used in lattice calculations;
    twice this value is the statistical inefficiency (and is 1 for IID data).
    """
    rho = torch.as_tensor(acf, dtype=torch.float64)
    if rho.ndim != 1 or len(rho) == 0:
        raise ValueError("acf must be a non-empty one-dimensional array")
    if not torch.all(torch.isfinite(rho)):
        raise ValueError("acf must contain only finite values")
    if rho[0] <= 0:
        raise ValueError("acf[0] must be positive")
    rho = rho / rho[0]

    # Gamma_k = rho[2k] + rho[2k + 1].  Drop a final unpaired lag.
    n_paired = 2 * (len(rho) // 2)
    if n_paired == 0:
        return 0.5
    pair_sums = rho[:n_paired].reshape(-1, 2).sum(dim=1)
    nonpositive = torch.nonzero(pair_sums <= 0, as_tuple=True)[0]
    if len(nonpositive):
        pair_sums = pair_sums[: int(nonpositive[0].item())]
    if len(pair_sums) == 0:
        return 0.5

    pair_sums = torch.cummin(pair_sums, dim=0).values
    tau_int = -0.5 + pair_sums.sum().item()
    return max(0.5, tau_int)


def _xy_magnetization(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return real and imaginary magnetization without a full complex copy."""
    if x.ndim < 3:
        raise ValueError("x must have shape (chains, samples, *lattice_shape)")
    if not x.is_floating_point():
        raise ValueError("XY configurations must be floating-point angles")

    shape = x.shape[:2]
    real = torch.empty(shape, dtype=x.dtype, device=x.device)
    imag = torch.empty_like(real)
    # Processing one chain at a time bounds temporary memory for large 3D
    # ensembles, for which a full complex-valued copy can require many GB.
    for chain in range(shape[0]):
        angles = x[chain].flatten(start_dim=1)
        real[chain] = angles.cos().mean(dim=1)
        imag[chain] = angles.sin().mean(dim=1)
    return real, imag


def load_ensemble(
    filename: str | PathLike[str],
    *,
    decorrelation_factor: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Load and thin an XY ensemble using its slowest standard diagnostic.

    The saved energy and the real, imaginary, and absolute magnetization are
    checked.  Energy alone can mix much faster than the long-wavelength spin
    modes, especially near criticality.  ``decorrelation_factor`` multiplies
    the statistical inefficiency ``g = 2 * tau_int``.  The default samples once
    per ``g``; values below 1 retain more, correlated configurations, while
    values above 1 demand stronger decorrelation.
    """
    if not np.isfinite(decorrelation_factor) or decorrelation_factor <= 0:
        raise ValueError("decorrelation_factor must be positive and finite")

    data = np.load(filename, allow_pickle=True).item()
    x = torch.as_tensor(data["x"])
    beta = float(data["beta"])
    energies = torch.as_tensor(data["energies"], dtype=torch.float64)
    if x.ndim < 3:
        raise ValueError("x must have shape (chains, samples, *lattice_shape)")
    if energies.ndim not in {1, 2}:
        raise ValueError(
            "energies must have shape (samples,) or (chains, samples)")
    if energies.shape[-1] != x.shape[1]:
        raise ValueError(
            "energies and configurations have different time lengths")
    if energies.ndim == 2 and energies.shape[0] != x.shape[0]:
        raise ValueError(
            "energies and configurations have different chain counts")

    magnetization_real, magnetization_imag = _xy_magnetization(x)
    diagnostics = {
        "energy": energies,
        "Re(magnetization)": magnetization_real,
        "Im(magnetization)": magnetization_imag,
        "|magnetization|": torch.hypot(magnetization_real, magnetization_imag),
    }
    autocorrelation_times: dict[str, float] = {}
    for name, values in diagnostics.items():
        _, acf = autocorrelation(values)
        try:
            autocorrelation_times[name] = integrated_autocorrelation_time(acf)
        except ValueError as error:
            warnings.warn(f"Skipping constant {name} diagnostic: {
                          error}", stacklevel=2)

    if not autocorrelation_times:
        raise ValueError(
            "cannot estimate autocorrelation from constant diagnostics")
    limiting_diagnostic, tau_int = max(
        autocorrelation_times.items(), key=lambda item: item[1]
    )

    # In the lattice convention used by integrated_autocorrelation_time,
    # g = 2*tau_int and N_eff is approximately N/g.  Thinning by much more than
    # g discards configurations without increasing the available information.
    statistical_inefficiency = 2 * tau_int
    step = max(1, int(np.ceil(decorrelation_factor * statistical_inefficiency)))
    n_steps = x.shape[1]
    effective_sample_size = x.shape[0] * n_steps / statistical_inefficiency
    if n_steps < 50 * statistical_inefficiency:
        warnings.warn(
            "The chain is shorter than 50 autocorrelation times; the thinning "
            "estimate may be unreliable.",
            stacklevel=2,
        )

    x_eff = x[:, ::step]
    x_eff = x_eff.reshape(-1, *x.shape[2:])

    print(
        "\n"
        f"beta={beta:.4g}, "
        f"fields.shape={tuple(x_eff.shape)}, "
        f"tau_int={tau_int:.3f}, "
        f"g={statistical_inefficiency:.3f}, "
        f"ESS~{effective_sample_size:.0f}, "
        f"stride={step}, "
        f"limited_by={limiting_diagnostic}"
    )

    return x_eff, beta
