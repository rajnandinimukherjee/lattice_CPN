"""Plot CP(2) radial deformation profiles for four OPF strategies."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from XY_utils import autocorrelation, integrated_autocorrelation_time
from CPN_utils import ComplexPair
from actions import CPnGaugeAction
from cpn_opf.model import CPnHomogeneousDeformations
from cpn_opf.optimization import (
    optimize_field_scale,
    solve_approx2_penalty_field,
    solve_approx3_penalty_field,
)
from cpn_opf.persistence import training_savename
from observables import cpn_one_pt_fn
from plot_settings import fig_width, latex_font_params


plt.rcParams.update(latex_font_params)
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    }
)


PROFILE_STYLES = {
    "variance": {
        "color": "tab:blue",
        "marker": "o",
        "label": "direct variance",
    },
    "penalty": {
        "color": "tab:orange",
        "marker": "s",
        "label": "Approx. 1 (weak corr.)",
    },
    "approx2": {
        "color": "tab:green",
        "marker": "^",
        "label": "Approx. 2 (small fluc.)",
    },
    "approx3": {
        "color": "tab:red",
        "marker": "D",
        "label": "Approx. 3 (smooth deforms)",
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Build the CP(2) radial-profile CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=3.3)
    parser.add_argument("--cp-n", type=int, default=2)
    parser.add_argument("--lattice-size", type=int, default=64)
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument("--j", type=int, default=1)
    parser.add_argument("--mask-radius", type=int, default=6)
    parser.add_argument(
        "--ensemble",
        type=Path,
        default=None,
        help="split-real CP(N) gauge ensemble; inferred from beta by default",
    )
    parser.add_argument(
        "--training-data-directory", type=Path, default=Path("training_data")
    )
    parser.add_argument(
        "--decorrelation",
        type=float,
        default=1.0,
        help="multiply 2*tau_int for deterministic ensemble thinning",
    )
    parser.add_argument(
        "--max-configurations",
        type=int,
        default=1280,
        help="maximum number of decorrelated configurations for alpha fits",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="configuration batch size for OPF variance evaluation",
    )
    parser.add_argument(
        "--alpha-bounds",
        nargs=2,
        type=float,
        default=(-5.0, 5.0),
        metavar=("LOWER", "UPPER"),
    )
    parser.add_argument("--alpha-grid-points", type=int, default=41)
    parser.add_argument("--alpha-xatol", type=float, default=1e-4)
    parser.add_argument(
        "--panel-scale",
        type=float,
        default=1.0,
        help="scale plot text, lines, and markers for smaller panel layouts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="profile PDF path; a PNG and JSON report are saved alongside it",
    )
    parser.add_argument(
        "--fit-results",
        type=Path,
        default=None,
        help="reusable NPZ cache; defaults to a sibling of --output",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="ignore an existing fit cache and redo both penalty and alpha fits",
    )
    return parser


def _diagnostic_autocorrelation(
    values: np.ndarray,
    name: str,
) -> tuple[float, str] | None:
    """Estimate an integrated autocorrelation time, skipping constants."""
    try:
        _, acf = autocorrelation(values)
        return integrated_autocorrelation_time(acf), name
    except (ValueError, RuntimeError):
        return None


def _load_profile_ensemble(
    filename: Path,
    *,
    cp_n: int,
    ndim: int,
    lattice_size: int,
    i: int,
    j: int,
    decorrelation: float,
    max_configurations: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Load a thinned split-real ensemble without copying its full payload.

    The local 64^2 ensemble is a pickled mapping whose arrays occupy several
    gigabytes.  Diagnostics are reduced one chain at a time; only the selected
    decorrelated configurations are copied into the model's interleaved-real
    format.
    """
    if not np.isfinite(decorrelation) or decorrelation <= 0.0:
        raise ValueError("decorrelation must be positive and finite")
    if max_configurations < 1:
        raise ValueError("max_configurations must be positive")
    loaded = np.load(filename, allow_pickle=True)
    if loaded.shape != () or not isinstance(loaded.item(), dict):
        raise ValueError(f"expected a mapping payload in {filename}")
    payload = loaded.item()
    if "x" not in payload or "lam" not in payload:
        raise KeyError("the profile script expects split-real x and lam arrays")
    raw_fields = np.asarray(payload["x"])
    raw_links = np.asarray(payload["lam"])
    del loaded

    channels = cp_n + 1
    expected_field_tail = (2 * channels, lattice_size, lattice_size)
    expected_link_tail = (ndim, lattice_size, lattice_size)
    if raw_fields.ndim != 5 or tuple(raw_fields.shape[2:]) != expected_field_tail:
        raise ValueError(
            f"x must have shape (chains, time, {expected_field_tail}), "
            f"got {raw_fields.shape}"
        )
    if raw_links.ndim != 5 or tuple(raw_links.shape[2:]) != expected_link_tail:
        raise ValueError(
            f"lam must have shape (chains, time, {expected_link_tail}), "
            f"got {raw_links.shape}"
        )
    if raw_fields.shape[:2] != raw_links.shape[:2]:
        raise ValueError("field and link chain/time dimensions do not match")
    chains, time_extent = raw_fields.shape[:2]

    diagnostics: list[tuple[float, str]] = []
    raw_energy = payload.get("energies", payload.get("energy"))
    if raw_energy is not None:
        energy = np.asarray(raw_energy, dtype=np.float64)
        if energy.ndim in (1, 2) and energy.shape[-1] == time_extent:
            estimate = _diagnostic_autocorrelation(energy, "energy")
            if estimate is not None:
                diagnostics.append(estimate)

    density_series = np.empty((channels, chains, time_extent), dtype=np.float64)
    off_diagonal_series = np.empty((chains, time_extent), dtype=np.float64)
    for chain in range(chains):
        chain_fields = raw_fields[chain]
        for channel in range(channels):
            real = chain_fields[:, channel]
            imag = chain_fields[:, channels + channel]
            density_series[channel, chain] = np.mean(
                real * real + imag * imag, axis=(-2, -1)
            )
        real_i = chain_fields[:, i]
        real_j = chain_fields[:, j]
        imag_i = chain_fields[:, channels + i]
        imag_j = chain_fields[:, channels + j]
        off_diagonal_series[chain] = np.mean(
            real_i * real_j + imag_i * imag_j, axis=(-2, -1)
        )

    for channel in range(channels):
        estimate = _diagnostic_autocorrelation(
            density_series[channel], f"P_{channel}{channel}"
        )
        if estimate is not None:
            diagnostics.append(estimate)
    estimate = _diagnostic_autocorrelation(off_diagonal_series, f"Re(P_{i}{j})")
    if estimate is not None:
        diagnostics.append(estimate)
    if not diagnostics:
        raise ValueError("could not estimate ensemble autocorrelation")
    limiting_tau, limiting_name = max(diagnostics, key=lambda item: item[0])
    stride = max(1, int(np.ceil(decorrelation * 2.0 * limiting_tau)))

    thinned_time_indices = np.arange(0, time_extent, stride, dtype=np.int64)
    max_per_chain = max(1, max_configurations // chains)
    n_per_chain = min(len(thinned_time_indices), max_per_chain)
    if n_per_chain < len(thinned_time_indices):
        selection = np.linspace(
            0, len(thinned_time_indices) - 1, n_per_chain, dtype=np.int64
        )
        thinned_time_indices = thinned_time_indices[selection]
    selected_fields = raw_fields[:, thinned_time_indices]
    selected_links = raw_links[:, thinned_time_indices]

    field_channels_last = selected_fields.transpose(0, 1, 3, 4, 2)
    interleaved = np.empty(
        (*field_channels_last.shape[:-1], 2 * channels), dtype=np.float32
    )
    interleaved[..., 0::2] = field_channels_last[..., :channels]
    interleaved[..., 1::2] = field_channels_last[..., channels:]
    interleaved = np.ascontiguousarray(
        interleaved.reshape(-1, lattice_size, lattice_size, 2 * channels)
    )
    links_channels_last = selected_links.transpose(0, 1, 3, 4, 2)
    links_channels_last = np.ascontiguousarray(
        links_channels_last.reshape(-1, lattice_size, lattice_size, ndim),
        dtype=np.float32,
    )
    n_used = len(interleaved)
    if n_used < 2:
        raise ValueError("at least two decorrelated configurations are required")

    ess = chains * time_extent / (2.0 * limiting_tau)
    effective_stride = (
        int(np.median(np.diff(thinned_time_indices)))
        if len(thinned_time_indices) > 1
        else stride
    )
    del payload, raw_fields, raw_links, selected_fields, selected_links
    del field_channels_last, density_series
    del off_diagonal_series
    del chain_fields, real, imag, real_i, real_j, imag_i, imag_j
    gc.collect()
    print(
        f"ensemble thinning: tau_int={limiting_tau:.3f} ({limiting_name}), "
        f"stride={stride} (effective {effective_stride}), "
        f"selected={n_used}/{chains * time_extent}, "
        f"ESS~{ess:.0f}"
    )
    return (
        torch.from_numpy(interleaved),
        torch.from_numpy(links_channels_last),
        {
            "tau_int": float(limiting_tau),
            "stride": int(stride),
            "effective_stride": int(effective_stride),
            "n_used": int(n_used),
            "effective_sample_size": float(ess),
        },
    )


class _BatchedCPnVariance:
    """Evaluate full OPF variances in configuration chunks."""

    def __init__(
        self,
        fields: torch.Tensor,
        links: torch.Tensor,
        action: CPnGaugeAction,
        model: CPnHomogeneousDeformations,
        *,
        i: int,
        j: int,
        source: tuple[int, ...],
        batch_size: int,
    ) -> None:
        self.fields = fields
        self.links = links
        self.action = action
        self.model = model
        self.i = i
        self.j = j
        self.source = source
        self.batch_size = batch_size
        self._chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        undeformed = []
        with torch.no_grad():
            for start in range(0, len(fields), batch_size):
                stop = min(start + batch_size, len(fields))
                x_chunk = fields[start:stop]
                link_chunk = links[start:stop]
                base_action = self.action((x_chunk, link_chunk))
                bare = cpn_one_pt_fn(x_chunk, i, j, source)
                assert hasattr(bare, "real")
                undeformed.append(bare.real)
                self._chunks.append((x_chunk, link_chunk, base_action))
        self.initial_var = torch.cat(undeformed).var().item()
        if not np.isfinite(self.initial_var) or self.initial_var <= 0.0:
            raise ValueError("undeformed CP(N) OPF variance must be positive")

    def __call__(
        self,
        alphas: torch.Tensor | None = None,
        *,
        sampletype: str = "all",
    ) -> torch.Tensor:
        """Return the sample variance at one profile scale."""
        if sampletype != "all":
            raise ValueError("the radial profile evaluator uses all selected samples")
        if alphas is None:
            values = [
                cpn_one_pt_fn(chunk, self.i, self.j, self.source).real
                for chunk, _, _ in self._chunks
            ]
            return torch.cat(values).var()

        transformed: list[tuple[ComplexPair, ComplexPair, torch.Tensor]] = []
        with torch.no_grad():
            for x_chunk, link_chunk, base_action in self._chunks:
                deformed = self.model.deform(x_chunk, alphas)
                action_difference = -self.action(
                    (deformed, link_chunk)
                ) + base_action
                reweighting = action_difference.exp()
                jacobian = self.model.detJac(x_chunk, alphas)
                jacobian = jacobian.reshape(len(x_chunk), -1).prod(dim=-1)
                bare = cpn_one_pt_fn(deformed, self.i, self.j, self.source)
                transformed.append((bare, reweighting, jacobian))

        # CPnOnePtFn subtracts the ensemble mean of the continued bare OPF
        # before multiplying by the action and Jacobian weights. Centering the
        # final weighted estimator would optimize a different variance.
        bare_real = torch.cat([bare.real for bare, _, _ in transformed])
        bare_imag = torch.cat([bare.imag for bare, _, _ in transformed])
        mean_real = bare_real.mean()
        mean_imag = bare_imag.mean()
        estimates = [
            (
                ComplexPair(bare.real - mean_real, bare.imag - mean_imag)
                * reweighting
                * jacobian
            ).real
            for bare, reweighting, jacobian in transformed
        ]
        return torch.cat(estimates).var()


def _load_saved_field(
    filename: Path,
    *,
    cp_n: int,
    beta: float,
    lattice_shape: tuple[int, ...],
    i: int,
    j: int,
) -> tuple[np.ndarray, dict]:
    """Read a saved training field and check it matches this comparison."""
    if not filename.exists():
        raise FileNotFoundError(f"missing saved training profile: {filename}")
    data = np.load(filename, allow_pickle=True).item()
    metadata = data.get("metadata", {})
    expected = {
        "cp_n": cp_n,
        "ndim": len(lattice_shape),
        "beta": beta,
        "lattice_size": lattice_shape,
        "i": i,
        "j": j,
    }
    for key, value in expected.items():
        stored = metadata.get(key)
        if key == "beta":
            matches = stored is not None and np.isclose(float(stored), value)
        elif key == "lattice_size":
            matches = tuple(stored or ()) == value
        else:
            matches = stored == value
        if not matches:
            raise ValueError(f"{filename} has incompatible {key}: {stored}")
    field = np.asarray(data.get("scaled_field", data["learned_field"]), dtype=float)
    if field.shape != (*lattice_shape, cp_n + 1) or not np.all(np.isfinite(field)):
        raise ValueError(f"{filename} contains an invalid CP(N) field")
    return field, data


def _file_signature(filename: Path) -> dict[str, str | int]:
    """Return a small source-file fingerprint for cache invalidation."""
    stat = filename.stat()
    return {
        "path": str(filename.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _fit_signature(
    args: argparse.Namespace,
    *,
    ensemble: Path,
    variance_path: Path,
    penalty_path: Path,
) -> dict:
    """Describe every input that can change the numerical fit results."""
    return {
        "cp_n": int(args.cp_n),
        "ndim": 2,
        "beta": float(args.beta),
        "lattice_size": [int(args.lattice_size)] * 2,
        "source": [0, 0],
        "i": int(args.i),
        "j": int(args.j),
        "mask_radius": int(args.mask_radius),
        "decorrelation": float(args.decorrelation),
        "max_configurations": int(args.max_configurations),
        "alpha_bounds": [float(value) for value in args.alpha_bounds],
        "alpha_grid_points": int(args.alpha_grid_points),
        "alpha_xatol": float(args.alpha_xatol),
        "estimator_centering": "global bare-operator mean before weights",
        "ensemble": _file_signature(ensemble),
        "variance_training": _file_signature(variance_path),
        "penalty_training": _file_signature(penalty_path),
    }


def _load_fit_cache(filename: Path, signature: dict) -> dict | None:
    """Load a matching fit cache, returning ``None`` when it is stale."""
    if not filename.exists():
        return None
    with np.load(filename, allow_pickle=False) as cached:
        saved_signature = json.loads(str(cached["fit_signature"].item()))
        if saved_signature != signature:
            return None
        return {
            "variance_field": np.asarray(cached["variance_field"]),
            "penalty_field": np.asarray(cached["penalty_field"]),
            "approx2_field": np.asarray(cached["approx2_field"]),
            "approx3_field": np.asarray(cached["approx3_field"]),
            "alphas": np.asarray(cached["alphas"], dtype=float),
            "variance_ratios": np.asarray(cached["variance_ratios"], dtype=float),
            "initial_variance": float(cached["initial_variance"].item()),
            "ensemble_info": json.loads(str(cached["ensemble_info"].item())),
            "approx2_diagnostics": json.loads(
                str(cached["approx2_diagnostics"].item())
            ),
            "approx3_diagnostics": json.loads(
                str(cached["approx3_diagnostics"].item())
            ),
        }


def _save_fit_cache(
    filename: Path,
    *,
    signature: dict,
    fields: dict[str, np.ndarray],
    alphas: tuple[float, float, float, float],
    variance_ratios: tuple[float, float, float, float],
    initial_variance: float,
    ensemble_info: dict,
    approx2_diagnostics: dict,
    approx3_diagnostics: dict,
) -> None:
    """Persist vector fields and fit summaries for future plot-only runs."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        filename,
        fit_signature=json.dumps(signature, sort_keys=True),
        variance_field=fields["variance"],
        penalty_field=fields["penalty"],
        approx2_field=fields["approx2"],
        approx3_field=fields["approx3"],
        alphas=np.asarray(alphas, dtype=float),
        variance_ratios=np.asarray(variance_ratios, dtype=float),
        initial_variance=float(initial_variance),
        ensemble_info=json.dumps(ensemble_info, sort_keys=True),
        approx2_diagnostics=json.dumps(approx2_diagnostics, sort_keys=True),
        approx3_diagnostics=json.dumps(approx3_diagnostics, sort_keys=True),
    )


def _radial_average(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average a scalar field over periodic Euclidean distance shells."""
    coordinates = np.indices(field.shape, dtype=np.int64)
    lengths = np.asarray(field.shape, dtype=np.int64).reshape(
        -1, *([1] * field.ndim)
    )
    displacement = np.minimum(coordinates, lengths - coordinates)
    radius_squared = np.square(displacement).sum(axis=0)
    shells = np.unique(radius_squared)
    radii = np.sqrt(shells.astype(float))
    means = np.asarray(
        [field[radius_squared == shell].mean() for shell in shells], dtype=float
    )
    return radii, means


def _add_variance_ratio_inset(
    axis: plt.Axes,
    variance_ratios: dict[str, float],
    panel_scale: float,
) -> None:
    """Add a compact comparison of the four relative variances."""
    methods = tuple(PROFILE_STYLES)
    values = np.asarray([variance_ratios[method] for method in methods])
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("variance ratios must be finite and non-negative")

    inset = axis.inset_axes([0.73, 0.60, 0.25, 0.37])
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.97)
    positions = np.arange(len(methods))
    inset.bar(
        positions,
        values,
        color=[PROFILE_STYLES[method]["color"] for method in methods],
        width=0.68,
        linewidth=0.0,
        zorder=2,
    )
    ymax = max(1.12, float(values.max(initial=0.0)) * 1.15 + 0.08)
    inset.set_ylim(0.0, ymax)
    inset.set_xlim(-0.55, len(methods) - 0.45)
    inset.set_xticks([])
    inset.set_yticks([0.0, 1.0])
    inset.set_ylabel(
        r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$",
        rotation=90,
        fontsize=9.0 * panel_scale,
        labelpad=3.0 * panel_scale,
    )
    inset.axhline(
        1.0,
        color="0.25",
        linestyle="--",
        linewidth=0.65 * panel_scale,
        zorder=1,
    )
    inset.tick_params(
        axis="y",
        labelsize=6.5 * panel_scale,
        length=2.0 * panel_scale,
        pad=1.0 * panel_scale,
    )
    inset.tick_params(axis="x", bottom=False, labelbottom=False)
    for x, value in zip(positions, values):
        inset.annotate(
            f"{value:.3f}",
            xy=(x, value),
            xytext=(0, 1.3 * panel_scale),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5 * panel_scale,
            color="0.2",
            clip_on=True,
        )
    for spine in inset.spines.values():
        spine.set_color("0.55")
        spine.set_linewidth(0.55 * panel_scale)


def _plot_profiles(
    profiles: dict[str, np.ndarray],
    variance_ratios: dict[str, float],
    *,
    beta: float,
    lattice_size: int,
    cp_n: int,
    output: Path,
    panel_scale: float = 1.0,
) -> None:
    """Render radial profiles with optional scaling for small panels."""
    fig, axis = plt.subplots(
        1,
        1,
        figsize=(fig_width, 0.50 * fig_width),
    )
    max_nonzero_radius = 1.0
    for method, style in PROFILE_STYLES.items():
        radii, values = _radial_average(profiles[method])
        nonzero = np.flatnonzero(np.abs(values) > 1e-7)
        if len(nonzero):
            max_nonzero_radius = max(
                max_nonzero_radius, float(radii[nonzero[-1]])
            )
        axis.plot(
            radii,
            values,
            color=style["color"],
            marker=style["marker"],
            markersize=3.2 * panel_scale,
            linewidth=1.0 * panel_scale,
            label=style["label"],
        )
    axis.tick_params(axis="both", labelsize=7 * panel_scale)
    axis.set_ylabel(r"$F_r$", fontsize=9 * panel_scale)
    axis.set_xlabel(r"$r=|x|$", labelpad=1, fontsize=9 * panel_scale)
    axis.set_xlim(0.0, max_nonzero_radius + 0.75)
    axis.set_axisbelow(True)
    axis.grid(color="0.8", linewidth=0.5, alpha=0.5)
    _add_variance_ratio_inset(axis, variance_ratios, panel_scale)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95), w_pad=0.2, h_pad=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Generate CP(2) numerical Approximation 2/3 profiles and the plot."""
    args = build_parser().parse_args(argv)
    if not np.isfinite(args.panel_scale) or args.panel_scale <= 0:
        raise ValueError("panel-scale must be positive and finite")
    if args.beta <= 0 or not np.isfinite(args.beta):
        raise ValueError("beta must be positive and finite")
    if args.cp_n < 1 or args.lattice_size < 2:
        raise ValueError("cp_n and lattice_size must be positive")
    if not 0 <= args.i <= args.cp_n or not 0 <= args.j <= args.cp_n:
        raise ValueError("i and j must address CP(N) components")
    if args.i == args.j:
        raise ValueError("this radial comparison is configured for off-diagonal OPFs")
    if args.mask_radius < 0 or args.batch_size < 1:
        raise ValueError("mask-radius must be non-negative and batch-size positive")
    if args.alpha_bounds[0] >= args.alpha_bounds[1]:
        raise ValueError("alpha bounds must be strictly increasing")
    if args.alpha_grid_points < 3:
        raise ValueError("alpha-grid-points must be at least 3")
    output = args.output or Path(
        f"plots/cp{args.cp_n}_radial_profiles_b{args.beta:.2f}.pdf"
    )
    ensemble = args.ensemble or Path(
        f"data/cp{args.cp_n}_gauge_dim2_b{args.beta:.2f}_"
        f"L{args.lattice_size}_ens.npy"
    )
    lattice_shape = (args.lattice_size, args.lattice_size)
    source = (0, 0)
    variance_path = args.training_data_directory / training_savename(
        args.cp_n,
        2,
        args.beta,
        args.lattice_size,
        args.i,
        args.j,
        "variance",
        "gauge",
    ).name
    penalty_path = args.training_data_directory / training_savename(
        args.cp_n,
        2,
        args.beta,
        args.lattice_size,
        args.i,
        args.j,
        "penalty",
        "gauge",
    ).name
    fit_results = args.fit_results or output.with_name(
        f"{output.stem}_fit_results.npz"
    )
    signature = _fit_signature(
        args,
        ensemble=ensemble,
        variance_path=variance_path,
        penalty_path=penalty_path,
    )
    cached = None if args.recompute else _load_fit_cache(fit_results, signature)
    if cached is not None:
        print(f"reusing numerical fit results from {fit_results}")
        variance_profile = cached["variance_field"]
        penalty_profile = cached["penalty_field"]
        approx2_profile = cached["approx2_field"]
        approx3_profile = cached["approx3_field"]
        direct_alpha, penalty_alpha, approx2_alpha, approx3_alpha = cached[
            "alphas"
        ]
        (
            variance_ratio,
            penalty_ratio,
            approx2_ratio,
            approx3_ratio,
        ) = cached["variance_ratios"]
        initial_variance = cached["initial_variance"]
        ensemble_info = cached["ensemble_info"]
        approx2_diagnostics = cached["approx2_diagnostics"]
        approx3_diagnostics = cached["approx3_diagnostics"]
        cache_reused = True
    else:
        variance_profile, variance_data = _load_saved_field(
            variance_path,
            cp_n=args.cp_n,
            beta=args.beta,
            lattice_shape=lattice_shape,
            i=args.i,
            j=args.j,
        )
        penalty_profile, penalty_data = _load_saved_field(
            penalty_path,
            cp_n=args.cp_n,
            beta=args.beta,
            lattice_shape=lattice_shape,
            i=args.i,
            j=args.j,
        )
        direct_alpha = float(variance_data.get("alpha_scale", 1.0))
        penalty_alpha = float(penalty_data.get("alpha_scale", 1.0))
        fields, links, ensemble_info = _load_profile_ensemble(
            ensemble,
            cp_n=args.cp_n,
            ndim=2,
            lattice_size=args.lattice_size,
            i=args.i,
            j=args.j,
            decorrelation=args.decorrelation,
            max_configurations=args.max_configurations,
        )
        model = CPnHomogeneousDeformations(args.cp_n, action_type="gauge")
        model.samples = {"all": fields}
        model.set_gauge_samples({"all": links})
        action = CPnGaugeAction(args.beta, args.cp_n, lattice_shape)
        variance_objective = _BatchedCPnVariance(
            fields,
            links,
            action,
            model,
            i=args.i,
            j=args.j,
            source=source,
            batch_size=args.batch_size,
        )
        initial_variance = variance_objective.initial_var
        print(f"undeformed OPF variance={initial_variance:.8g}")

        approx2_profile, approx2_diagnostics = solve_approx2_penalty_field(
            lattice_shape,
            source,
            args.beta,
            cp_n=args.cp_n,
            i=args.i,
            j=args.j,
            mask_radius=args.mask_radius,
        )
        approx3_profile, approx3_diagnostics = solve_approx3_penalty_field(
            lattice_shape,
            source,
            args.beta,
            cp_n=args.cp_n,
            i=args.i,
            j=args.j,
            mask_radius=args.mask_radius,
            initial_field=approx2_profile,
        )
        approx2_tensor = torch.as_tensor(approx2_profile, dtype=fields.dtype)
        approx3_tensor = torch.as_tensor(approx3_profile, dtype=fields.dtype)
        approx2_alpha, approx2_variance = optimize_field_scale(
            approx2_tensor,
            variance_objective,
            bounds=tuple(args.alpha_bounds),
            grid_points=args.alpha_grid_points,
            xatol=args.alpha_xatol,
        )
        approx3_alpha, approx3_variance = optimize_field_scale(
            approx3_tensor,
            variance_objective,
            bounds=tuple(args.alpha_bounds),
            grid_points=args.alpha_grid_points,
            xatol=args.alpha_xatol,
        )

        variance_ratio = variance_objective(
            alphas=torch.as_tensor(variance_profile, dtype=fields.dtype)
        ).item() / initial_variance
        penalty_ratio = variance_objective(
            alphas=torch.as_tensor(penalty_profile, dtype=fields.dtype)
        ).item() / initial_variance
        approx2_ratio = approx2_variance / initial_variance
        approx3_ratio = approx3_variance / initial_variance
        _save_fit_cache(
            fit_results,
            signature=signature,
            fields={
                "variance": variance_profile,
                "penalty": penalty_profile,
                "approx2": approx2_profile,
                "approx3": approx3_profile,
            },
            alphas=(
                direct_alpha,
                penalty_alpha,
                float(approx2_alpha),
                float(approx3_alpha),
            ),
            variance_ratios=(
                float(variance_ratio),
                float(penalty_ratio),
                float(approx2_ratio),
                float(approx3_ratio),
            ),
            initial_variance=float(initial_variance),
            ensemble_info=ensemble_info,
            approx2_diagnostics=approx2_diagnostics,
            approx3_diagnostics=approx3_diagnostics,
        )
        print(f"saved reusable field and alpha fits to {fit_results}")
        cache_reused = False

    print(f"undeformed OPF variance={initial_variance:.8g}")
    print(
        f"Approximation 2: alpha={approx2_alpha:.8g}, "
        f"variance ratio={approx2_ratio:.8g}, "
        f"penalty exponent={approx2_diagnostics['penalty_exponent']:.8g}"
    )
    print(
        f"Approximation 3: alpha={approx3_alpha:.8g}, "
        f"variance ratio={approx3_ratio:.8g}, "
        f"penalty exponent={approx3_diagnostics['penalty_exponent']:.8g}"
    )
    print(
        f"saved training profiles on this selected ensemble: "
        f"variance ratio={variance_ratio:.8g}; Approximation 1 ratio="
        f"{penalty_ratio:.8g}, saved alpha={penalty_alpha:.8g}"
    )

    direction = np.zeros(args.cp_n + 1, dtype=float)
    direction[args.i] = 1.0 / np.sqrt(2.0)
    direction[args.j] = -1.0 / np.sqrt(2.0)
    profiles = {
        "variance": np.tensordot(variance_profile, direction, axes=([-1], [0])),
        "penalty": np.tensordot(penalty_profile, direction, axes=([-1], [0])),
        "approx2": np.tensordot(
            approx2_alpha * approx2_profile, direction, axes=([-1], [0])
        ),
        "approx3": np.tensordot(
            approx3_alpha * approx3_profile, direction, axes=([-1], [0])
        ),
    }
    plot_variance_ratios = {
        "variance": float(variance_ratio),
        "penalty": float(penalty_ratio),
        "approx2": float(approx2_ratio),
        "approx3": float(approx3_ratio),
    }
    _plot_profiles(
        profiles,
        plot_variance_ratios,
        beta=args.beta,
        lattice_size=args.lattice_size,
        cp_n=args.cp_n,
        output=output,
        panel_scale=args.panel_scale,
    )

    result = {
        "metadata": {
            "cp_n": args.cp_n,
            "ndim": 2,
            "beta": args.beta,
            "lattice_size": lattice_shape,
            "source": source,
            "i": args.i,
            "j": args.j,
            "mask_radius": args.mask_radius,
            "profile_component": "projection onto normalized OPF source direction",
            "ensemble": str(ensemble),
            "decorrelation_factor": args.decorrelation,
            "fit_results": str(fit_results),
            "cache_reused": cache_reused,
            "fit_signature": signature,
            **ensemble_info,
        },
        "variance": {
            "undeformed": float(initial_variance),
            "direct_training": float(variance_ratio),
            "approximation_1": float(penalty_ratio),
            "approximation_2": float(approx2_ratio),
            "approximation_3": float(approx3_ratio),
        },
        "alpha": {
            "direct_training": float(direct_alpha),
            "approximation_1": float(penalty_alpha),
            "approximation_2": float(approx2_alpha),
            "approximation_3": float(approx3_alpha),
            "bounds": [float(value) for value in args.alpha_bounds],
        },
        "optimization": {
            "approximation_2": approx2_diagnostics,
            "approximation_3": approx3_diagnostics,
        },
        "radial_profiles": {
            method: {
                "radii": _radial_average(profile)[0].tolist(),
                "values": _radial_average(profile)[1].tolist(),
            }
            for method, profile in profiles.items()
        },
    }
    report_path = output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"saved {output}, {output.with_suffix('.png')}, "
        f"and {report_path}"
    )


if __name__ == "__main__":
    main()
