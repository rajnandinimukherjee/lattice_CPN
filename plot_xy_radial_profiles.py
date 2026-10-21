"""Compare radial XY one-point deformation profiles for four strategies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import XY_utils as xy
from plot_settings import fig_width, latex_font_params
from xy_opf.config import DATA_DIRECTORY, TRAINING_DATA_DIRECTORY
from xy_opf.optimization import (
    fit_lattice_laplace_profile,
    solve_lattice_laplace_field,
)


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
    "saddle-point": {
        "color": "tab:green",
        "marker": "^",
        "label": "Approx. 2 (small fluc.)",
    },
    "lattice-laplace": {
        "color": "tab:red",
        "marker": "D",
        "label": "Approx. 3 (smooth deforms)",
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Build the radial-profile command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[0.65, 0.80, 1.20, 1.35],
        help="XY couplings with saved 2D training profiles",
    )
    parser.add_argument(
        "--lattice-size",
        type=int,
        default=32,
        help="linear extent of the square lattice",
    )
    parser.add_argument(
        "--ensemble-directory",
        type=Path,
        default=DATA_DIRECTORY,
        help="directory containing xy_dim2_b*_L*_ens.npy ensembles",
    )
    parser.add_argument(
        "--training-data-directory",
        type=Path,
        default=TRAINING_DATA_DIRECTORY,
        help="directory containing the saved Approximation 1 and 2 fields",
    )
    parser.add_argument(
        "--mask-radius",
        type=int,
        default=None,
        help="periodic support radius; defaults to the saved saddle-point value",
    )
    parser.add_argument(
        "--alpha-bounds",
        nargs=2,
        type=float,
        default=(-5.0, 5.0),
        metavar=("LOWER", "UPPER"),
        help="bounds for the Approximation 3 variance-minimizing scale",
    )
    parser.add_argument(
        "--alpha-grid-points",
        type=int,
        default=41,
        help="coarse grid points used before bounded alpha refinement",
    )
    parser.add_argument(
        "--alpha-xatol",
        type=float,
        default=1e-4,
        help="absolute tolerance for bounded scalar alpha refinement",
    )
    parser.add_argument(
        "--approx3-alpha-scale",
        type=float,
        default=None,
        help=(
            "use a previously optimized Approximation 3 scale without loading "
            "an XY ensemble (single beta only)"
        ),
    )
    parser.add_argument(
        "--approx3-fit-results",
        type=Path,
        default=None,
        help="saved Approximation 3 fit JSON used with --approx3-alpha-scale",
    )
    parser.add_argument(
        "--decorrelation",
        type=float,
        default=1.0,
        help="ensemble thinning factor passed to XY_utils.load_ensemble",
    )
    parser.add_argument(
        "--panel-scale",
        type=float,
        default=1.0,
        help="scale plot text, lines, and markers for smaller panel layouts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figs/fig5_xy_radial_profiles.pdf"),
        help="output profile-only PDF path; a PNG preview is saved alongside it",
    )
    parser.add_argument(
        "--legend-output",
        type=Path,
        default=None,
        help="output path for the reusable standalone legend PDF",
    )
    parser.add_argument(
        "--legend-only",
        action="store_true",
        help="write only the reusable legend, without loading profiles or data",
    )
    return parser


def _load_saved_profile(
    training_directory: Path,
    beta: float,
    loss_function: str,
    lattice_shape: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Load one saved, already-scaled profile, its variance ratio, and metadata."""
    suffix = "result" if loss_function == "saddle-point" else "training"
    filename = training_directory / (
        f"xy_scaling_dim2_b{beta:.2f}_loss-{loss_function}_{suffix}.npy"
    )
    if not filename.exists():
        raise FileNotFoundError(f"missing saved profile: {filename}")
    data = np.load(filename, allow_pickle=True).item()
    metadata = data.get("metadata", {})
    if int(metadata.get("ndim", -1)) != 2:
        raise ValueError(f"{filename} is not a two-dimensional profile")
    if tuple(metadata.get("lattice_size", ())) != lattice_shape:
        raise ValueError(
            f"{filename} uses lattice {metadata.get('lattice_size')}, "
            f"expected {lattice_shape}"
        )
    if not np.isclose(float(metadata.get("beta", np.nan)), beta):
        raise ValueError(f"{filename} has a different beta value")
    if "scaled_field" in data:
        field = np.asarray(data["scaled_field"], dtype=float)
    else:
        field = float(data.get("alpha_scale", 1.0)) * np.asarray(
            data["learned_field"], dtype=float
        )
    if field.shape == (*lattice_shape, 1):
        field = field[..., 0]
    if field.shape != lattice_shape or not np.all(np.isfinite(field)):
        raise ValueError(f"{filename} contains an invalid deformation field")
    variance_ratio = float(data.get("variance_ratio", np.nan))
    if not np.isfinite(variance_ratio) or variance_ratio < 0.0:
        raise ValueError(f"{filename} contains an invalid variance ratio")
    return field, variance_ratio


def _radial_average(
    field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average a field over shells of periodic Euclidean distance."""
    shape = field.shape
    coordinates = np.indices(shape, dtype=np.int64)
    periodic_distance = np.minimum(
        coordinates,
        np.asarray(shape, dtype=np.int64).reshape(-1, *([1] * field.ndim))
        - coordinates,
    )
    radius_squared = np.square(periodic_distance).sum(axis=0)
    shells = np.unique(radius_squared)
    radii = np.sqrt(shells.astype(float))
    means = np.asarray(
        [field[radius_squared == shell].mean() for shell in shells],
        dtype=float,
    )
    return radii, means


def _legend_output_path(output_path: Path) -> Path:
    """Return the default standalone legend path for a profile figure."""
    stem = output_path.stem.removesuffix("_profiles")
    return output_path.with_name(f"{stem}_legend{output_path.suffix}")


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


def save_profile_legend(output_path: Path) -> None:
    """Save the shared strategy legend without loading any profile data."""
    handles = [
        Line2D(
            [],
            [],
            color=style["color"],
            marker=style["marker"],
            markersize=3.2,
            linewidth=1.0,
            label=style["label"],
        )
        for style in PROFILE_STYLES.values()
    ]
    figure = plt.figure(figsize=(fig_width, 0.32))
    figure.legend(
        handles,
        [style["label"] for style in PROFILE_STYLES.values()],
        loc="center",
        ncols=4,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(
        output_path.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


def plot_profiles(
    betas: list[float],
    profiles: dict[float, dict[str, np.ndarray]],
    output_path: Path,
    lattice_size: int,
    variance_ratios: dict[float, dict[str, float]],
    panel_scale: float = 1.0,
) -> None:
    """Plot radial profiles, optionally scaling elements for small panels."""
    ncols = min(2, len(betas))
    nrows = math.ceil(len(betas) / ncols)
    nonzero_radii = []
    for beta in betas:
        for field in profiles[beta].values():
            radii, values = _radial_average(field)
            nonzero = np.flatnonzero(np.abs(values) > 1e-7)
            if len(nonzero):
                nonzero_radii.append(float(radii[nonzero[-1]]))
    max_radius = max(nonzero_radii, default=1.0) + 0.75
    fig, axes = plt.subplots(
        nrows,
        ncols,
        squeeze=False,
        sharex=True,
        sharey=True,
        figsize=(fig_width, 0.50 * fig_width * nrows),
    )
    flat_axes = axes.ravel()

    for axis, beta in zip(flat_axes, betas):
        for method, style in PROFILE_STYLES.items():
            radii, values = _radial_average(profiles[beta][method])
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
        axis.set_axisbelow(True)
        axis.grid(color="0.8", linewidth=0.5, alpha=0.5)
        axis.set_xlim(0.0, max_radius)
        _add_variance_ratio_inset(axis, variance_ratios[beta], panel_scale)

    for axis in flat_axes[len(betas) :]:
        axis.set_visible(False)

    axes[0, 0].set_ylabel(r"$F_r$", fontsize=9 * panel_scale)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel(
                r"$r=|x|$", labelpad=1, fontsize=9 * panel_scale
            )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95), w_pad=0.2, h_pad=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Load saved profiles, solve Approximation 3, and plot the comparison."""
    args = build_parser().parse_args(argv)
    if not np.isfinite(args.panel_scale) or args.panel_scale <= 0:
        raise ValueError("panel-scale must be positive and finite")
    legend_output = args.legend_output or _legend_output_path(args.output)
    if args.legend_only:
        save_profile_legend(legend_output)
        print(
            f"saved standalone legend to {legend_output} and "
            f"{legend_output.with_suffix('.png')}"
        )
        return
    if not args.betas or any(beta <= 0.0 for beta in args.betas):
        raise ValueError("at least one positive beta value is required")
    if args.lattice_size < 2:
        raise ValueError("lattice-size must be at least two")
    if args.approx3_alpha_scale is not None:
        if not np.isfinite(args.approx3_alpha_scale):
            raise ValueError("Approximation 3 alpha scale must be finite")
        if len(set(args.betas)) != 1:
            raise ValueError(
                "a fixed Approximation 3 alpha scale requires one beta value"
            )
    if args.alpha_bounds[0] >= args.alpha_bounds[1]:
        raise ValueError("alpha bounds must be strictly increasing")
    lattice_shape = (args.lattice_size, args.lattice_size)
    profiles: dict[float, dict[str, np.ndarray]] = {}
    variance_ratios: dict[float, dict[str, float]] = {}

    for beta in sorted(set(args.betas)):
        fields = None
        if args.approx3_alpha_scale is None:
            ensemble_path = args.ensemble_directory / (
                f"xy_dim2_b{beta:.2f}_L{args.lattice_size}_ens.npy"
            )
            if not ensemble_path.exists():
                raise FileNotFoundError(
                    f"missing XY ensemble: {ensemble_path}; provide it with "
                    "--ensemble-directory"
                )
            fields, saved_beta = xy.load_ensemble(
                str(ensemble_path),
                decorrelation_factor=args.decorrelation,
            )
            if not np.isclose(saved_beta, beta):
                raise ValueError(
                    f"{ensemble_path} contains beta={saved_beta:g}, "
                    f"expected beta={beta:g}"
                )
            if tuple(fields.shape[1:]) != lattice_shape:
                raise ValueError(
                    f"{ensemble_path} uses lattice {tuple(fields.shape[1:])}, "
                    f"expected {lattice_shape}"
                )

        saved_profiles = {
            method: _load_saved_profile(
                args.training_data_directory,
                beta,
                method,
                lattice_shape,
            )
            for method in ("variance", "penalty", "saddle-point")
        }
        beta_profiles = {
            method: loaded[0] for method, loaded in saved_profiles.items()
        }
        beta_variance_ratios = {
            method: loaded[1] for method, loaded in saved_profiles.items()
        }
        saddle_filename = args.training_data_directory / (
            f"xy_scaling_dim2_b{beta:.2f}_loss-saddle-point_result.npy"
        )
        saddle_data = np.load(saddle_filename, allow_pickle=True).item()
        mask_radius = (
            args.mask_radius
            if args.mask_radius is not None
            else int(saddle_data.get("metadata", {}).get("mask_radius", 6))
        )
        if fields is None:
            laplace_field, diagnostics = solve_lattice_laplace_field(
                lattice_shape,
                (0, 0),
                beta,
                mask_radius=mask_radius,
            )
            alpha_scale = float(args.approx3_alpha_scale)
            beta_profiles["lattice-laplace"] = alpha_scale * laplace_field
            fit_results_path = args.approx3_fit_results or args.output.with_name(
                f"{args.output.stem}_fit_results.json"
            )
            if not fit_results_path.exists():
                raise FileNotFoundError(
                    "missing saved Approximation 3 fit report needed for the "
                    f"variance-ratio inset: {fit_results_path}"
                )
            with fit_results_path.open(encoding="utf-8") as fit_file:
                fit_report = json.load(fit_file)
            if not np.isclose(float(fit_report.get("beta", np.nan)), beta):
                raise ValueError(
                    f"{fit_results_path} has a different beta value"
                )
            if not np.isclose(
                float(fit_report.get("alpha_scale", np.nan)), alpha_scale
            ):
                raise ValueError(
                    "the requested Approximation 3 scale does not match "
                    f"the fit report in {fit_results_path}"
                )
            variance_ratio = float(fit_report.get("variance_ratio", np.nan))
        else:
            laplace_result = fit_lattice_laplace_profile(
                fields,
                beta,
                (0, 0),
                mask_radius=mask_radius,
                alpha_bounds=tuple(args.alpha_bounds),
                alpha_grid_points=args.alpha_grid_points,
                alpha_xatol=args.alpha_xatol,
            )
            beta_profiles["lattice-laplace"] = np.asarray(
                laplace_result["scaled_field"], dtype=float
            )[..., 0]
            alpha_scale = float(laplace_result["alpha_scale"])
            variance_ratio = float(laplace_result["variance_ratio"])
            diagnostics = laplace_result["optimization_diagnostics"]
        profiles[beta] = beta_profiles
        if not np.isfinite(variance_ratio) or variance_ratio < 0.0:
            raise ValueError("Approximation 3 fit report has an invalid variance ratio")
        beta_variance_ratios["lattice-laplace"] = variance_ratio
        variance_ratios[beta] = beta_variance_ratios
        print(
            f"beta={beta:g}: Approximation 3 alpha={alpha_scale:.6g}, "
            f"variance ratio={variance_ratio:.6g}, "
            f"mask radius={mask_radius}, "
            f"Laplace residual="
            f"{diagnostics['max_abs_laplace_residual']:.3g}"
        )

    plot_profiles(
        sorted(profiles),
        profiles,
        args.output,
        args.lattice_size,
        variance_ratios,
        panel_scale=args.panel_scale,
    )
    save_profile_legend(legend_output)
    report_path = args.output.with_suffix(".json")
    report = {
        "model": "XY",
        "dimension": 2,
        "lattice_size": args.lattice_size,
        "variance_ratios": {
            f"{beta:g}": {
                method: float(value)
                for method, value in beta_ratios.items()
            }
            for beta, beta_ratios in variance_ratios.items()
        },
        "ratio_definition": "deformed variance divided by undeformed variance",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"saved plot {args.output} and {args.output.with_suffix('.png')}; "
        f"legend {legend_output} and {legend_output.with_suffix('.png')}; "
        f"variance ratios {report_path}"
    )


if __name__ == "__main__":
    main()
