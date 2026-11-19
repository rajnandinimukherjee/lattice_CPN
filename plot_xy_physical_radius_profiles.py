"""Plot saved Approximation 3 XY profiles at fixed R/xi for Figure 10.

The physical-radius scan has already optimized each deformation amplitude.
This script reads its saved scaled fields without rerunning the fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_settings import fig_width, latex_font_params


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "training_data" / "xy_physical_radius"
OUTPUT = ROOT / "paper" / "figs" / "xy_physical_radius_profiles.pdf"
LATTICE_SIZES = {1: 64, 2: 32, 3: 16}
CRITICAL_BETAS = {1: np.inf, 2: 1.1199, 3: 0.454165}


def load_profiles(dimension: int) -> tuple[float, list[tuple[float, int, np.ndarray]]]:
    """Return (target R/xi, [(beta, R, scaled field), ...])."""
    with (DATA / f"xy_physrad_dim{dimension}_summary.json").open() as handle:
        summary = json.load(handle)
    if summary["ndim"] != dimension:
        raise ValueError(f"Incorrect dimension in summary for d={dimension}")

    lattice_size = LATTICE_SIZES[dimension]
    profiles = []
    for record in sorted(summary["records"], key=lambda item: item["beta"]):
        beta = float(record["beta"])
        radius = int(record["radius"])
        filename = DATA / (
            f"xy_physrad_dim{dimension}_b{beta:.2f}_R{radius:02d}"
            "_loss-approx3_result.npy"
        )
        result = np.load(filename, allow_pickle=True).item()
        metadata = result["metadata"]
        expected_shape = (lattice_size,) * dimension
        if (
            metadata["study"] != "xy_physical_radius"
            or metadata["loss_function"] != "approx3"
            or metadata["ndim"] != dimension
            or tuple(metadata["lattice_size"]) != expected_shape
            or tuple(metadata["source"]) != (0,) * dimension
            or metadata["mask_radius"] != radius
            or not np.isclose(metadata["beta"], beta)
        ):
            raise ValueError(f"Inconsistent metadata in {filename}")
        field = np.asarray(result["scaled_field"], dtype=float)
        if field.shape != expected_shape + (1,) or not np.all(np.isfinite(field)):
            raise ValueError(f"Invalid scaled field in {filename}")
        if not np.isclose(field[(0,) * dimension + (0,)], record["field_at_source"]):
            raise ValueError(f"Source amplitude disagrees with summary in {filename}")
        profiles.append((beta, radius, field[..., 0]))
    if not profiles:
        raise ValueError(f"No fixed-physical-radius profiles for d={dimension}")
    return float(summary["physical_radius"]), profiles


def plot() -> None:
    datasets = {dimension: load_profiles(dimension) for dimension in (1, 2, 3)}
    targets = [target for target, _ in datasets.values()]
    if not np.allclose(targets, targets[0]):
        raise ValueError("The dimensions use different target physical radii")

    plt.rcParams.update(latex_font_params)
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    figure, axes = plt.subplots(1, 3, figsize=(fig_width, 1.9),
                               gridspec_kw={"left": 0.09, "right": 0.99,
                                            "top": 0.86, "bottom": 0.20,
                                            "wspace": 0.05})
    ymax = max(float(field.max()) for _, profiles in datasets.values()
               for _, _, field in profiles)

    for column, dimension in enumerate((1, 2, 3)):
        axis = axes[column]
        _, profiles = datasets[dimension]
        extent = max(radius for _, radius, _ in profiles) + 1
        x = np.arange(0, extent + 1)
        colors = plt.cm.viridis(np.linspace(0.06, 0.94, len(profiles)))
        critical_beta = CRITICAL_BETAS[dimension]
        critical_label = (
            r"\infty" if np.isinf(critical_beta) else f"{critical_beta:.2f}"
        )
        handles = []
        for (beta, radius, field), color in zip(profiles, colors):
            # The source is at the origin; all other coordinates of this
            # one-dimensional lattice slice are held at zero.
            slice_values = field[(x,) + (0,) * (dimension - 1)]
            if np.any(np.abs(slice_values[x > radius]) > 1e-10):
                raise ValueError(f"Nonzero field outside support for d={dimension}, beta={beta}")
            line, = axis.plot(x, slice_values, color=color, lw=1.05, marker="o",
                              markersize=1.55, label=rf"${beta:g}$", zorder=2)
            handles.append(line)

        axis.set_title(rf"$d={dimension},\quad L={LATTICE_SIZES[dimension]}$", pad=3)
        axis.set_xlim(0, extent)
        axis.set_ylim(-0.03, ymax * 1.05)
        axis.set_xlabel(r"$x$", labelpad=2)
        if column == 0:
            axis.set_ylabel(r"$F_x$", labelpad=2)
        else:
            axis.tick_params(labelleft=False)
        axis.set_xticks(np.arange(0, extent + 1, 2))
        axis.grid(color="0.83", lw=0.45, alpha=0.65)
        axis.set_axisbelow(True)
        axis.tick_params(length=2.5, pad=2)
        for spine in axis.spines.values():
            spine.set_linewidth(0.7)

        axis.text(0.42, 0.96,
                  rf"$\beta_{{\rm c}}={critical_label}$",
                  transform=axis.transAxes, ha="right", va="top",
                  fontsize=6.8, color="black")

        axis.legend(handles=handles, title=r"$\beta$", ncol=2,
                    loc="upper right", bbox_to_anchor=(0.99, 0.99),
                    frameon=True, facecolor="white", edgecolor="0.85",
                    framealpha=0.9, fontsize=6.8, title_fontsize=7,
                    columnspacing=0.45, handlelength=1.0,
                    handletextpad=0.3, labelspacing=0.12, borderpad=0.25)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=240)
    plt.close(figure)
    print(f"Saved {OUTPUT} with {[len(datasets[d][1]) for d in (1, 2, 3)]} beta curves")


if __name__ == "__main__":
    plot()
