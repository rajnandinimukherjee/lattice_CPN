"""Top-row-only companion of the ``xy_scaling``/``cp2_scaling`` scan.

Reuses the saved Approximation 3 (smooth-deformation) results already
indexed by the ``xy_opf``/``cpn_opf`` studies -- nothing is retrained here.
The panel layout (one column per dimension/lattice size, critical-temperature
marking and disordered-phase shading, axis labels) matches the full
``xy_scaling``/``cp2_scaling`` figure exactly, but keeps only the
variance-ratio row and only the Approximation 3 series, drawn with the
connected line+marker style of ``xy_mask_radius_scaling`` instead of bare
scatter points.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_settings import fig_width, latex_font_params

from cpn_opf.config import GAUGE_CRITICAL_TEMPERATURES
from cpn_opf.study import _load_dimension as _load_cpn_dimension
from xy_opf.config import XY_CRITICAL_TEMPERATURES
from xy_opf.study import _load_dimension as _load_xy_dimension

FIGURES_DIRECTORY = Path("paper/figs")

APPROX3_COLOR = "tab:red"
APPROX3_MARKER = "D"
APPROX3_LABEL = "Approx. 3 (smooth deformation)"

plt.rcParams.update(latex_font_params)
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    }
)


def _load_xy_approx3() -> tuple[
    list[int], list[str], list[np.ndarray], list[np.ndarray]
]:
    """Load the saved Approximation 3 XY variance ratios for every dimension."""
    dims = [1, 2, 3]
    lattice_sizes = ["64", "32", "16"]
    betas_by_dim: list[np.ndarray] = []
    ratios_by_dim: list[np.ndarray] = []
    for ndim in dims:
        variance_ratios: dict[str, list] = {"approx3": []}
        deformations_at_source: dict[str, list] = {"approx3": []}
        betas_by_loss: dict[str, list] = {"approx3": []}
        _load_xy_dimension(
            args=argparse.Namespace(plot_training=False),
            ndim=ndim,
            loss_functions=["approx3"],
            variance_ratios=variance_ratios,
            deformations_at_source=deformations_at_source,
            betas_by_loss=betas_by_loss,
            training_figures=[],
        )
        betas_by_dim.append(
            np.asarray(betas_by_loss["approx3"][0], dtype=float)
        )
        ratios_by_dim.append(
            np.asarray(variance_ratios["approx3"][0], dtype=float)
        )
    return dims, lattice_sizes, betas_by_dim, ratios_by_dim


def _load_cp2_approx3() -> tuple[
    list[int], list[int], list[np.ndarray], list[np.ndarray]
]:
    """Load the saved Approximation 3 CP(2) variance ratios for every dimension."""
    dims = [1, 2, 3]
    lattice_sizes = [128, 64, 16]
    betas_by_dim: list[np.ndarray] = []
    ratios_by_dim: list[np.ndarray] = []
    for ndim, lattice_size in zip(dims, lattice_sizes):
        variance_ratios: dict[str, list] = {"approx3": []}
        deformations_at_source: dict[str, list] = {"approx3": []}
        betas_by_loss: dict[str, list] = {"approx3": []}
        _load_cpn_dimension(
            argparse.Namespace(
                N=2,
                i=0,
                j=1,
                action="gauge",
                plot_training=False,
                save_plot=None,
            ),
            ndim,
            lattice_size,
            ["approx3"],
            variance_ratios,
            deformations_at_source,
            betas_by_loss,
        )
        betas_by_dim.append(
            np.asarray(betas_by_loss["approx3"][0], dtype=float)
        )
        ratios_by_dim.append(
            np.asarray(variance_ratios["approx3"][0], dtype=float)
        )
    return dims, lattice_sizes, betas_by_dim, ratios_by_dim


def _plot_row(
    axes_row: np.ndarray,
    dims: list[int],
    lattice_sizes: list,
    betas_by_dim: list[np.ndarray],
    ratios_by_dim: list[np.ndarray],
    critical_temperatures: dict[int, float],
) -> None:
    """Draw one Approximation-3 variance-ratio row, one column per dimension."""
    for column, (dim, lattice_size, betas, ratios) in enumerate(
        zip(dims, lattice_sizes, betas_by_dim, ratios_by_dim)
    ):
        axis = axes_row[column]
        if np.any(betas <= 0.0):
            raise ValueError(
                "beta values must be positive to plot T=1/beta"
            )
        order = np.argsort(betas)
        betas = betas[order]
        ratios = ratios[order]
        temperatures = 1.0 / betas

        axis.plot(
            temperatures,
            ratios,
            color=APPROX3_COLOR,
            lw=0.9,
            zorder=2,
        )
        axis.scatter(
            temperatures,
            ratios,
            color=APPROX3_COLOR,
            marker=APPROX3_MARKER,
            s=14,
            linewidths=0.6,
            edgecolors="white",
            zorder=3,
        )

        axis.axhline(1.0, color="black", lw=0.8, zorder=1)
        axis.set_title(rf"$d={dim},\ L={lattice_size}$", pad=3)
        axis.set_axisbelow(True)
        axis.grid(color="0.8", lw=0.5, alpha=0.5)
        if column > 0:
            axis.tick_params(axis="y", left=True, labelleft=False)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

        critical_temperature = (
            critical_temperatures.get(dim)
            if len(set(betas.tolist())) > 1
            else None
        )
        if critical_temperature is not None and critical_temperature >= 0:
            x_limits = axis.get_xlim()
            if critical_temperature < x_limits[0]:
                axis.set_xlim(left=critical_temperature)
                x_limits = axis.get_xlim()
            axis.axvline(
                critical_temperature,
                color="black",
                ls=":",
                lw=0.8,
                zorder=1,
            )
            axis.axvspan(
                critical_temperature,
                x_limits[1],
                facecolor="0.96",
                edgecolor="none",
                zorder=-1,
            )
            axis.set_xlim(x_limits)
            critical_axis = axis.secondary_xaxis("top")
            critical_axis.set_xticks(
                [critical_temperature],
                labels=[rf"$T_{{\rm c}}={critical_temperature:.3g}$"],
            )
            critical_axis.tick_params(axis="x", pad=1, labelsize=6)


def _make_panel(
    dims: list[int],
    lattice_sizes: list,
    betas_by_dim: list[np.ndarray],
    ratios_by_dim: list[np.ndarray],
    critical_temperatures: dict[int, float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(dims),
        sharex="col",
        sharey="row",
        squeeze=False,
        figsize=(fig_width, 0.36 * fig_width),
    )
    _plot_row(
        axes[0],
        dims,
        lattice_sizes,
        betas_by_dim,
        ratios_by_dim,
        critical_temperatures,
    )
    axes[0, 0].set_ylabel(
        r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$"
    )
    fig.supxlabel(r"$T=\beta^{-1}$")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=0.0, h_pad=0.0)
    fig.subplots_adjust(wspace=0.025, bottom=0.26, top=0.78)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    xy_dims, xy_lattice_sizes, xy_betas, xy_ratios = _load_xy_approx3()
    cp2_dims, cp2_lattice_sizes, cp2_betas, cp2_ratios = _load_cp2_approx3()

    _make_panel(
        xy_dims,
        xy_lattice_sizes,
        xy_betas,
        xy_ratios,
        XY_CRITICAL_TEMPERATURES,
        FIGURES_DIRECTORY / "xy_scaling_top.pdf",
    )
    _make_panel(
        cp2_dims,
        cp2_lattice_sizes,
        cp2_betas,
        cp2_ratios,
        GAUGE_CRITICAL_TEMPERATURES,
        FIGURES_DIRECTORY / "cp2_scaling_top.pdf",
    )

    for path in (
        FIGURES_DIRECTORY / "xy_scaling_top.pdf",
        FIGURES_DIRECTORY / "cp2_scaling_top.pdf",
    ):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
