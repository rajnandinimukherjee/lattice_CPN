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
import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory
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
EXTRA_RADIUS_MARKER_SIZE = 2.5
CHOSEN_RADIUS_MARKER_AREA = 14

# Companion mask-radius scan (the ``xy_mask_radius_scaling`` figure): the
# periodic support radius was varied at fixed Approximation 3, repeated over
# every XY ensemble used here.  Only this one dataset backs the extra, faded
# series added to the XY panel.
XY_MASK_RADIUS_RESULTS = Path("training_data/xy_mask_radius_scaling/results.json")

# Radius to drop from each dimension's ladder, leaving three extra series
# alongside the dimension's own Approximation 3 radius (R=20 for d=1, R=6
# for d=2, 3).
XY_MASK_RADIUS_EXCLUDE = {1: 4, 2: 8, 3: 1}

# Extra radii smaller than the dimension's own Approximation 3 radius are
# drawn in faded reds (lighter for smaller R); extra radii larger than it are
# drawn in faded browns (darker for larger R) instead, so the one radius
# actually used for the rest of the paper's Approximation 3 results keeps its
# plain ``APPROX3_COLOR`` and is never confused with a merely lighter red.
BROWN_COLOR = "saddlebrown"
BLEND_LIGHT = 0.32
BLEND_DARK = 0.82


def _blend_with_white(
    base_color: str, fraction: float
) -> tuple[float, float, float]:
    """Blend white and ``base_color`` so 0 is white and 1 is the base color."""
    white = np.array([1.0, 1.0, 1.0])
    base = np.array(to_rgb(base_color))
    return tuple(white * (1.0 - fraction) + base * fraction)


def _graded_shades(
    base_color: str, count: int
) -> list[tuple[float, float, float]]:
    """``count`` shades of ``base_color``, lightest to darkest."""
    if count <= 0:
        return []
    if count == 1:
        fractions = [BLEND_DARK]
    else:
        fractions = np.linspace(BLEND_LIGHT, BLEND_DARK, count)
    return [_blend_with_white(base_color, fraction) for fraction in fractions]


def _load_xy_mask_radius_series(
    exclude_by_dim: dict[int, int],
) -> dict[int, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Load the saved mask-radius scan, grouped by dimension and radius.

    Returns, for each dimension, a list of ``(radius, temperature, ratio)``
    triples sorted by ascending radius, skipping the dimension's excluded
    radius.
    """
    with open(XY_MASK_RADIUS_RESULTS) as handle:
        payload = json.load(handle)

    by_dim_radius: dict[int, dict[int, list[tuple[float, float]]]] = (
        collections.defaultdict(lambda: collections.defaultdict(list))
    )
    for entry in payload["entries"].values():
        by_dim_radius[entry["ndim"]][entry["mask_radius"]].append(
            (entry["beta"], entry["variance_ratio"])
        )

    series_by_dim: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for ndim, radius_map in by_dim_radius.items():
        excluded = exclude_by_dim.get(ndim)
        series = []
        for radius in sorted(radius_map):
            if radius == excluded:
                continue
            points = sorted(radius_map[radius])
            betas = np.asarray([beta for beta, _ in points], dtype=float)
            ratios = np.asarray([ratio for _, ratio in points], dtype=float)
            series.append((radius, 1.0 / betas, ratios))
        series_by_dim[ndim] = series
    return series_by_dim

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
    extra_series_by_dim: dict[int, list[tuple[int, np.ndarray, np.ndarray]]]
    | None = None,
    main_radius_by_dim: dict[int, int] | None = None,
) -> None:
    """Draw one Approximation-3 variance-ratio row, one column per dimension.

    When ``extra_series_by_dim`` is given, each column also draws its extra
    ``(radius, temperature, ratio)`` series: radii smaller than the
    dimension's own Approximation 3 radius in faded reds, radii larger than
    it in faded browns, with a per-column legend listing every radius
    (including the dimension's own series, labelled via
    ``main_radius_by_dim``, which always keeps the plain ``APPROX3_COLOR``).
    Each logarithmic y-axis spans its own data, with room for the legend.
    """
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
        if np.any(ratios <= 0.0):
            raise ValueError("variance ratios must be positive for a log scale")
        axis_minimum = float(np.min(ratios))

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
            s=CHOSEN_RADIUS_MARKER_AREA,
            linewidths=0.6,
            edgecolors="white",
            zorder=3,
        )

        legend_entries = []
        main_radius = (
            main_radius_by_dim[dim] if main_radius_by_dim is not None else None
        )
        if main_radius is not None:
            legend_entries.append((main_radius, APPROX3_COLOR, APPROX3_MARKER))

        extra = (extra_series_by_dim or {}).get(dim, [])
        # Radii below the dimension's own Approximation 3 radius get faded
        # reds (lighter for smaller R); radii above it get faded browns
        # (darker for larger R), so a glance at the colour family alone
        # shows which side of the chosen radius a curve falls on.
        below = [item for item in extra if item[0] < main_radius]
        above = [item for item in extra if item[0] > main_radius]
        color_by_radius = {
            radius: color
            for (radius, _, _), color in zip(
                below, _graded_shades(APPROX3_COLOR, len(below))
            )
        }
        color_by_radius.update(
            {
                radius: color
                for (radius, _, _), color in zip(
                    above, _graded_shades(BROWN_COLOR, len(above))
                )
            }
        )

        for radius, extra_temperatures, extra_ratios in extra:
            color = color_by_radius[radius]
            if np.any(extra_ratios <= 0.0):
                raise ValueError("variance ratios must be positive for a log scale")
            axis_minimum = min(axis_minimum, float(np.min(extra_ratios)))
            # Tiny circles identify the non-chosen data points while the
            # larger diamond keeps the chosen radius visually prominent.
            axis.plot(
                extra_temperatures,
                extra_ratios,
                color=color,
                lw=0.9,
                marker="o",
                markersize=EXTRA_RADIUS_MARKER_SIZE,
                markeredgewidth=0.0,
                zorder=2,
            )
            legend_entries.append((radius, color, "o"))

        # Sort by radius so the legend reads in increasing order regardless
        # of where the dimension's own Approximation 3 radius falls.
        legend_handles = [
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                markersize=3.2 if radius == main_radius else EXTRA_RADIUS_MARKER_SIZE,
                markeredgewidth=0.0,
                linewidth=1.0,
                label=rf"$R={radius}$",
            )
            for radius, color, marker in sorted(legend_entries)
        ]

        axis.axhline(1.0, color="black", lw=0.8, zorder=1)
        axis.set_yscale("log")
        axis.set_ylim(bottom=axis_minimum / 1.5, top=1.2)
        axis.yaxis.set_minor_formatter(NullFormatter())
        if axis_minimum > 0.01:
            ticks = [
                tick
                for tick in (0.1, 0.2, 0.5, 1.0)
                if axis.get_ylim()[0] < tick < axis.get_ylim()[1]
            ]
            axis.yaxis.set_major_locator(FixedLocator(ticks))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        axis.set_title(rf"$d={dim},\ L={lattice_size}$", pad=3)
        axis.set_axisbelow(True)
        axis.grid(color="0.8", lw=0.5, alpha=0.5)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

        if legend_handles:
            if extra and dim == 1:
                # The lower-left corner stays clear of the 1D XY curves.
                legend_position = {
                    "loc": "lower left",
                    "bbox_to_anchor": (0.02, 0.02),
                    "bbox_transform": axis.transAxes,
                }
            else:
                legend_position = {
                    "loc": "upper right",
                    "bbox_to_anchor": (1.0, 0.93),
                    "bbox_transform": blended_transform_factory(
                        axis.transAxes, axis.transData
                    ),
                }
            axis.legend(
                handles=legend_handles,
                **legend_position,
                ncols=2,
                frameon=False,
                fontsize=6.5,
                handlelength=1.3,
                handletextpad=0.3,
                columnspacing=0.9,
                labelspacing=0.18,
                borderpad=0.1,
                borderaxespad=0.3,
            )

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
    extra_series_by_dim: dict[int, list[tuple[int, np.ndarray, np.ndarray]]]
    | None = None,
    main_radius_by_dim: dict[int, int] | None = None,
) -> None:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(dims),
        sharex="col",
        sharey=False,
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
        extra_series_by_dim=extra_series_by_dim,
        main_radius_by_dim=main_radius_by_dim,
    )
    axes[0, 0].set_ylabel(
        r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$"
    )
    fig.supxlabel(r"$T=\beta^{-1}$")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=0.0, h_pad=0.0)
    fig.subplots_adjust(wspace=0.26, bottom=0.26, top=0.78)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    xy_dims, xy_lattice_sizes, xy_betas, xy_ratios = _load_xy_approx3()
    cp2_dims, cp2_lattice_sizes, cp2_betas, cp2_ratios = _load_cp2_approx3()

    xy_extra_series = _load_xy_mask_radius_series(XY_MASK_RADIUS_EXCLUDE)
    # The radius already used for each dimension's own Approximation 3
    # series, matching the xy_opf README (R=20 in d=1, R=6 in d=2, 3).
    xy_main_radius = {1: 20, 2: 6, 3: 6}

    _make_panel(
        xy_dims,
        xy_lattice_sizes,
        xy_betas,
        xy_ratios,
        XY_CRITICAL_TEMPERATURES,
        FIGURES_DIRECTORY / "xy_scaling_top.pdf",
        extra_series_by_dim=xy_extra_series,
        main_radius_by_dim=xy_main_radius,
    )
    # The mask radius used for every CP(2) dimension's own Approximation 3
    # series (confirmed from the saved artifacts' metadata).
    cp2_main_radius = {1: 6, 2: 6, 3: 6}

    _make_panel(
        cp2_dims,
        cp2_lattice_sizes,
        cp2_betas,
        cp2_ratios,
        GAUGE_CRITICAL_TEMPERATURES,
        FIGURES_DIRECTORY / "cp2_scaling_top.pdf",
        main_radius_by_dim=cp2_main_radius,
    )

    for path in (
        FIGURES_DIRECTORY / "xy_scaling_top.pdf",
        FIGURES_DIRECTORY / "cp2_scaling_top.pdf",
    ):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
