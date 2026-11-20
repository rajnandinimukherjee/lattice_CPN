"""Render appendix one-point variance ratios and source-field predictions.

The two XY d=1 Approximation 3 points that saturated the original alpha=5
bound are read from the independently saved R=20 physical-radius scan, whose
scale search used |alpha| <= 40 on the same ensembles and Laplace profiles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator, NullFormatter
import numpy as np

from cpn_opf.config import GAUGE_CRITICAL_TEMPERATURES
from cpn_opf.study import _load_dimension as load_cp2_dimension
from csft_opf.persistence import training_files_for_dimension as scalar_training_files
from plot_settings import OBJECTIVE_COLORS, UNDEFORMED_COLOR, fig_width, latex_font_params
from xy_opf.config import XY_CRITICAL_TEMPERATURES
from xy_opf.study import _load_dimension as load_xy_dimension

FIGS = Path("paper/figs")
LOSSES = ("variance", "approx3")
XY_RADII = {1: 20, 2: 6, 3: 6}

STYLES = {
    "variance": dict(color=OBJECTIVE_COLORS["variance"], marker="o", label="Direct variance"),
    "approx3": dict(color=OBJECTIVE_COLORS["approx3"], marker="D", label="Approx. 3"),
}

plt.rcParams.update(latex_font_params)
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})


def load_xy() -> list[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    by_dim = []
    for dim in (1, 2, 3):
        ratios = {loss: [] for loss in LOSSES}
        fields = {loss: [] for loss in LOSSES}
        betas = {loss: [] for loss in LOSSES}
        load_xy_dimension(
            args=argparse.Namespace(plot_training=False),
            ndim=dim,
            loss_functions=list(LOSSES),
            variance_ratios=ratios,
            deformations_at_source=fields,
            betas_by_loss=betas,
            training_figures=[],
        )
        curves = {}
        for loss in LOSSES:
            curve_betas = np.asarray(betas[loss][0], dtype=float)
            curve_ratios = np.asarray(ratios[loss][0], dtype=float)
            curve_sources = np.asarray(fields[loss][0], dtype=float)
            if loss == "approx3" and dim == 1:
                for beta in (1.84, 2.30):
                    path = Path("training_data/xy_physical_radius") / (
                        f"xy_physrad_dim1_b{beta:.2f}_R20_loss-approx3_result.npy"
                    )
                    saved = np.load(path, allow_pickle=True).item()
                    meta = saved["metadata"]
                    if (tuple(meta["alpha_bounds"]) != (-40.0, 40.0)
                            or meta["mask_radius"] != XY_RADII[dim]):
                        raise ValueError(f"unexpected corrected XY result: {path}")
                    index = np.flatnonzero(np.isclose(curve_betas, beta))
                    if len(index) != 1:
                        raise ValueError(f"missing XY beta={beta:g} in appendix scan")
                    curve_ratios[index[0]] = saved["variance_ratio"]
                    curve_sources[index[0]] = saved["metadata"]["field_at_source"]
            order = np.argsort(1.0 / curve_betas)
            curves[loss] = (
                1.0 / curve_betas[order], curve_ratios[order], curve_sources[order]
            )
        by_dim.append(curves)
    return by_dim


def load_cp2() -> list[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    by_dim = []
    for dim, lattice_size in zip((1, 2, 3), (128, 64, 16)):
        ratios = {loss: [] for loss in LOSSES}
        fields = {loss: [] for loss in LOSSES}
        betas = {loss: [] for loss in LOSSES}
        load_cp2_dimension(
            argparse.Namespace(N=2, i=0, j=1, action="gauge",
                               plot_training=False, save_plot=None),
            dim, lattice_size, list(LOSSES), ratios, fields, betas,
        )
        curves = {}
        for loss in LOSSES:
            curve_betas = np.asarray(betas[loss][0], dtype=float)
            curve_ratios = np.asarray(ratios[loss][0], dtype=float)
            curve_sources = np.asarray(fields[loss][0], dtype=float)
            order = np.argsort(1.0 / curve_betas)
            curves[loss] = (
                1.0 / curve_betas[order], curve_ratios[order], curve_sources[order]
            )
        by_dim.append(curves)
    return by_dim


def load_scalar() -> list[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    by_dim = []
    for dim in (1, 2, 3):
        curves = {}
        for loss in LOSSES:
            artifacts = scalar_training_files(dim, loss)
            if not artifacts:
                raise FileNotFoundError(f"no scalar {loss} results for d={dim}")
            masses = []
            ratios = []
            sources = []
            for mass, path in artifacts:
                saved = np.load(path, allow_pickle=True).item()
                field = np.asarray(saved["scaled_field"], dtype=float)
                source = tuple(saved["metadata"]["source"])
                masses.append(mass)
                ratios.append(float(saved["variance_ratio"]))
                sources.append(float(field[*source, 0]))
            curves[loss] = (
                np.asarray(masses), np.asarray(ratios), np.asarray(sources)
            )
        if not np.array_equal(curves["variance"][0], curves["approx3"][0]):
            raise ValueError(f"scalar direct and Approximation 3 masses differ in d={dim}")
        by_dim.append(curves)
    return by_dim


def render_row(
    curves_by_dim: list[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    lattice_sizes: tuple[int, int, int],
    critical_temperatures: dict[int, float] | None,
    output: Path,
    *,
    scalar: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 0.36 * fig_width))
    for dim, (axis, lattice_size, curves) in enumerate(
        zip(axes, lattice_sizes, curves_by_dim), start=1
    ):
        axis_minimum = float("inf")
        for loss in LOSSES:
            temperatures, ratios, sources = curves[loss]
            source_prediction = np.exp(-sources)
            if (np.any(ratios <= 0) or np.any(temperatures <= 0)
                    or not np.isfinite(source_prediction).all()
                    or np.any(source_prediction <= 0)):
                raise ValueError(f"invalid {loss} scan for d={dim}")
            style = STYLES[loss]
            axis_minimum = min(axis_minimum, float(ratios.min()),
                               float(source_prediction.min()))
            axis.scatter(temperatures, ratios, s=16, color=style["color"],
                         marker=style["marker"], zorder=3)
            axis.scatter(temperatures, source_prediction, s=26,
                         facecolors="none", edgecolors=style["color"],
                         linewidths=1.0, marker=style["marker"], zorder=4)
        all_temperatures = np.concatenate([curves[loss][0] for loss in LOSSES])
        t_min, t_max = float(all_temperatures.min()), float(all_temperatures.max())
        padding = 0.035 * (t_max - t_min)
        axis.set_xlim(max(0, t_min - padding), t_max + padding)
        axis.axhline(1.0, color=UNDEFORMED_COLOR, lw=0.65, zorder=1)
        axis.set_yscale("log")
        axis.set_ylim(bottom=axis_minimum / 1.5, top=1.2)
        axis.yaxis.set_minor_formatter(NullFormatter())
        if axis_minimum > 0.01:
            ticks = [tick for tick in (0.1, 0.2, 0.5, 1.0)
                     if axis.get_ylim()[0] < tick < axis.get_ylim()[1]]
            axis.yaxis.set_major_locator(FixedLocator(ticks))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        axis.set_title(rf"$d={dim},\ L={lattice_size}$", pad=3)
        axis.grid(color="0.8", lw=0.5, alpha=0.5)
        axis.set_axisbelow(True)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        if scalar:
            axis.set_xlim(-0.035, 1.04)
            axis.set_xticks([0.25, 0.5, 0.75, 1.0])
            axis.axvspan(0.0, axis.get_xlim()[1],
                         facecolor="0.96", edgecolor="none", zorder=-1)
            axis.axvline(0.0, color="black", ls=":", lw=0.8, zorder=1)
            top = axis.secondary_xaxis("top")
            top.set_xticks([0.0], labels=[r"$a=0$"])
            top.tick_params(axis="x", pad=1, labelsize=6)
        critical = None if critical_temperatures is None else critical_temperatures.get(dim)
        if critical is not None:
            if critical < axis.get_xlim()[0]:
                axis.set_xlim(left=critical)
            if critical > axis.get_xlim()[1]:
                axis.set_xlim(right=critical)
            axis.axvspan(critical, axis.get_xlim()[1],
                         facecolor="0.96", edgecolor="none", zorder=-1)
            axis.axvline(critical, color="black", ls=":", lw=0.8, zorder=1)
            top = axis.secondary_xaxis("top")
            top.set_xticks([critical], labels=[rf"$T_{{\rm c}}={critical:.3g}$"])
            top.tick_params(axis="x", pad=1, labelsize=6)
    axes[0].set_ylabel(
        r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$"
    )
    fig.supxlabel(r"$am$" if scalar else r"$T=\beta^{-1}$", y=0.08)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=0.0, h_pad=0.0)
    fig.subplots_adjust(wspace=0.26, bottom=0.26, top=0.78)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_legend() -> None:
    handles = []
    for loss in LOSSES:
        style = STYLES[loss]
        for open_marker, quantity in ((False, "ratio"), (True, r"$e^{-F_0}$")):
            handles.append(Line2D(
                [], [], color=style["color"], marker=style["marker"],
                markerfacecolor="none" if open_marker else style["color"],
                markeredgewidth=1.0, ls="none", markersize=4.5,
                label=f'{style["label"]}: {quantity}',
            ))
    fig = plt.figure(figsize=(fig_width, 0.30))
    fig.legend(handles=handles, loc="center", ncols=4, frameon=False,
               fontsize=7.5, columnspacing=0.9, handletextpad=0.35)
    path = FIGS / "opf_scaling_legend.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".png"), dpi=220,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    render_row(load_xy(), (64, 32, 16), XY_CRITICAL_TEMPERATURES,
               FIGS / "xy_scaling_ratio.pdf")
    render_row(load_cp2(), (128, 64, 16), GAUGE_CRITICAL_TEMPERATURES,
               FIGS / "cp2_scaling_ratio.pdf")
    render_row(load_scalar(), (64, 32, 16), None,
               FIGS / "csft_scaling_ratio.pdf", scalar=True)
    render_legend()


if __name__ == "__main__":
    main()
