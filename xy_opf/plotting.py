"""Summary plots for XY OPF scaling studies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_settings import fig_width, latex_font_params

from .config import (
    LOSS_LABELS,
    LOSS_MARKERS,
    XY_CRITICAL_TEMPERATURES,
)

SCALING_STYLES = {
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
}


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


def plot_scaling(
    dims: list[int],
    lattice_sizes: list[str],
    betas_by_loss: dict[str, list[list[float]]],
    variance_ratios: dict[str, list[np.ndarray]],
    deformations_at_source: dict[str, list[np.ndarray]],
    loss_functions: list[str],
    output_path: str | Path | None = None,
    show: bool = True,
    legend_output_path: str | Path | None = None,
) -> None:
    """Plot scaling panels and save a reusable standalone strategy legend."""
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(dims),
        sharex="col",
        sharey="row",
        squeeze=False,
        figsize=(fig_width, 0.68 * fig_width),
    )
    for dim_idx, dim in enumerate(dims):
        ratio_ax = axes[0, dim_idx]
        field_ax = axes[1, dim_idx]
        all_betas = [
            beta
            for loss_function in loss_functions
            for beta in betas_by_loss[loss_function][dim_idx]
        ]
        critical_temperature = (
            XY_CRITICAL_TEMPERATURES.get(dim)
            if len(set(all_betas)) > 1
            else None
        )
        for loss_function in loss_functions:
            style = SCALING_STYLES.get(loss_function)
            if style is None:
                style = {
                    "color": "black",
                    "marker": LOSS_MARKERS[loss_function],
                    "label": LOSS_LABELS[loss_function],
                }
            betas = np.asarray(
                betas_by_loss[loss_function][dim_idx],
                dtype=float,
            )
            if np.any(betas <= 0.0):
                raise ValueError(
                    "beta values must be positive to plot T=1/beta"
                )
            temperatures = 1.0 / betas
            ratios = np.asarray(
                variance_ratios[loss_function][dim_idx],
                dtype=float,
            )
            source_values = np.asarray(
                deformations_at_source[loss_function][dim_idx],
                dtype=float,
            )
            ratio_ax.scatter(
                temperatures,
                ratios,
                color=style["color"],
                marker=style["marker"],
                s=20,
            )
            field_ax.scatter(
                temperatures,
                source_values,
                color=style["color"],
                marker=style["marker"],
                s=20,
            )

        ratio_ax.axhline(1.0, color="black", lw=0.8, zorder=1)
        ratio_ax.set_title(
            rf"$d={dim},\ L={lattice_sizes[dim_idx]}$",
            pad=3,
        )
        for plot_ax in (ratio_ax, field_ax):
            plot_ax.set_axisbelow(True)
            plot_ax.grid(color="0.8", lw=0.5, alpha=0.5)
            if dim_idx > 0:
                plot_ax.spines["left"].set_visible(False)
                plot_ax.tick_params(axis="y", left=False, labelleft=False)
        field_ax.xaxis.set_major_locator(
            MaxNLocator(nbins=5, prune="both")
        )

        if critical_temperature is not None and critical_temperature >= 0:
            x_limits = ratio_ax.get_xlim()
            if critical_temperature < x_limits[0]:
                ratio_ax.set_xlim(left=critical_temperature)
                x_limits = ratio_ax.get_xlim()
            for plot_ax in (ratio_ax, field_ax):
                plot_ax.axvline(
                    critical_temperature,
                    color="black",
                    ls=":",
                    lw=0.8,
                    zorder=1,
                )
            for plot_ax in (ratio_ax, field_ax):
                plot_ax.axvspan(
                    critical_temperature,
                    x_limits[1],
                    facecolor="0.96",
                    edgecolor="none",
                    zorder=-1,
                )
            ratio_ax.set_xlim(x_limits)
            critical_axis = ratio_ax.secondary_xaxis("top")
            critical_axis.set_xticks(
                [critical_temperature],
                labels=[
                    rf"$T_{{\rm c}}={critical_temperature:.3g}$"
                ],
            )
            critical_axis.tick_params(axis="x", pad=1, labelsize=6)

    axes[0, 0].set_ylabel(
        r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$"
    )
    axes[1, 0].set_ylabel(r"$F_0$")
    fig.supxlabel(r"$T=\beta^{-1}$")

    legend_handles = [
        Line2D(
            [],
            [],
            color=SCALING_STYLES[loss_function]["color"],
            marker=SCALING_STYLES[loss_function]["marker"],
            markersize=3.2,
            linewidth=1.0,
            label=SCALING_STYLES[loss_function]["label"],
        )
        if loss_function in SCALING_STYLES
        else Line2D(
            [],
            [],
            color="black",
            marker=LOSS_MARKERS[loss_function],
            markersize=3.2,
            linewidth=1.0,
            label=LOSS_LABELS[loss_function],
        )
        for loss_function in loss_functions
    ]
    legend_fig = plt.figure(figsize=(fig_width, 0.32))
    legend_fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="center",
        ncols=len(legend_handles),
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.35,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=0.0, h_pad=0.0)
    fig.subplots_adjust(wspace=0.0, hspace=0.0, bottom=0.13, top=0.82)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
        if legend_output_path is None:
            legend_output_path = output_path.with_name(
                f"{output_path.stem}_legend{output_path.suffix}"
            )
        legend_output_path = Path(legend_output_path)
        legend_output_path.parent.mkdir(parents=True, exist_ok=True)
        legend_fig.savefig(
            legend_output_path,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        legend_fig.savefig(
            legend_output_path.with_suffix(".png"),
            dpi=220,
            bbox_inches="tight",
            pad_inches=0.02,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(legend_fig)
