"""Summary plotting for CP(N) OPF scaling studies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_settings import fig_width, latex_font_params

from .config import (
    GAUGE_CRITICAL_TEMPERATURES,
    LOSS_LABELS,
    LOSS_MARKERS,
)
from .persistence import load_deformation_field


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
    "approx2": {
        "color": "tab:green",
        "marker": "^",
        "label": "Approx. 2 (small fluc.)",
    },
    "approx3": {
        "color": "tab:red",
        "marker": "D",
        "size": 12,
        "label": "Approx. 3 (smooth deformation)",
    },
}


def plot_deformation_fields(
    artifacts: list[tuple[float, Path]],
    *,
    n: int,
    ndim: int,
    lattice_size: int,
    i: int,
    j: int,
    loss_function: str,
    action_type: str = "gauge",
    filename: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot every saved deformation channel, with one beta per figure row.

    One-dimensional fields are shown as single-row heatmaps.  For a 3D
    field, the displayed plane fixes the final coordinate at the source, so
    the slice passes through the OPF insertion.
    """
    if ndim not in (1, 2, 3):
        raise ValueError(
            "deformation-field plots require dimension 1, 2, or 3")
    if not artifacts:
        raise ValueError("at least one training artifact is required")

    fields_and_sources = [
        (float(beta), *load_deformation_field(path)) for beta, path in artifacts
    ]
    expected_shape = (lattice_size,) * ndim + (n + 1,)
    for beta, field, _ in fields_and_sources:
        if field.shape != expected_shape:
            raise ValueError(
                f"field for beta={beta:g} has shape {field.shape}; "
                f"expected {expected_shape}"
            )

    values = np.concatenate([field.ravel()
                            for _, field, _ in fields_and_sources])
    limit = float(np.max(np.abs(values)))
    if limit == 0.0:
        limit = 1.0
    nrows = len(fields_and_sources)
    ncols = n + 1
    # Size the plotting area in physical units so square 2D images do not
    # shrink inside an overly wide subplot cell for long beta scans.  1D
    # fields intentionally use short, wide cells.
    panel_width = 2.0
    panel_height = 0.8 if ndim == 1 else panel_width
    row_column_gap = 0.06
    grid_width = panel_width * (ncols + (ncols - 1) * row_column_gap)
    grid_height = panel_height * (nrows + (nrows - 1) * row_column_gap)
    left_margin = 0.75
    bottom_margin = 0.55
    top_margin = 0.65
    colorbar_gap = 0.30
    colorbar_width = 0.20
    right_margin = 0.25
    figure_width = (
        left_margin
        + grid_width
        + colorbar_gap
        + colorbar_width
        + right_margin
    )
    figure_height = bottom_margin + grid_height + top_margin
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        squeeze=False,
        figsize=(figure_width, figure_height),
        sharex=True,
        sharey=True,
    )
    image = None
    for row, (beta, field, source) in enumerate(fields_and_sources):
        for channel, axis in enumerate(axes[row]):
            channel_field = field[..., channel]
            if ndim == 1:
                image = axis.imshow(
                    channel_field[None, :], aspect="auto", origin="lower",
                    cmap="coolwarm", vmin=-limit, vmax=limit,
                )
                axis.set_yticks([])
            else:
                if ndim == 3:
                    channel_field = channel_field[..., source[-1]]
                image = axis.imshow(
                    channel_field, origin="lower", cmap="coolwarm",
                    vmin=-limit, vmax=limit,
                )
            if row == 0:
                axis.set_title(rf"$\alpha_{{{channel}}}(x)$")
            if channel == 0:
                axis.set_ylabel(rf"$\beta={beta:g}$")
        if ndim > 1:
            axes[row, 0].set_ylabel(rf"$\beta={beta:g}$")

    assert image is not None
    for axis in axes.flat:
        axis.label_outer()
    fig.supxlabel(r"$x_0$")
    if ndim > 1:
        fig.supylabel(r"$x_1$")
    fig.subplots_adjust(
        left=left_margin / figure_width,
        right=(left_margin + grid_width) / figure_width,
        bottom=bottom_margin / figure_height,
        top=1.0 - top_margin / figure_height,
        wspace=row_column_gap,
        hspace=row_column_gap,
    )
    colorbar_axis = fig.add_axes(
        (
            (left_margin + grid_width + colorbar_gap) / figure_width,
            bottom_margin / figure_height,
            colorbar_width / figure_width,
            grid_height / figure_height,
        )
    )
    fig.colorbar(image, cax=colorbar_axis, label=r"deformation $\alpha$")
    slice_note = (
        rf", slice $x_2={fields_and_sources[0][2][-1]}$" if ndim == 3 else ""
    )
    fig.suptitle(
        rf"CP({n}), {action_type.title()}, $d={ndim}$, $L={lattice_size}$"
        rf", $i={i}$, $j={j}${slice_note}"
        + f", loss function: {loss_function}",
        y=0.99,
    )
    if filename is not None:
        output = Path(filename)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_scaling(
    dims: list[int],
    lattice_sizes: list[int],
    betas_by_loss: dict[str, list[list[float]]],
    variance_ratios: dict[str, list[np.ndarray]],
    deformations_at_source: dict[str, list[np.ndarray]],
    loss_functions: list[str],
    *,
    n: int,
    i: int,
    j: int,
    action_type: str = "gauge",
    filename: str | Path | None = None,
    save_legend: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Plot variance ratios and source deformation versus temperature."""
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(dims),
        sharex="col",
        sharey="row",
        squeeze=False,
        figsize=(fig_width, 0.56 * fig_width),
    )
    for dim_idx, ndim in enumerate(dims):
        ratio_ax, field_ax = axes[:, dim_idx]
        all_betas = [
            beta
            for loss_function in loss_functions
            for beta in betas_by_loss[loss_function][dim_idx]
        ]
        for loss_function in loss_functions:
            style = SCALING_STYLES.get(
                loss_function,
                {
                    "color": "black",
                    "marker": LOSS_MARKERS[loss_function],
                    "label": LOSS_LABELS[loss_function],
                },
            )
            betas = np.asarray(
                betas_by_loss[loss_function][dim_idx], dtype=float)
            if np.any(betas <= 0.0):
                raise ValueError(
                    "beta values must be positive to plot T=1/beta"
                )
            temperatures = 1.0 / betas
            ratios = np.asarray(
                variance_ratios[loss_function][dim_idx], dtype=float
            )
            source_values = np.asarray(
                deformations_at_source[loss_function][dim_idx], dtype=float
            )
            ratio_ax.scatter(
                temperatures,
                ratios,
                color=style["color"],
                marker=style["marker"],
                s=style.get("size", 20),
            )
            field_ax.scatter(
                temperatures,
                source_values,
                color=style["color"],
                marker=style["marker"],
                s=style.get("size", 20),
            )
        ratio_ax.axhline(1.0, color="black", lw=0.8, zorder=1)
        ratio_ax.set_title(
            rf"$d={ndim},\ L={lattice_sizes[dim_idx]}$",
            pad=3,
        )
        for axis in (ratio_ax, field_ax):
            axis.set_axisbelow(True)
            axis.grid(color="0.8", lw=0.5, alpha=0.5)
            if dim_idx > 0:
                axis.tick_params(axis="y", left=True, labelleft=False)
        field_ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

        critical_temperature = (
            GAUGE_CRITICAL_TEMPERATURES.get(ndim)
            if action_type == "gauge" and len(set(all_betas)) > 1
            else None
        )
        if critical_temperature is not None:
            x_limits = ratio_ax.get_xlim()
            if critical_temperature < x_limits[0]:
                ratio_ax.set_xlim(left=critical_temperature)
                x_limits = ratio_ax.get_xlim()
            for axis in (ratio_ax, field_ax):
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
            ratio_ax.set_xlim(x_limits)
            critical_axis = ratio_ax.secondary_xaxis("top")
            critical_axis.set_xticks(
                [critical_temperature],
                labels=[rf"$T_{{\rm c}}={critical_temperature:.3g}$"],
            )
            critical_axis.tick_params(axis="x", pad=1, labelsize=6)

    axes[0, 0].set_ylabel(
        r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$"
    )
    axes[1, 0].set_ylabel(r"$F_0$")
    fig.supxlabel(r"$T=\beta^{-1}$")

    handles = [
        Line2D(
            [],
            [],
            color=SCALING_STYLES[loss_function]["color"],
            marker=SCALING_STYLES[loss_function]["marker"],
            markersize=3.2,
            linewidth=1.0,
            label=SCALING_STYLES[loss_function]["label"],
        )
        for loss_function in loss_functions
    ]
    legend_fig = None
    if save_legend:
        legend_fig = plt.figure(figsize=(fig_width, 0.32))
        legend_fig.legend(
            handles,
            [handle.get_label() for handle in handles],
            loc="center",
            ncols=len(handles),
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.35,
        )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=0.0, h_pad=0.0)
    fig.subplots_adjust(wspace=0.025, hspace=0.04, bottom=0.13, top=0.82)
    if filename is not None:
        output = Path(filename)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
        if legend_fig is not None:
            legend_output = output.with_name(
                f"{output.stem}_legend{output.suffix}"
            )
            legend_output.parent.mkdir(parents=True, exist_ok=True)
            legend_fig.savefig(
                legend_output, bbox_inches="tight", pad_inches=0.02
            )
            legend_fig.savefig(
                legend_output.with_suffix(".png"),
                dpi=220,
                bbox_inches="tight",
                pad_inches=0.02,
            )
    if show:
        plt.show()
    else:
        plt.close(fig)
        if legend_fig is not None:
            plt.close(legend_fig)
    return fig
