"""Summary plots for complex-scalar mass-scaling studies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_settings import OBJECTIVE_COLORS, UNDEFORMED_COLOR, fig_width, latex_font_params

from .config import LOSS_LABELS, LOSS_MARKERS

SCALING_STYLES = {
    "variance": {"color": OBJECTIVE_COLORS["variance"], "marker": "o", "label": "direct variance"},
    "penalty": {"color": OBJECTIVE_COLORS["penalty"], "marker": "s", "label": "Approx. 1"},
    "saddle-point": {"color": OBJECTIVE_COLORS["saddle-point"], "marker": "^", "label": "Approx. 2"},
    "approx3": {"color": OBJECTIVE_COLORS["approx3"], "marker": "D", "label": "Approx. 3"},
}

plt.rcParams.update(latex_font_params)


def plot_scaling(
    dims: list[int],
    lattice_sizes: list[str],
    masses_by_loss: dict[str, list[list[float]]],
    variance_ratios: dict[str, list[np.ndarray]],
    deformations_at_source: dict[str, list[np.ndarray]],
    loss_functions: list[str],
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot variance ratios and source fields directly against bare mass ``am``."""
    fig, axes = plt.subplots(
        2,
        len(dims),
        sharex="col",
        sharey="row",
        squeeze=False,
        figsize=(fig_width, 0.56 * fig_width),
    )
    for dim_idx, dim in enumerate(dims):
        for loss_function in loss_functions:
            style = SCALING_STYLES.get(
                loss_function,
                {
                    "color": "black",
                    "marker": LOSS_MARKERS[loss_function],
                    "label": LOSS_LABELS[loss_function],
                },
            )
            masses = np.asarray(masses_by_loss[loss_function][dim_idx], dtype=float)
            axes[0, dim_idx].scatter(
                masses,
                variance_ratios[loss_function][dim_idx],
                color=style["color"],
                marker=style["marker"],
                s=20,
            )
            axes[1, dim_idx].scatter(
                masses,
                deformations_at_source[loss_function][dim_idx],
                color=style["color"],
                marker=style["marker"],
                s=20,
            )
        axes[0, dim_idx].axhline(1.0, color=UNDEFORMED_COLOR, lw=0.8)
        axes[0, dim_idx].set_title(
            rf"$d={dim},\ L={lattice_sizes[dim_idx]}$"
        )
        axes[1, dim_idx].xaxis.set_major_locator(MaxNLocator(nbins=5))
        for axis in axes[:, dim_idx]:
            axis.grid(color="0.8", lw=0.5, alpha=0.5)
    axes[0, 0].set_ylabel(
        r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$"
    )
    axes[1, 0].set_ylabel(r"$F_0$")
    fig.supxlabel(r"$am$")
    handles = [
        Line2D(
            [],
            [],
            color=SCALING_STYLES.get(loss, {"color": "black"})["color"],
            marker=SCALING_STYLES.get(loss, {"marker": LOSS_MARKERS[loss]})["marker"],
            label=SCALING_STYLES.get(loss, {"label": LOSS_LABELS[loss]})["label"],
        )
        for loss in loss_functions
    ]
    fig.legend(handles=handles, loc="upper center", ncols=len(handles), frameon=False)
    # Explicit margins are stable even when a mass scan currently contains a
    # single point (``tight_layout`` can produce NaN axes extents for that
    # degenerate shared-axis case on some Matplotlib versions).
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.16,
        top=0.82,
        wspace=0.08,
        hspace=0.08,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
