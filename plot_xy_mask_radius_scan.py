"""Combine the XY periodic-support-radius scans for d=1, 2, 3 into one figure.

Each panel is produced by ``xy_mask_radius_experiment.py --dim {1,2,3}``,
which trains the full variance objective at a sequence of periodic support
radii ``R`` and also solves the deterministic Approximation 3 profile at
each radius.  This script only loads the three saved scan summaries and
draws them side by side, sharing a log-scale y-axis so the very different
radius ranges (set by each ensemble's lattice size) and R-dependence can be
compared directly: d=1 improves with R without bound (vanishing lattice
capacity, 2/R), d=2 improves logarithmically, and d=3's capacity saturates.
Each panel also overlays the closed-form Approximation-3 scaling derived
from treating it as a free massless lattice scalar sourced at the origin
and IR-regulated at the mask radius (``theoretical_ratio``).

Run, for example:

    python xy_mask_radius_experiment.py --dim 1 --beta 1.84 --run
    python xy_mask_radius_experiment.py --dim 2 --beta 0.65 --run
    python xy_mask_radius_experiment.py --dim 3 --beta 0.40 --run
    python plot_xy_mask_radius_scan.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lattice-mplconfig")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

from plot_settings import fig_width, latex_font_params

SCAN_DATA_DIRECTORY = Path("training_data/xy_mask_radius_scan")
FIGURES_DIRECTORY = Path("paper/figs")
OUTPUT_STEM = "xy_mask_radius_scan"

# (dimension, beta, lattice size) for the three panels, left to right.
DEFAULT_PANELS = ((1, 1.84, 64), (2, 0.65, 32), (3, 0.40, 16))

# Lattice constant for the d=3 simple-cubic free-propagator floor G_infinity
# at a=1 (numerically exact value, e.g. Cserti's tabulated lattice Green's
# functions); see the "XY one-point variance reduction vs. mask radius"
# derivation notes.
G_INFINITY_D3 = 0.2527


def theoretical_ratio(ndim: int, beta: float, radii: np.ndarray) -> np.ndarray:
    """Closed-form Approximation-3 scaling of Var[Q]/Var[O] with R/a.

    Approximation 3 is a free massless lattice scalar sourced at the origin
    and forced to vanish on the mask boundary at radius R; its on-shell
    action E[f](R) is that propagator's IR-regulated value at the source.
    The regulator is linearly IR-divergent in d=1 (exponential suppression
    in R), logarithmically divergent in d=2 (power law in R/a with the
    model's own spin-wave exponent eta=1/(2*pi*beta)), and IR-safe in d=3
    (a beta-dependent floor approached as (a/R)^{d-2}).
    """
    radii = np.asarray(radii, dtype=float)
    if ndim == 1:
        return np.exp(-(radii + 1.0) / (2.0 * beta))
    if ndim == 2:
        return radii ** (-1.0 / (2.0 * np.pi * beta))
    if ndim == 3:
        return np.exp(-(G_INFINITY_D3 / beta) * (1.0 - 1.0 / radii))
    raise ValueError(f"no closed-form theory curve implemented for d={ndim}")

plt.rcParams.update(latex_font_params)
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def summary_path(ndim: int, beta: float, lattice_size: int) -> Path:
    """Return the scan summary path written by xy_mask_radius_experiment.py."""
    return SCAN_DATA_DIRECTORY / (
        f"xy_mask_radius_scan_dim{ndim}_b{beta:.2f}_L{lattice_size}_summary.npy"
    )


def load_summary(ndim: int, beta: float, lattice_size: int) -> dict[str, object]:
    """Load one dimension's saved radius-scan summary."""
    path = summary_path(ndim, beta, lattice_size)
    saved = np.load(path, allow_pickle=True)
    if saved.shape != () or not isinstance(saved.item(), dict):
        raise ValueError(f"invalid scan summary: {path}")
    return saved.item()


def save_legend(output_path: Path) -> None:
    """Save the shared legend strip placed above the three panels."""
    handles = [
        Line2D(
            [],
            [],
            color="tab:blue",
            marker="o",
            markersize=3.2,
            linewidth=1.0,
            label="direct variance",
        ),
        Line2D(
            [],
            [],
            color="tab:red",
            marker="D",
            markersize=3.4,
            linewidth=1.0,
            label="Approx. 3 (smooth deformation)",
        ),
        Line2D(
            [],
            [],
            color="0.2",
            linestyle="-.",
            linewidth=1.1,
            label="Approx. 3 (closed form)",
        ),
        Line2D(
            [],
            [],
            color="0.45",
            linestyle=":",
            linewidth=0.9,
            label=r"chosen $R/a$",
        ),
    ]
    figure = plt.figure(figsize=(fig_width, 0.4))
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="center",
        ncols=4,
        frameon=False,
        columnspacing=1.3,
        handletextpad=0.4,
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


def _panel_data(summary: dict[str, object]) -> dict[str, object]:
    """Extract the arrays and metadata ``plot_panel`` needs from a summary."""
    radii = np.asarray(summary["radii"], dtype=float)
    ratios = np.asarray(summary["variance_ratios"], dtype=float)
    approx3 = np.asarray(summary["approx3_variance_ratios"], dtype=float)
    if ratios.ndim != 2 or len(radii) != len(ratios):
        raise ValueError("variance ratios must be one row per scanned radius")
    if not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise ValueError("the scan contains invalid trained variance ratios")
    if not np.isfinite(approx3).all() or np.any(approx3 <= 0):
        raise ValueError("the scan contains invalid Approximation 3 ratios")
    metadata = summary["metadata"]
    return {
        "radii": radii,
        # The ratio is the minimum of a non-convex objective, so the
        # attainable value is the best of the repeated restarts.
        "best": ratios.min(axis=1),
        "approx3": approx3,
        "ndim": int(metadata["ndim"]),
        "beta": float(metadata["beta"]),
        "lattice_size": int(metadata["lattice_size"]),
        "published_radius": int(metadata["published_mask_radius"]),
    }


def plot_panel(axis: plt.Axes, data: dict[str, object]) -> None:
    """Draw one dimension's variance ratio against the support radius.

    Only $d=1$ spans enough decades to need a log-scale $y$-axis (the
    lattice capacity vanishes as $R$ grows, Sec.~\\ref{sec:results}); $d=2,3$
    stay within a factor of a few of 1 and are clearer on a linear scale.
    """
    radii = data["radii"]
    axis.axhline(1.0, color="black", linewidth=0.8, zorder=1)
    axis.axvline(
        data["published_radius"], color="0.45", linestyle=":", linewidth=0.9, zorder=1
    )
    axis.plot(
        radii,
        data["best"],
        color="tab:blue",
        marker="o",
        markersize=3.2,
        linestyle="-",
        linewidth=1.0,
        zorder=4,
    )
    axis.plot(
        radii,
        data["approx3"],
        color="tab:red",
        marker="D",
        markersize=3.4,
        linestyle="-",
        linewidth=1.0,
        zorder=3,
    )
    if data["ndim"] == 1:
        theory_radii = np.linspace(0.0, radii.max(), 200)
    else:
        theory_radii = np.linspace(max(radii.min(), 1.0), radii.max(), 200)
    theory = theoretical_ratio(data["ndim"], data["beta"], theory_radii)
    axis.plot(
        theory_radii,
        theory,
        color="0.2",
        linestyle="-.",
        linewidth=1.1,
        zorder=2,
    )

    is_log = data["ndim"] == 1
    if is_log:
        axis.set_yscale("log")
        ymin = 0.6 * min(data["best"].min(), data["approx3"].min(), theory.min())
        axis.set_ylim(ymin, 1.45)
        axis.yaxis.set_major_locator(
            LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0), numticks=30)
        )
        axis.yaxis.set_minor_formatter(NullFormatter())
    else:
        ymax = 1.08 * max(data["best"].max(), data["approx3"].max(), theory.max(), 1.0)
        axis.set_ylim(0.0, ymax)
    axis.set_xlim(radii.min() - 0.6, radii.max() + 0.6)
    axis.set_xticks(
        [radius for radius in radii if radius % max(1, round(radii.max() / 5)) == 0]
    )
    axis.set_xlabel(r"$R/a$", labelpad=2)
    axis.set_title(
        rf"$d={data['ndim']}$, $\beta={data['beta']:.2f}$, $L={data['lattice_size']}$",
        pad=4,
    )
    axis.grid(color="0.8", linewidth=0.5, alpha=0.5, which="major")
    axis.set_axisbelow(True)
    axis.tick_params(direction="out", length=3.5, width=0.8, pad=2)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def make_figure(summaries: list[dict[str, object]]) -> plt.Figure:
    """Build the three-panel mask-radius-scan figure.

    Only the $d=1$ panel is log-scale (see ``plot_panel``), so the $y$-axis
    is not shared across panels; a single label centred on the figure
    stands in for the three (identical) axis labels.
    """
    panels = [_panel_data(summary) for summary in summaries]
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(fig_width, 0.32 * fig_width),
        gridspec_kw={"wspace": 0.22},
    )
    for axis, data in zip(axes, panels):
        plot_panel(axis, data)

    figure.supylabel(
        r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$",
        fontsize=9,
        x=0.01,
    )
    figure.subplots_adjust(
        left=0.095, right=0.998, bottom=0.21, top=0.88, wspace=0.22
    )
    return figure


def parse_args() -> argparse.Namespace:
    """Parse which (dim, beta, lattice size) scans to combine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panels",
        nargs=3,
        type=float,
        action="append",
        dest="panels",
        metavar=("DIM", "BETA", "LATTICE_SIZE"),
        help=(
            "a (dimension, beta, lattice size) scan to include as a panel, "
            "left to right; repeat for each panel (default: the three "
            "published scans, d=1,2,3)"
        ),
    )
    args = parser.parse_args()
    if not args.panels:
        args.panels = [list(panel) for panel in DEFAULT_PANELS]
    args.panels = [
        (int(ndim), float(beta), int(lattice_size))
        for ndim, beta, lattice_size in args.panels
    ]
    return args


def main() -> None:
    """Load the three scans, draw the combined figure, and save it."""
    args = parse_args()
    summaries = [load_summary(*panel) for panel in args.panels]

    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    legend_path = FIGURES_DIRECTORY / f"{OUTPUT_STEM}_legend.pdf"
    save_legend(legend_path)
    print(f"Wrote {legend_path}")
    print(f"Wrote {legend_path.with_suffix('.png')}")

    figure = make_figure(summaries)
    pdf_output = FIGURES_DIRECTORY / f"{OUTPUT_STEM}.pdf"
    png_output = FIGURES_DIRECTORY / f"{OUTPUT_STEM}.png"
    figure.savefig(pdf_output)
    figure.savefig(png_output, dpi=240)
    plt.close(figure)
    print(f"Wrote {pdf_output}")
    print(f"Wrote {png_output}")


if __name__ == "__main__":
    main()
