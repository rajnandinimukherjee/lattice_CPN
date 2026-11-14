"""Plot the CP(2) two-point signal-to-noise data for Figure 3."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

cache_dir = Path(tempfile.gettempdir()) / "lattice-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lattice-mplconfig")
)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib

matplotlib.use("pdf")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from plot_settings import fig_width, latex_font_params

SUMMARY = (
    ROOT
    / "training_data/cpn_two_point/"
    "cp2_twopt_dim2_b2.5_L64_i0_j1_summary.npy"
)
FIGURES = ROOT / "paper" / "figs"
PDF_OUTPUT = FIGURES / "cp2_two_point_stn.pdf"
PNG_OUTPUT = FIGURES / "cp2_two_point_stn.png"
LEGEND_PDF_OUTPUT = FIGURES / "cp2_two_point_stn_legend.pdf"
LEGEND_PNG_OUTPUT = FIGURES / "cp2_two_point_stn_legend.png"

plt.rcParams.update(latex_font_params)
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def load_summary() -> dict:
    payload = np.load(SUMMARY, allow_pickle=True)
    if payload.shape != () or not isinstance(payload.item(), dict):
        raise ValueError(f"invalid two-point summary: {SUMMARY}")
    data = payload.item()
    metadata = data.get("metadata", {})
    expected = {
        "cp_n": 2,
        "ndim": 2,
        "beta": 2.5,
        "lattice_size": 64,
        "i": 0,
        "j": 1,
        "observable": "P_ij(0) * P_ji((0,t))",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"summary metadata {key}={metadata.get(key)!r}; expected {value!r}"
            )
    for key in ("undeformed_snr", "glued_snr", "deformed_snr"):
        values = np.asarray(data[key], dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"summary contains invalid values for {key}")
    return data


def standard_error_series(data: dict) -> dict[str, np.ndarray]:
    n_cfgs = np.asarray(data["n_cfgs"], dtype=float)
    if not np.isfinite(n_cfgs).all() or np.any(n_cfgs <= 0):
        raise ValueError("summary has a non-positive configuration count")
    series = {}
    for method in ("undeformed", "glued", "deformed"):
        variance = np.asarray(data[f"{method}_variance"], dtype=float)
        if not np.isfinite(variance).all() or np.any(variance <= 0):
            raise ValueError(f"summary contains invalid {method} variances")
        series[method] = np.sqrt(variance / n_cfgs)
    return series


def fixed_signal_snr_series(data: dict) -> dict[str, np.ndarray]:
    signal = np.abs(np.asarray(data["undeformed_mean"], dtype=np.complex128))
    if not np.isfinite(signal).all() or np.any(signal <= 0):
        raise ValueError("summary contains invalid undeformed signal values")
    standard_errors = standard_error_series(data)
    return {
        method: signal / standard_error
        for method, standard_error in standard_errors.items()
    }


def make_plot(data: dict) -> plt.Figure:
    separations = np.asarray(data["time_separations"], dtype=int)
    if not np.array_equal(separations, np.arange(1, 33)):
        raise ValueError(f"unexpected time separations: {separations}")
    signal = np.abs(np.asarray(data["undeformed_mean"], dtype=np.complex128))
    standard_errors = standard_error_series(data)
    snr = fixed_signal_snr_series(data)
    gain = snr["glued"] / snr["undeformed"]

    figure, (axis, gain_axis) = plt.subplots(
        2,
        1,
        figsize=(0.50 * fig_width, 0.40 * fig_width),
        sharex=True,
        gridspec_kw={"height_ratios": [2.8, 1.2], "hspace": 0.12},
    )
    axis.errorbar(
        separations,
        signal,
        yerr=standard_errors["undeformed"],
        fmt="o-",
        color="black",
        ecolor="black",
        markersize=2.6,
        linewidth=0.8,
        elinewidth=0.7,
        capsize=2.0,
        capthick=0.7,
        zorder=6,
    )
    series = (
        {
            "values": standard_errors["undeformed"],
            "label": "Undeformed SE",
            "color": "tab:blue",
            "marker": "o",
            "linestyle": "-",
            "markersize": 4.0,
            "zorder": 3,
        },
        {
            "values": standard_errors["glued"],
            "label": "Glued one-point profile SE",
            "color": "tab:orange",
            "marker": "s",
            "linestyle": "-",
            "markersize": 6.5,
            "markerfacecolor": "none",
            "markeredgewidth": 1.2,
            "zorder": 4,
        },
        {
            "values": standard_errors["deformed"],
            "label": "Trained per separation SE",
            "color": "tab:green",
            "marker": "^",
            "linestyle": "-",
            "markersize": 6.5,
            "markerfacecolor": "none",
            "markeredgewidth": 1.2,
            "zorder": 5,
        },
    )
    for options in series:
        options = dict(options)
        values = options.pop("values")
        axis.plot(separations, values, **options)

    all_values = np.concatenate([signal, *standard_errors.values()])
    min_exponent = int(np.floor(np.log10(all_values.min())))
    max_exponent = int(np.ceil(np.log10(all_values.max())))
    if min_exponent == max_exponent:
        max_exponent += 1
    exponents = np.arange(min_exponent, max_exponent + 1)
    y_ticks = 10.0**exponents
    axis.set_yscale("log")
    axis.set_xlim(0.5, 32.5)
    axis.set_ylim(y_ticks[0], y_ticks[-1])
    axis.set_xticks(np.arange(2, 33, 2))
    axis.set_yticks(y_ticks, [rf"$10^{{{exponent}}}$" for exponent in exponents])
    axis.set_ylabel(r"$\mathrm{SE}(t),\,|C(t)|$", labelpad=4)
    axis.grid(color="0.8", linewidth=0.5, alpha=0.5)
    axis.set_axisbelow(True)
    axis.tick_params(direction="out", length=3.5, width=0.8, pad=2)

    gain_axis.plot(
        separations,
        gain,
        color="tab:purple",
        marker="D",
        markersize=3.8,
        linestyle="--",
        linewidth=0.9,
        zorder=3,
    )
    gain_axis.axhline(1.0, color="0.35", linestyle=":", linewidth=0.8, zorder=1)
    gain_axis.set_xlim(0.5, 32.5)
    gain_upper = float(np.ceil(gain.max() * 2) / 2)
    gain_axis.set_ylim(0.8, gain_upper)
    gain_tick_step = 0.5 if gain_upper <= 3 else 2.0
    gain_tick_margin = 0.5 if gain_upper <= 3 else 1.0
    gain_ticks = np.arange(
        1.0, gain_upper - gain_tick_margin + 1e-9, gain_tick_step
    )
    gain_axis.set_yticks(gain_ticks, [f"{tick:g}" for tick in gain_ticks])
    gain_axis.set_xticks(np.arange(2, 33, 2))
    gain_axis.set_xlabel(r"$t/a$")
    gain_axis.set_ylabel("SNR gain", labelpad=4)
    gain_axis.grid(color="0.8", linewidth=0.5, alpha=0.5)
    gain_axis.set_axisbelow(True)
    gain_axis.tick_params(direction="out", length=3.5, width=0.8, pad=2)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
    for spine in gain_axis.spines.values():
        spine.set_linewidth(0.8)
    figure.subplots_adjust(left=0.18, right=0.99, bottom=0.22, top=0.99, hspace=0.12)
    return figure


def make_legend() -> plt.Figure:
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            markersize=markersize,
            markerfacecolor=markerfacecolor,
            markeredgewidth=markeredgewidth,
            linewidth=1.0,
            label=label,
        )
        for color, marker, label, markersize, markerfacecolor, markeredgewidth in (
            ("tab:blue", "o", "Undeformed SE", 3.2, "tab:blue", 0.8),
            ("tab:orange", "s", "Glued one-point profile SE", 6.0, "none", 1.2),
            ("tab:green", "^", "Trained per separation SE", 6.0, "none", 1.2),
        )
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="black",
            marker="o",
            markersize=2.6,
            linewidth=0.8,
            label=r"Signal $|C(t)|$ (error: undeformed SE)",
        )
    )
    figure = plt.figure(figsize=(fig_width, 0.32))
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="center",
        ncols=4,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    return figure


def main() -> None:
    data = load_summary()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure = make_plot(data)
    figure.savefig(PDF_OUTPUT)
    figure.savefig(PNG_OUTPUT, dpi=240)
    plt.close(figure)
    legend = make_legend()
    legend.savefig(LEGEND_PDF_OUTPUT, bbox_inches="tight", pad_inches=0.02)
    legend.savefig(LEGEND_PNG_OUTPUT, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(legend)
    print(f"Wrote {PDF_OUTPUT}")
    print(f"Wrote {PNG_OUTPUT}")
    print(f"Wrote {LEGEND_PDF_OUTPUT}")
    print(f"Wrote {LEGEND_PNG_OUTPUT}")


if __name__ == "__main__":
    main()
