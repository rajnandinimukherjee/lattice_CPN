"""Plot the XY two-point variance, SNR gain, and shared figure legend."""

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
    / "training_data/xy_two_point/"
    "xy_twopt_dim2_b0.65_L32_summary.npy"
)
FIGURES = ROOT / "paper" / "figs"
PDF_OUTPUT = FIGURES / "xy_two_point_variance.pdf"
PNG_OUTPUT = FIGURES / "xy_two_point_variance.png"
LEGEND_PDF_OUTPUT = FIGURES / "xy_two_point_stn_legend.pdf"
LEGEND_PNG_OUTPUT = FIGURES / "xy_two_point_stn_legend.png"

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
        "model": "XY",
        "ndim": 2,
        "beta": 0.65,
        "lattice_size": 32,
        "source": (0, 0),
        "observable": "exp(i*(theta_source-theta_sink))",
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


def make_plot(data: dict) -> plt.Figure:
    separations = np.asarray(data["time_separations"], dtype=int)
    if not np.array_equal(separations, np.arange(1, 17)):
        raise ValueError(f"unexpected time separations: {separations}")
    variances = {
        method: np.asarray(data[f"{method}_variance"], dtype=float)
        for method in ("undeformed", "glued", "deformed")
    }
    if any(values.shape != separations.shape or np.any(values <= 0)
           or not np.isfinite(values).all() for values in variances.values()):
        raise ValueError("invalid two-point variance series")
    profile_path = ROOT / data["metadata"]["one_point_profile"]
    one_point = np.load(profile_path, allow_pickle=True).item()
    one_point_ratio = float(one_point["variance_ratio"])
    if not np.isfinite(one_point_ratio) or one_point_ratio <= 0:
        raise ValueError(f"invalid one-point variance ratio in {profile_path}")
    gain = np.sqrt(variances["undeformed"] / variances["glued"])

    figure, (axis, gain_axis) = plt.subplots(
        2,
        1,
        figsize=(0.50 * fig_width, 0.40 * fig_width),
        sharex=True,
        gridspec_kw={"height_ratios": [2.8, 1.2], "hspace": 0.12},
    )
    for method, color, marker in (
        ("undeformed", "tab:blue", "o"),
        ("glued", "tab:orange", "s"),
        ("deformed", "tab:green", "^"),
    ):
        axis.plot(separations, variances[method], color=color, marker=marker,
                  markerfacecolor="none" if method != "undeformed" else color,
                  markersize=3.8, linewidth=0.9)
    axis.axhline(one_point_ratio**2, color="tab:red", linestyle="--",
                 linewidth=1.0)

    all_values = np.concatenate([*variances.values(), [one_point_ratio**2]])
    min_exponent = int(np.floor(np.log10(all_values.min())))
    max_exponent = int(np.ceil(np.log10(all_values.max())))
    if min_exponent == max_exponent:
        max_exponent += 1
    exponents = np.arange(min_exponent, max_exponent + 1)
    y_ticks = 10.0**exponents
    axis.set_yscale("log")
    axis.set_xlim(0.5, 16.5)
    axis.set_ylim(y_ticks[0], y_ticks[-1] * 1.2)
    axis.set_xticks(separations)
    axis.set_yticks(y_ticks, [rf"$10^{{{exponent}}}$" for exponent in exponents])
    axis.set_ylabel(r"$\operatorname{Var}[O_t O_0^\dagger]$", labelpad=4)
    axis.grid(color="0.8", linewidth=0.5, alpha=0.5)
    axis.set_axisbelow(True)
    axis.tick_params(direction="out", length=3.5, width=0.8, pad=2)

    gain_axis.plot(
        separations,
        gain,
        color="tab:orange",
        marker="D",
        markersize=3.8,
        linestyle="-",
        linewidth=0.9,
        zorder=3,
    )
    gain_axis.axhline(1.0, color="0.35", linestyle=":", linewidth=0.8, zorder=1)
    gain_axis.set_xlim(0.5, 16.5)
    gain_upper = float(np.ceil(gain.max() * 2) / 2 + 0.5)
    gain_axis.set_ylim(0.8, gain_upper)
    gain_tick_step = 0.5 if gain_upper <= 3 else 2.0
    gain_tick_margin = 0.5 if gain_upper <= 3 else 1.0
    gain_ticks = np.arange(
        1.0, gain_upper - gain_tick_margin + 1e-9, gain_tick_step
    )
    gain_axis.set_yticks(gain_ticks, [f"{tick:g}" for tick in gain_ticks])
    gain_axis.set_xticks(separations)
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
            ("tab:blue", "o", "Undeformed", 3.2, "tab:blue", 0.8),
            ("tab:orange", "s", "Glued one-point profile", 5.0, "none", 1.0),
            ("tab:green", "^", "Trained per separation", 5.0, "none", 1.0),
        )
    ]
    handles.append(Line2D([], [], color="tab:red", linestyle="--",
                          linewidth=1.0, label="Squared one-point variance ratio"))
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
    legend.savefig(
        LEGEND_PNG_OUTPUT, dpi=220, bbox_inches="tight", pad_inches=0.02
    )
    plt.close(legend)
    print(f"Wrote {PDF_OUTPUT}")
    print(f"Wrote {PNG_OUTPUT}")
    print(f"Wrote {LEGEND_PDF_OUTPUT}")
    print(f"Wrote {LEGEND_PNG_OUTPUT}")


if __name__ == "__main__":
    main()
