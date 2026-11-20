"""Plot the left panel of the XY two-point figure in the paper draft."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lattice-mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "lattice-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_settings import TWO_POINT_COLORS, UNDEFORMED_COLOR, latex_font_params


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "training_data/xy_two_point/xy_twopt_dim2_b0.65_L32_summary.npy"
OUTPUT = ROOT / "paper" / "figs" / "xy_two_point_signal"
plt.rcParams.update(latex_font_params)
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "pdf.fonttype": 42})


def load_summary() -> dict:
    data = np.load(SUMMARY, allow_pickle=True).item()
    if not isinstance(data, dict) or data.get("metadata", {}).get("configuration_count") != 10688:
        raise ValueError(f"unexpected XY two-point summary: {SUMMARY}")
    return data


def first_floor_crossing(t: np.ndarray, lower_edge: np.ndarray, floor: float) -> float:
    """Interpolate the first lower-error-bar crossing on the log y-axis."""
    crossing = np.flatnonzero(
        (lower_edge[:-1] > floor) & (lower_edge[1:] <= floor)
    )
    if crossing.size == 0:
        raise ValueError("the lower error-bar edge does not cross its noise floor")
    index = int(crossing[0])
    if lower_edge[index + 1] <= 0:
        raise ValueError("cannot interpolate a nonpositive lower edge on a log axis")
    fraction = (
        (np.log(floor) - np.log(lower_edge[index]))
        / (np.log(lower_edge[index + 1]) - np.log(lower_edge[index]))
    )
    return float(t[index] + fraction * (t[index + 1] - t[index]))


def make_plot(data: dict) -> plt.Figure:
    t = np.asarray(data["time_separations"], dtype=int)
    if not np.array_equal(t, np.arange(1, 17)):
        raise ValueError(f"unexpected time separations: {t}")

    n_cfgs = np.asarray(data["n_cfgs"], dtype=float)
    if not np.isfinite(n_cfgs).all() or np.any(n_cfgs <= 0):
        raise ValueError("invalid configuration counts")
    n_samples = int(data["metadata"]["configuration_count"])
    if not np.all(n_cfgs == n_samples):
        raise ValueError("constant noise floors require the same sample count at every separation")
    se = {}
    for method in ("undeformed", "deformed"):
        variance = np.asarray(data[f"{method}_variance"], dtype=float)
        if variance.shape != t.shape or not np.isfinite(variance).all() or np.any(variance <= 0):
            raise ValueError(f"invalid {method} variance")
        se[method] = np.sqrt(variance / n_cfgs)
    fig, ax = plt.subplots(figsize=(3.9, 2.5))
    lower_limit = 1e-4

    profile_path = ROOT / data["metadata"]["one_point_profile"]
    one_point_ratio = float(np.load(profile_path, allow_pickle=True).item()["variance_ratio"])
    if not np.isfinite(one_point_ratio) or one_point_ratio <= 0:
        raise ValueError(f"invalid one-point variance ratio in {profile_path}")
    floors = (
        ("undeformed", 1.0 / np.sqrt(n_samples), UNDEFORMED_COLOR),
        ("deformed", np.sqrt(one_point_ratio**2 / n_samples), TWO_POINT_COLORS["deformed"]),
    )
    for method, floor, color in floors:
        signal = np.abs(np.asarray(data[f"{method}_mean"], dtype=np.complex128))
        start = first_floor_crossing(t, signal - se[method], floor)
        ax.fill_between(
            [start, 16.5], lower_limit, floor,
            color=color, alpha=0.16, linewidth=0, zorder=1,
        )
        ax.hlines(floor, start, 16.5, color=color, alpha=0.8, linewidth=0.8, zorder=2)

    for method, color, marker, facecolor in (
        ("undeformed", UNDEFORMED_COLOR, "o", UNDEFORMED_COLOR),
        ("deformed", TWO_POINT_COLORS["deformed"], "s", "none"),
    ):
        signal = np.abs(np.asarray(data[f"{method}_mean"], dtype=np.complex128))
        ax.errorbar(
            t, signal, yerr=se[method], fmt=marker + "-", color=color,
            markerfacecolor=facecolor, markersize=4.5, linewidth=1,
            elinewidth=0.8, capsize=2.8, capthick=0.8, zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(lower_limit, 1)
    ax.set_xticks(t)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\langle O_t O_0^\dagger\rangle$")
    ax.grid(color="0.8", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.5, width=0.8, pad=2)
    fig.tight_layout(pad=0.7)
    return fig


def main() -> None:
    figure = make_plot(load_summary())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"))
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
