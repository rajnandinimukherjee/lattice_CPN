"""Generate Figure 2 from the saved XY and CP(2) one-point analyses."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lattice-mplconfig")
)

import matplotlib

matplotlib.use("pdf")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from plot_settings import latex_font_params


XY_VARIANCE = (
    ROOT
    / "training_data/xy_scaling_dim2_b0.65_loss-variance_near-zero_training.npy"
)
XY_PENALTY = ROOT / "training_data/xy_scaling_dim2_b0.65_loss-penalty_training.npy"
CP2_VARIANCE = (
    ROOT
    / "training_data/cp2_scaling_dim2_b2.5_L64_i0_j1_action-gauge_"
    "loss-variance_training.npy"
)
CP2_PENALTY = (
    ROOT
    / "training_data/cp2_scaling_dim2_b2.5_L64_i0_j1_action-gauge_"
    "loss-penalty_training.npy"
)
FIGURES = ROOT / "paper" / "figs"
OUTPUTS = {
    "xy_variance": FIGURES / "xy_training_variance.pdf",
    "xy_penalty": FIGURES / "xy_training_penalty.pdf",
    "cp2_variance": FIGURES / "cp2_training_variance.pdf",
    "cp2_penalty": FIGURES / "cp2_training_penalty.pdf",
    "legend": FIGURES / "training_legend.pdf",
}


plt.rcParams.update(latex_font_params)
plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "lines.linewidth": 0.9,
        "pdf.fonttype": 42,
    }
)


def load_artifact(path: Path) -> dict:
    """Read a saved training dictionary."""
    payload = np.load(path, allow_pickle=True)
    if payload.shape != () or not isinstance(payload.item(), dict):
        raise ValueError(f"invalid training artifact: {path}")
    return payload.item()


def plot_history(
    axis: plt.Axes,
    data: dict,
    key: str,
    *,
    error_key: Mapping[str, str] | None = None,
    log: bool = False,
    log_floor: float | None = None,
) -> None:
    """Draw training and validation series with their recorded errors."""
    x = np.arange(len(data[f"train_{key}"]))
    for kind, color in (
        ("train", "tab:blue"),
        ("test", "tab:orange"),
    ):
        y = np.asarray(data[f"{kind}_{key}"], dtype=float)
        finite = np.isfinite(y)
        if log:
            finite &= y > 0
        if error_key is not None:
            errors = np.asarray(data[error_key[kind]], dtype=float)
            band = finite & np.isfinite(errors)
            if log:
                band &= (y - errors) > 0
            axis.fill_between(
                x,
                y - errors,
                y + errors,
                where=band,
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )
        axis.plot(
            x,
            np.where(finite, y, np.nan),
            color=color,
            alpha=0.30 if kind == "train" else 0.95,
            linewidth=0.7 if kind == "train" else 1.0,
            zorder=2 if kind == "train" else 3,
        )
    if log:
        axis.set_yscale("log")
        if log_floor is not None:
            axis.set_ylim(bottom=log_floor)
    axis.set_xlim(0, len(x) - 1)
    axis.set_axisbelow(True)
    axis.tick_params(direction="out", length=2.5, width=0.65, pad=2)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)


def add_reference(axis: plt.Axes, value: float) -> None:
    """Draw the undeformed reference line."""
    axis.axhline(value, color="0.15", linestyle="--", linewidth=0.85, zorder=0)


def add_ratio_bars(
    axis: plt.Axes,
    ratio_alpha_one: float,
    ratio_alpha_star: float,
    *,
    precision: int,
) -> None:
    """Compare the penalty field at unit and optimized scale."""
    values = [ratio_alpha_one, ratio_alpha_star]
    bars = axis.bar(
        [0, 1],
        values,
        width=0.62,
        color=["0.66", "tab:green"],
        edgecolor="0.2",
        linewidth=0.45,
        zorder=2,
    )
    add_reference(axis, 1.0)
    axis.set_ylim(0, 1.10)
    axis.set_yticks([0, 0.5, 1.0])
    axis.set_xticks([0, 1], ["1", r"$\alpha_\star$"])
    axis.set_xlim(-0.55, 1.55)
    axis.yaxis.tick_right()
    axis.yaxis.set_label_position("right")
    axis.set_ylabel(
        r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$",
        rotation=270,
        labelpad=7,
        fontsize=7.5,
    )
    axis.tick_params(
        direction="out",
        length=2.2,
        width=0.6,
        pad=2,
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_linewidth(0.65)
    axis.spines["bottom"].set_linewidth(0.65)
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:.{precision}f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def add_variance_group(
    figure: plt.Figure,
    data: dict,
    *,
    variance_ratio_tick: float | None = None,
) -> None:
    """Add the expectation and normalized variance panels."""
    inner = figure.add_gridspec(
        2,
        1,
        height_ratios=[0.86, 1.0],
        left=0.24,
        right=0.82 if variance_ratio_tick is not None else 0.99,
        bottom=0.22,
        top=0.94,
        hspace=0.045,
    )
    expectation_axis = plt.subplot(inner[0])
    variance_axis = plt.subplot(inner[1], sharex=expectation_axis)
    plot_history(
        expectation_axis,
        data,
        "exp",
        error_key={"train": "train_exp_err", "test": "test_exp_err"},
    )
    target = float(data["target_exp"])
    error = float(data["exp_err"])
    if np.isfinite(error) and error > 0:
        expectation_axis.axhspan(
            target - error,
            target + error,
            color="0.15",
            alpha=0.08,
            linewidth=0,
            zorder=0,
        )
    add_reference(expectation_axis, target)
    expectation_axis.set_ylabel(r"$\langle\mathcal{Q}\rangle$", labelpad=3)
    expectation_axis.tick_params(labelbottom=False)

    initial_variance = float(data["initial_var"])
    normalized = dict(data)
    normalized["train_var"] = np.asarray(data["train_var"]) / initial_variance
    normalized["test_var"] = np.asarray(data["test_var"]) / initial_variance
    plot_history(variance_axis, normalized, "var", log=True, log_floor=0.08)
    add_reference(variance_axis, 1.0)
    variance_axis.set_ylabel(
        r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$",
        labelpad=3,
    )
    variance_axis.set_xlabel("training epoch", labelpad=2)

    if variance_ratio_tick is not None:
        if not np.isfinite(variance_ratio_tick) or variance_ratio_tick <= 0:
            raise ValueError("variance_ratio_tick must be finite and positive")
        ratio_axis = variance_axis.twinx()
        ratio_axis.set_yscale("log")
        ratio_axis.set_ylim(variance_axis.get_ylim())
        ratio_axis.set_yticks(
            [variance_ratio_tick], labels=[f"{variance_ratio_tick:.3f}"]
        )
        ratio_axis.tick_params(
            axis="y",
            direction="out",
            length=2.5,
            width=0.65,
            pad=2,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )
        ratio_axis.tick_params(axis="x", bottom=False, labelbottom=False)
        ratio_axis.spines["left"].set_visible(False)
        ratio_axis.spines["top"].set_visible(False)
        ratio_axis.spines["bottom"].set_visible(False)
        ratio_axis.spines["right"].set_linewidth(0.7)
        variance_axis.spines["right"].set_visible(False)


def add_penalty_group(
    history_axis: plt.Axes,
    ratio_axis: plt.Axes,
    data: dict,
    *,
    alpha_one_ratio: float,
    alpha_star_ratio: float,
    precision: int,
    max_epochs: int,
) -> None:
    """Add penalty minimization and its variance-ratio comparison."""
    plot_history(
        history_axis,
        data,
        "exp",
        error_key={"train": "train_exp_err", "test": "test_exp_err"},
        log=True,
        log_floor=0.8,
    )
    add_reference(history_axis, float(data["target_exp"]))
    history_axis.set_ylabel(
        r"$\langle e^{-2\,\mathrm{Re}\,\Delta S}\rangle$",
        labelpad=3,
    )
    history_axis.set_xlabel("training epoch", labelpad=2)
    history_axis.set_xlim(0, max_epochs)
    # Keep the right spine to close the penalty panel and separate it from
    # the neighboring alpha comparison. The left axis carries the tick labels.
    history_axis.spines["right"].set_visible(True)
    add_ratio_bars(
        ratio_axis,
        alpha_one_ratio,
        alpha_star_ratio,
        precision=precision,
    )


def main() -> None:
    xy_variance = load_artifact(XY_VARIANCE)
    xy_penalty = load_artifact(XY_PENALTY)
    cp2_variance = load_artifact(CP2_VARIANCE)
    cp2_penalty = load_artifact(CP2_PENALTY)

    handles = [
        Line2D([0], [0], color="tab:blue", linewidth=1.0, label="Training"),
        Line2D([0], [0], color="tab:orange", linewidth=1.0, label="Validation"),
        Line2D(
            [0],
            [0],
            color="0.15",
            linestyle="--",
            linewidth=0.85,
            label="Undeformed reference",
        ),
    ]
    FIGURES.mkdir(parents=True, exist_ok=True)

    for name, data in (
        ("xy_variance", xy_variance),
        ("cp2_variance", cp2_variance),
    ):
        figure = plt.figure(figsize=(2.35, 1.78))
        add_variance_group(figure, data)
        figure.savefig(OUTPUTS[name])
        plt.close(figure)

    for name, data, alpha_one_ratio, alpha_star_ratio, precision, max_epochs in (
        ("xy_penalty", xy_penalty, 0.218, float(xy_penalty["variance_ratio"]), 2, 250),
        ("cp2_penalty", cp2_penalty, 0.79253885, 0.46453393, 3, 400),
    ):
        figure = plt.figure(figsize=(3.65, 1.78))
        grid = figure.add_gridspec(
            1,
            2,
            width_ratios=[3.0, 1.0],
            left=0.17,
            right=0.84,
            bottom=0.22,
            top=0.94,
            wspace=0.015,
        )
        history_axis = figure.add_subplot(grid[0, 0])
        ratio_axis = figure.add_subplot(grid[0, 1])
        add_penalty_group(
            history_axis,
            ratio_axis,
            data,
            alpha_one_ratio=alpha_one_ratio,
            alpha_star_ratio=alpha_star_ratio,
            precision=precision,
            max_epochs=max_epochs,
        )
        figure.savefig(OUTPUTS[name])
        plt.close(figure)

    # A separate legend asset lets the four panels share one common key.
    figure = plt.figure(figsize=(4.0, 0.28))
    figure.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.45,
        borderaxespad=0,
        fontsize=9,
    )
    figure.savefig(OUTPUTS["legend"], bbox_inches="tight", pad_inches=0.005)
    plt.close(figure)
    for output in OUTPUTS.values():
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
