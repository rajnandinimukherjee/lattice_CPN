"""Update the CP(2) training diagnostics from isolated near-zero runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plot_training_diagnostics import (
    add_penalty_group,
    add_variance_group,
    load_artifact,
    plt,
)


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "paper" / "figs"
VARIANCE = ROOT / (
    "training_data/cp2_scaling_dim2_b2.5_L64_i0_j1_action-gauge_"
    "loss-variance_near-zero_training.npy"
)
PENALTY = ROOT / (
    "training_data/cp2_scaling_dim2_b2.5_L64_i0_j1_action-gauge_"
    "loss-penalty_near-zero_training.npy"
)


def connect_validation_checkpoints(
    axis: plt.Axes,
    data: dict,
    key: str,
    *,
    scale: float = 1.0,
) -> None:
    """Connect recorded validation checkpoints across unscored epochs."""
    x = np.arange(len(data[f"test_{key}"]))
    values = np.asarray(data[f"test_{key}"], dtype=float) / scale
    finite = np.isfinite(values)
    if not finite.any():
        return
    error_key = f"test_{key}_err"
    if error_key in data:
        errors = np.asarray(data[error_key], dtype=float) / scale
        band = finite & np.isfinite(errors)
        if band.any():
            axis.fill_between(
                x[band],
                values[band] - errors[band],
                values[band] + errors[band],
                color="tab:orange",
                alpha=0.12,
                linewidth=0,
                zorder=2,
            )
    axis.plot(
        x[finite],
        values[finite],
        color="tab:orange",
        alpha=0.95,
        linewidth=1.0,
        zorder=4,
    )


def main() -> None:
    variance_data = load_artifact(VARIANCE)
    penalty_data = load_artifact(PENALTY)
    FIGURES.mkdir(parents=True, exist_ok=True)

    variance_figure = plt.figure(figsize=(2.35, 1.78))
    add_variance_group(variance_figure, variance_data)
    connect_validation_checkpoints(variance_figure.axes[0], variance_data, "exp")
    connect_validation_checkpoints(
        variance_figure.axes[1],
        variance_data,
        "var",
        scale=float(variance_data["initial_var"]),
    )
    variance_output = FIGURES / "cp2_training_variance.pdf"
    variance_figure.savefig(variance_output)
    plt.close(variance_figure)

    penalty_figure = plt.figure(figsize=(3.65, 1.78))
    grid = penalty_figure.add_gridspec(
        1,
        2,
        width_ratios=[3.0, 1.0],
        left=0.17,
        right=0.84,
        bottom=0.22,
        top=0.94,
        wspace=0.015,
    )
    history_axis = penalty_figure.add_subplot(grid[0, 0])
    ratio_axis = penalty_figure.add_subplot(grid[0, 1])
    alpha_star_ratio = float(
        penalty_data.get(
            "alpha_star_full_ensemble_variance_ratio",
            penalty_data["variance_ratio"],
        )
    )
    add_penalty_group(
        history_axis,
        ratio_axis,
        penalty_data,
        alpha_one_ratio=float(penalty_data["alpha_one_variance_ratio"]),
        alpha_star_ratio=alpha_star_ratio,
        precision=3,
        max_epochs=max(1, len(penalty_data["train_exp"]) - 1),
    )
    connect_validation_checkpoints(history_axis, penalty_data, "exp")
    penalty_output = FIGURES / "cp2_training_penalty.pdf"
    penalty_figure.savefig(penalty_output)
    plt.close(penalty_figure)

    print(f"Wrote {variance_output}")
    print(f"Wrote {penalty_output}")
    print(f"alpha=1 variance ratio: {penalty_data['alpha_one_variance_ratio']:.8f}")
    print(f"alpha*= variance ratio: {alpha_star_ratio:.8f}")


if __name__ == "__main__":
    main()
