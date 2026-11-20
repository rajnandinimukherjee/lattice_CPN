"""Plot the combined validation history for the paper's field comparison."""

from pathlib import Path

import numpy as np

from plot_training_diagnostics import load_artifact, plt
from plot_settings import OBJECTIVE_COLORS, UNDEFORMED_COLOR


ROOT = Path(__file__).resolve().parent
TRAINING = {
    "Field-independent": ROOT / "training_data" / (
        "xy_scaling_dim2_b0.65_loss-variance_near-zero_seed-1407_training.npy"
    ),
    "Field-dependent": ROOT / "training_data" / (
        "xy_scaling_dim2_b0.65_loss-field-dependent_training.npy"
    ),
}
OUTPUT = ROOT / "paper" / "figs" / "xy_field_comparison.pdf"
DARK_VARIANCE_BLUE = "#00456B"


def main() -> None:
    figure, axis = plt.subplots(figsize=(3.5, 1.9))
    for label, path, color in (
        ("Field-independent", TRAINING["Field-independent"], OBJECTIVE_COLORS["variance"]),
        ("Field-dependent", TRAINING["Field-dependent"], DARK_VARIANCE_BLUE),
    ):
        data = load_artifact(path)
        epochs = np.arange(len(data["test_var"]))
        variance = np.asarray(data["test_var"], dtype=float) / float(
            data["initial_var"]
        )
        valid = (epochs > 0) & np.isfinite(variance) & (variance > 0)
        axis.plot(
            epochs[valid], variance[valid],
            color=color, linestyle="-",
            label=label, lw=1.25,
        )

    axis.axhline(1, color=UNDEFORMED_COLOR, linestyle="--", linewidth=0.8, zorder=0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(50, 4000)
    axis.set_ylim(0.1, 1.05)
    axis.set_xticks([50, 100, 200, 500, 1000, 2000, 4000])
    axis.set_xticklabels(["50", "100", "200", "500", "1000", "2000", "4000"])
    axis.set_yticks([0.1, 0.2, 0.5, 1.0])
    axis.set_yticklabels(["0.1", "0.2", "0.5", "1"])
    axis.minorticks_off()
    axis.set_xlabel("training epoch")
    axis.set_ylabel(
        r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$"
    )
    axis.legend(loc="upper right", frameon=False, handlelength=2.4)
    axis.grid(axis="y", color="0.88", linewidth=0.5)
    axis.tick_params(direction="out", length=2.5, width=0.65)
    figure.subplots_adjust(left=0.20, right=0.94, bottom=0.25, top=0.95)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT)
    plt.close(figure)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
