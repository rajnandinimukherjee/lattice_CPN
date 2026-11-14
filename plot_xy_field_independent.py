"""Plot the Figure 6 field-independent XY training history."""

from pathlib import Path

from plot_training_diagnostics import add_variance_group, load_artifact, plt


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "paper" / "figs"
TRAINING = (
    ROOT
    / "training_data/xy_scaling_dim2_b0.65_loss-variance_near-zero_seed-1407_training.npy"
)
OUTPUT = FIGURES / "xy_field_independent.pdf"


def main() -> None:
    data = load_artifact(TRAINING)
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(2.35, 1.78))
    add_variance_group(
        figure,
        data,
        variance_ratio_tick=float(data["variance_ratio"]),
    )
    figure.savefig(OUTPUT)
    plt.close(figure)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
