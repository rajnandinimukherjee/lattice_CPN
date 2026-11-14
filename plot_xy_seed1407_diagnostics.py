"""Update the XY variance panels from the separate seed-1407 run."""

from __future__ import annotations

from pathlib import Path

from plot_training_diagnostics import add_variance_group, load_artifact, plt


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "paper" / "figs"
TRAINING = ROOT / (
    "training_data/xy_scaling_dim2_b0.65_loss-variance_"
    "near-zero_seed-1407_training.npy"
)


def main() -> None:
    data = load_artifact(TRAINING)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for output_name, show_final_ratio in (
        ("xy_training_variance.pdf", False),
        ("xy_field_independent.pdf", True),
    ):
        figure = plt.figure(figsize=(2.35, 1.78))
        add_variance_group(
            figure,
            data,
            variance_ratio_tick=(
                float(data["variance_ratio"]) if show_final_ratio else None
            ),
        )
        output = FIGURES / output_name
        figure.savefig(output)
        plt.close(figure)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
