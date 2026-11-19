"""Regenerate the Figure 3 one-axis slices of normalized XY and CP(2) profiles."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_settings import latex_font_params


ROOT = Path(__file__).resolve().parent
TRAINING = ROOT / "training_data"
FIGS = ROOT / "paper" / "figs"
STYLES = (
    ("variance", "tab:blue", "o"),
    ("penalty", "tab:orange", "s"),
    ("approx2", "tab:green", "^"),
    ("approx3", "tab:red", "D"),
)

plt.rcParams.update(latex_font_params)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 7,
                     "ytick.labelsize": 7})


def _filename(model: str, method: str) -> Path:
    if model == "xy":
        loss = {"approx2": "saddle-point"}.get(method, method)
        suffix = "result" if method in ("approx2", "approx3") else "training"
        return TRAINING / f"xy_scaling_dim3_b0.40_loss-{loss}_{suffix}.npy"
    suffix = "result" if method in ("approx2", "approx3") else "training"
    return TRAINING / (
        f"cp2_scaling_dim3_b1.8_L16_i0_j1_action-gauge_"
        f"loss-{method}_{suffix}.npy"
    )


def _load(model: str, method: str) -> tuple[np.ndarray, float]:
    filename = _filename(model, method)
    data = np.load(filename, allow_pickle=True).item()
    metadata = data["metadata"]
    if (int(metadata["ndim"]) != 3
            or tuple(metadata["lattice_size"]) != (16, 16, 16)
            or not np.isclose(float(metadata["beta"]), 0.4 if model == "xy" else 1.8)):
        raise ValueError(f"incompatible Figure 3 data: {filename}")

    # Direct variance training optimizes the full field, so its saved source
    # value is the fitted amplitude. The other methods save unit-source f_x.
    field = np.asarray(data.get("f_x", data["learned_field"]), dtype=float)
    expected = (16, 16, 16, 1 if model == "xy" else 3)
    if field.shape != expected or not np.all(np.isfinite(field)):
        raise ValueError(f"invalid profile in {filename}")
    if model == "xy":
        scalar = field[..., 0]
    else:
        scalar = (field[..., 0] - field[..., 1]) / np.sqrt(2.0)
    source_value = float(scalar[0, 0, 0])
    if not np.isfinite(source_value) or source_value <= 0.0:
        raise ValueError(f"invalid source normalization in {filename}")
    profile = scalar[:, 0, 0] / source_value
    ratio = float(data["variance_ratio"])
    if not np.isfinite(ratio) or ratio < 0.0:
        raise ValueError(f"invalid variance ratio in {filename}")
    return profile, ratio


def _plot(model: str, output: Path) -> None:
    curves = {method: _load(model, method) for method, _, _ in STYLES}
    fig, axis = plt.subplots(figsize=(3.25, 1.85))
    sites = np.arange(8)
    for method, color, marker in STYLES:
        profile, _ = curves[method]
        axis.plot(sites, profile[:8], color=color, marker=marker,
                  markersize=3.2, linewidth=1.0)

    axis.set_xlim(0, 7.25)
    axis.set_ylim(bottom=-0.04)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$f_x$" if model == "xy" else
                    r"$(f_{x,0}-f_{x,1})/\sqrt{2}$")
    axis.grid(color="0.8", linewidth=0.5, alpha=0.5)
    axis.set_axisbelow(True)

    inset = axis.inset_axes([0.64, 0.53, 0.34, 0.43])
    inset.set_facecolor("white")
    inset.bar(range(4), [curves[method][1] for method, _, _ in STYLES],
              color=[color for _, color, _ in STYLES], width=0.68)
    inset.set_ylim(0, 1.12)
    inset.set_xticks([])
    inset.set_yticks([0, 1])
    inset.set_ylabel(r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$",
                     fontsize=7, labelpad=1)
    inset.axhline(1, color="0.25", linestyle="--", linewidth=0.65)
    inset.tick_params(axis="y", labelsize=6, length=2, pad=1)
    fig.tight_layout(pad=0.5)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    _plot("xy", FIGS / "xy_radial_profiles.pdf")
    _plot("cp2", FIGS / "cp2_radial_profiles.pdf")


if __name__ == "__main__":
    main()
