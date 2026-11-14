"""Plot every saved 2D radial-profile result for the radial-profile comparison.

This script only reads the already-scaled fields in ``training_data``. It does
not load ensembles, refit profile scales, or rerun an optimization.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "paper" / "figs"
TRAINING = ROOT / "training_data"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_settings import latex_font_params

XY_BETAS = (0.65, 0.70, 0.75, 0.80, 1.20, 1.25, 1.30, 1.35)
CP2_BETAS = (1.5, 2.0, 2.5, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6)

PROFILE_STYLES = {
    "variance": {
        "color": "tab:blue",
        "marker": "o",
        "label": "direct variance",
    },
    "penalty": {
        "color": "tab:orange",
        "marker": "s",
        "label": "Approx. 1 (weak corr.)",
    },
    "approx2": {
        "color": "tab:green",
        "marker": "^",
        "label": "Approx. 2 (small fluc.)",
    },
    "approx3": {
        "color": "tab:red",
        "marker": "D",
        "label": "Approx. 3 (smooth deforms)",
    },
}

XY_FILES = {
    "variance": ("variance", "training"),
    "penalty": ("penalty", "training"),
    "approx2": ("saddle-point", "result"),
    "approx3": ("approx3", "result"),
}
CP2_FILES = {
    "variance": ("variance", "training"),
    "penalty": ("penalty", "training"),
    "approx2": ("approx2", "result"),
    "approx3": ("approx3", "result"),
}

plt.rcParams.update(latex_font_params)
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)


def _load_scaled_field(
    filename: Path,
    *,
    beta: float,
    lattice_shape: tuple[int, int],
    cp_n: int | None = None,
) -> np.ndarray:
    """Load one validated, already-scaled XY or CP(2) result."""
    if not filename.exists():
        raise FileNotFoundError(f"missing saved profile: {filename}")
    loaded = np.load(filename, allow_pickle=True)
    if loaded.shape != () or not isinstance(loaded.item(), dict):
        raise ValueError(f"expected a saved profile mapping in {filename}")
    data = loaded.item()
    metadata = data.get("metadata", {})
    if int(metadata.get("ndim", -1)) != 2:
        raise ValueError(f"{filename} is not a two-dimensional profile")
    if tuple(metadata.get("lattice_size", ())) != lattice_shape:
        raise ValueError(
            f"{filename} uses lattice {metadata.get('lattice_size')}, "
            f"expected {lattice_shape}"
        )
    if not np.isclose(float(metadata.get("beta", np.nan)), beta):
        raise ValueError(f"{filename} has a different beta value")
    if cp_n is not None and int(metadata.get("cp_n", -1)) != cp_n:
        raise ValueError(f"{filename} is not a CP({cp_n}) profile")

    if "scaled_field" in data:
        field = np.asarray(data["scaled_field"], dtype=float)
    else:
        field = float(data.get("alpha_scale", 1.0)) * np.asarray(
            data["learned_field"], dtype=float
        )
    if cp_n is None and field.shape == (*lattice_shape, 1):
        field = field[..., 0]
    expected_shape = (*lattice_shape,) if cp_n is None else (*lattice_shape, cp_n + 1)
    if field.shape != expected_shape or not np.all(np.isfinite(field)):
        raise ValueError(f"{filename} contains an invalid deformation field")
    return field


def _radial_average(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average a scalar field over periodic Euclidean distance shells."""
    coordinates = np.indices(field.shape, dtype=np.int64)
    lengths = np.asarray(field.shape, dtype=np.int64).reshape(
        -1, *([1] * field.ndim)
    )
    displacement = np.minimum(coordinates, lengths - coordinates)
    radius_squared = np.square(displacement).sum(axis=0)
    shells = np.unique(radius_squared)
    radii = np.sqrt(shells.astype(float))
    values = np.asarray(
        [field[radius_squared == shell].mean() for shell in shells], dtype=float
    )
    return radii, values


def _xy_filename(beta: float, method: str) -> Path:
    loss, suffix = XY_FILES[method]
    return TRAINING / f"xy_scaling_dim2_b{beta:.2f}_loss-{loss}_{suffix}.npy"


def _cp2_filename(beta: float, method: str) -> Path:
    loss, suffix = CP2_FILES[method]
    beta_token = f"{beta:g}"
    return TRAINING / (
        f"cp2_scaling_dim2_b{beta_token}_L64_i0_j1_action-gauge_"
        f"loss-{loss}_{suffix}.npy"
    )


def _load_xy_profiles() -> dict[float, dict[str, np.ndarray]]:
    profiles: dict[float, dict[str, np.ndarray]] = {}
    for beta in XY_BETAS:
        profiles[beta] = {
            method: _load_scaled_field(
                _xy_filename(beta, method),
                beta=beta,
                lattice_shape=(32, 32),
            )
            for method in XY_FILES
        }
    return profiles


def _load_cp2_profiles() -> dict[float, dict[str, np.ndarray]]:
    profiles: dict[float, dict[str, np.ndarray]] = {}
    direction = np.asarray([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    for beta in CP2_BETAS:
        profiles[beta] = {
            method: np.tensordot(
                _load_scaled_field(
                    _cp2_filename(beta, method),
                    beta=beta,
                    lattice_shape=(64, 64),
                    cp_n=2,
                ),
                direction,
                axes=([-1], [0]),
            )
            for method in CP2_FILES
        }
    return profiles


def _plot_panel(
    profiles: dict[float, dict[str, np.ndarray]],
    output: Path,
    *,
    model_name: str,
) -> None:
    """Draw all beta curves with one fixed radial and vertical scale."""
    beta_values = sorted(profiles)
    radial_data: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    visible_values: list[np.ndarray] = []
    for beta in beta_values:
        radial_data[beta] = {}
        for method, field in profiles[beta].items():
            radii, values = _radial_average(field)
            radial_data[beta][method] = (radii, values)
            visible_values.append(values[radii <= 8.0])

    all_values = np.concatenate(visible_values)
    ymin = min(0.0, float(all_values.min()))
    ymax = max(0.0, float(all_values.max()))
    span = max(ymax - ymin, 1e-8)
    y_limits = (ymin - 0.05 * span, ymax + 0.06 * span)

    fig, axis = plt.subplots(figsize=(2.92, 3.78))
    for beta in beta_values:
        for method, style in PROFILE_STYLES.items():
            radii, values = radial_data[beta][method]
            visible = radii <= 8.0
            visible_radii = radii[visible]
            visible_values = values[visible]
            axis.plot(
                visible_radii,
                visible_values,
                color=style["color"],
                marker=style["marker"],
                markersize=2.4,
                markevery=3,
                linewidth=0.9,
                alpha=0.58,
                label=None,
                zorder=2,
            )

    axis.set_xlim(0.0, 8.0)
    axis.set_ylim(*y_limits)
    axis.set_xticks(np.arange(0.0, 8.1, 2.0))
    axis.set_xlabel(r"$r=|x|$", labelpad=2)
    axis.set_ylabel(r"$F_r$", labelpad=2)
    axis.tick_params(axis="both", labelsize=7, length=2.5, pad=2)
    axis.set_axisbelow(True)
    axis.grid(color="0.8", linewidth=0.5, alpha=0.55)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
    fig.tight_layout(pad=0.6)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {model_name} panel with {len(beta_values)} beta values to {output}")


def main() -> None:
    """Load saved scaled profiles and write the two Figure 5 panels."""
    xy_profiles = _load_xy_profiles()
    cp2_profiles = _load_cp2_profiles()
    _plot_panel(
        xy_profiles,
        FIGS / "xy_radial_profiles.pdf",
        model_name="XY",
    )
    _plot_panel(
        cp2_profiles,
        FIGS / "cp2_radial_profiles.pdf",
        model_name="CP(2)",
    )


if __name__ == "__main__":
    main()
