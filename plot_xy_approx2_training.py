"""Figure 7: both stages of the XY Approximation 2 training procedure.

The penalty training artifact stores every normalized profile f_x.  Evaluate
each profile on the same thinned ensemble used by the original analysis, then
replay the one-dimensional variance optimization for its selected profile.
Only bonds touching the support of f_x need to enter the action difference.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lattice-mplconfig")
)

import matplotlib

matplotlib.use("pdf")

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
from scipy.optimize import minimize

import XY_utils as xy
from plot_settings import latex_font_params


ROOT = Path(__file__).resolve().parent
TRAINING = ROOT / "training_data/xy_scaling_dim2_b0.65_loss-penalty_training.npy"
ENSEMBLE = ROOT / "data/xy_dim2_b0.65_L32_ens.npy"
OUTPUT = ROOT / "paper/figs/xy_approx2_training.pdf"


def load_training() -> dict:
    data = np.load(TRAINING, allow_pickle=True).item()
    metadata = data["metadata"]
    if metadata["ndim"] != 2 or not np.isclose(metadata["beta"], 0.65):
        raise ValueError("expected the d=2, beta=0.65 XY penalty run")
    profiles = np.asarray(data["alphas"])
    if profiles.ndim != 4 or profiles.shape[-1] != 1:
        raise ValueError("the training artifact lacks per-epoch fixed profiles")
    if not np.allclose(profiles[:, 0, 0, 0], 1.0, atol=1e-6):
        raise ValueError("the penalty profiles are not normalized at the source")
    return data


class XYVarianceEvaluator:
    """Evaluate Var[Q]/Var[O] for many fixed XY profiles on one ensemble."""

    def __init__(self, fields: np.ndarray, beta: float, profiles: np.ndarray):
        n_samples, *lattice_shape = fields.shape
        if tuple(profiles.shape[1:]) != (*lattice_shape, 1):
            raise ValueError("profile and ensemble lattice shapes differ")
        self.beta = float(beta)
        self.n_samples = n_samples
        source = fields[:, 0, 0].astype(np.float64)
        self.centered_source = np.exp(1j * source)
        self.centered_source -= self.centered_source.mean()
        self.initial_variance = np.var(np.cos(source), ddof=1)

        # Every other bond has Delta f = 0 and cancels from Delta S exactly.
        support = np.any(profiles[..., 0] != 0, axis=0)
        field_cosines = []
        field_sines = []
        profile_differences = []
        for axis in range(len(lattice_shape)):
            forward_support = np.roll(support, -1, axis=axis)
            active = support | forward_support
            angle = fields - np.roll(fields, -1, axis=axis + 1)
            field_cosines.append(np.cos(angle[:, active]).astype(np.float64))
            field_sines.append(np.sin(angle[:, active]).astype(np.float64))
            profile_differences.append(
                (
                    profiles[..., 0]
                    - np.roll(profiles[..., 0], -1, axis=axis + 1)
                )[:, active].astype(np.float64)
            )
        self.cosines = np.concatenate(field_cosines, axis=1)
        self.sines = np.concatenate(field_sines, axis=1)
        self.profile_differences = np.concatenate(profile_differences, axis=1)

    def ratios(self, differences: np.ndarray, source_scale: np.ndarray) -> np.ndarray:
        """Return exact real-observable sample variance ratios by column."""
        # BLAS can leave floating-point status flags set despite finite matrix
        # products; check the products explicitly after silencing those flags.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            real_delta_s = -self.beta * (
                self.cosines @ (np.cosh(differences).T - 1.0)
            )
            imag_delta_s = self.beta * (self.sines @ np.sinh(differences).T)
        if not np.all(np.isfinite(real_delta_s)) or not np.all(
            np.isfinite(imag_delta_s)
        ):
            raise FloatingPointError("non-finite XY action difference")
        weight = np.exp(-real_delta_s - 1j * imag_delta_s - source_scale)
        q_real = (self.centered_source[:, None] * weight).real
        return np.var(q_real, axis=0, ddof=1) / self.initial_variance

    def ratio_at_scale(self, profile_difference: np.ndarray, alpha: float) -> float:
        differences = alpha * profile_difference[None, :]
        return float(self.ratios(differences, np.array([alpha]))[0])


def optimize_alpha(
    evaluator: XYVarianceEvaluator, profile_difference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Record accepted L-BFGS-B iterates, including the unit-scale start."""
    steps = [(1.0, evaluator.ratio_at_scale(profile_difference, 1.0))]

    def objective(alpha_array: np.ndarray) -> float:
        return evaluator.ratio_at_scale(profile_difference, float(alpha_array[0]))

    def record(alpha_array: np.ndarray) -> None:
        alpha = float(alpha_array[0])
        steps.append((alpha, objective(alpha_array)))

    result = minimize(
        objective,
        x0=np.array([1.0]),
        method="L-BFGS-B",
        bounds=[(0.0, 3.0)],
        callback=record,
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 100},
    )
    if not result.success:
        raise RuntimeError(f"alpha optimization failed: {result.message}")
    if not np.isclose(steps[-1][0], result.x[0], atol=1e-8):
        steps.append((float(result.x[0]), float(result.fun)))
    return np.asarray(steps, dtype=float).T


def create_figure(
    epoch_ratios: np.ndarray, alpha_steps: np.ndarray,
    optimized_ratios: np.ndarray,
) -> None:
    plt.rcParams.update(latex_font_params)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(5.4, 2.4))
    grid = figure.add_gridspec(
        1, 2, left=0.14, right=0.79, bottom=0.24, top=0.88,
        width_ratios=(1.6, 1.0), wspace=0.045,
    )
    variance_axis = figure.add_subplot(grid[0, 0])
    scaled_axis = figure.add_subplot(grid[0, 1], sharey=variance_axis)
    alpha_axis = scaled_axis.twinx()

    epochs = np.arange(1, len(epoch_ratios) + 1)
    variance_axis.plot(epochs, epoch_ratios, color="tab:blue", lw=1.25)
    variance_axis.axhline(1.0, color="black", lw=0.8, zorder=1)
    variance_axis.set_ylabel(
        r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$"
    )
    variance_axis.set_title(r"Step 1: train $f_x$ (with $f_0=1$)", fontsize=9, pad=4)
    variance_axis.set_xscale("log")
    variance_axis.set_xlim(1, len(epochs))
    variance_axis.set_xlabel("training epoch")
    variance_axis.text(
        0.965, 0.83,
        r"$=\langle e^{-2\,\mathrm{Re}\,\Delta S}\rangle$",
        transform=variance_axis.transAxes,
        ha="right", va="top",
        fontsize=plt.rcParams["axes.labelsize"],
    )

    steps = np.arange(len(alpha_steps))
    scaled_axis.plot(steps, optimized_ratios, marker="o", ms=3.5,
                     color="tab:blue", lw=1.2)
    scaled_axis.set_title(r"Step 2: optimize $\alpha$", fontsize=9, pad=4)
    variance_axis.set_yscale("log")
    variance_axis.set_ylim(0.13, 1.08)
    variance_axis.set_yticks([0.15, 0.2, 0.3, 0.5, 1.0])
    variance_axis.set_yticklabels(["0.15", "0.2", "0.3", "0.5", "1"])
    variance_axis.yaxis.set_minor_locator(NullLocator())
    scaled_axis.tick_params(axis="y", which="both", left=True, labelleft=False)
    alpha_axis.plot(steps, alpha_steps, marker="o", ms=3.5,
                    color="tab:purple", lw=1.2)
    scaled_axis.set_xlabel("optimization step")
    scaled_axis.set_xlim(0, max(1, len(steps) - 1))
    scaled_axis.set_xticks(steps)
    alpha_axis.set_ylim(0.95, 1.85)
    alpha_axis.set_yticks(
        [1.0, 1.25, 1.5, alpha_steps[-1]],
        labels=[r"$1$", r"$1.25$", r"$1.5$",
                rf"$\alpha^\star={alpha_steps[-1]:.2f}$"],
    )
    alpha_axis.set_ylabel(r"$\alpha$", rotation=270, labelpad=12,
                          color="tab:purple")
    alpha_axis.tick_params(axis="y", colors="tab:purple")

    for axis in (variance_axis, scaled_axis):
        axis.grid(axis="y", color="0.88", linewidth=0.5)
        axis.tick_params(direction="out", length=2.5, width=0.65)
    alpha_axis.tick_params(direction="out", length=2.5, width=0.65)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT)
    plt.close(figure)


def main() -> None:
    data = load_training()
    profiles = np.asarray(data["alphas"], dtype=np.float32)
    fields, beta = xy.load_ensemble(str(ENSEMBLE))
    evaluator = XYVarianceEvaluator(fields.numpy(), beta, profiles)
    epoch_ratios = np.empty(len(profiles), dtype=float)
    for start in range(0, len(profiles), 64):
        stop = min(start + 64, len(profiles))
        epoch_ratios[start:stop] = evaluator.ratios(
            evaluator.profile_differences[start:stop],
            np.ones(stop - start),
        )
    selected = np.asarray(data["f_x"], dtype=np.float32)
    if not np.allclose(selected[0, 0, 0], 1.0, atol=1e-6):
        raise ValueError("the selected field is not normalized at the source")
    # The best-window profile need not equal the last recorded epoch.
    selected_evaluator = XYVarianceEvaluator(fields.numpy(), beta, selected[None])
    alpha_steps, optimized_ratios = optimize_alpha(
        selected_evaluator, selected_evaluator.profile_differences[0]
    )
    if not np.isclose(optimized_ratios[-1], data["variance_ratio"], atol=2e-4):
        raise RuntimeError(
            "recomputed optimum disagrees with saved variance ratio: "
            f"{optimized_ratios[-1]:.8g} vs {data['variance_ratio']:.8g}; "
            f"alpha={alpha_steps[-1]:.8g}"
        )
    create_figure(epoch_ratios, alpha_steps, optimized_ratios)
    print(
        f"Wrote {OUTPUT}: unit ratio={optimized_ratios[0]:.5f}, "
        f"alpha*={alpha_steps[-1]:.5f}, optimized ratio={optimized_ratios[-1]:.5f}"
    )


if __name__ == "__main__":
    main()
