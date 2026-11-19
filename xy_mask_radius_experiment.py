"""Scan the periodic support radius of masked XY one-point deformations.

Every deformation in the manuscript is trained on a periodic box of fixed
radius around the source.  Section "Temperature and dimension dependence"
predicts, through Approximation 3, that the attainable reduction is set by
the lattice capacity of that support, which in ``d=2`` improves only
logarithmically with the radius ``R``.  This script measures the prediction
directly: it retrains the full variance objective at a sequence of radii on
one ensemble and records ``Var[Q]/Var[O]`` for each.

Training uses the same objective and network as ``xy_OPF_analysis.py
--loss-function variance``.  Two optimizer settings differ, because the
published plateau schedule does not converge uniformly over this scan: the
source deformation grows slowly through ``F_0≈ 1.2`` at the larger radii,
and both the relative plateau stopping rule and the plateau learning-rate
collapse freeze a run there, well short of its optimum.  This scan therefore
anneals a fixed epoch budget with a cosine schedule and no early stopping,
so every radius receives an identical amount of optimization.  Each radius
is repeated over several network initializations, and the attainable ratio
is the best of those restarts.

The deterministic Approximation 3 profile is solved at every radius as well.
It costs nothing, isolates the capacity argument from training noise, and
provides the target that direct variance training should match or beat.

The scan stops at ``R = L/2 - 1``.  At ``R = L/2`` the periodic box covers
the whole lattice, where a constant imaginary shift leaves the XY action
invariant and rescales the one-point function by ``e^{-F_0}``.  Signal and
noise are then scaled together, so ``Var[Q]/Var[O]`` runs to zero at no
action cost without improving the signal-to-noise ratio.  That limit is the
unmasked degeneracy of Fig. 1, not a variance reduction, and the training
objective is unbounded there.

Run, for example:

    python xy_mask_radius_experiment.py --run
    python xy_mask_radius_experiment.py --no-run    # replot saved results
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lattice-mplconfig")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullFormatter
import numpy as np
import torch

import XY_utils as xy
from actions import XYAction
from observables import XYOnePtFn
from plot_settings import fig_width, latex_font_params
from xy_opf.objectives import train_variance_loss
from xy_opf.optimization import fit_lattice_laplace_profile
from xy_opf.training import make_split_mask


# Overwritten from CLI arguments in main(); the values below are the
# defaults for the original d=2, beta=0.65, L=32 scan.
NDIM = 2
BETA = 0.65
LATTICE_SIZE = 32
SOURCE = (0, 0)
ENSEMBLE = Path(
    f"data/xy_dim{NDIM}_b{BETA:.2f}_L{LATTICE_SIZE}_ens.npy"
)
SCAN_DATA_DIRECTORY = Path("training_data/xy_mask_radius_scan")
FIGURES_DIRECTORY = Path("paper/figs")
OUTPUT_STEM = "xy_mask_radius"

# The box side is min(2R+1, L), so R=15 is the largest support that is still
# a proper subset of the 32^2 lattice.  R=6 is the radius used everywhere
# else in the manuscript at d=2,3; d=1 instead publishes R=20 (see
# xy_opf/README.md).
DEFAULT_RADII = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15)
# Roughly one restart in three stalls while the source deformation is still
# growing, independently of the radius, so the scan uses five of them.
DEFAULT_SEEDS = (1234, 1235, 1236, 1237, 1238)
PUBLISHED_RADIUS = 6

# Per-dimension defaults used when --radii/--published-radius are not given
# explicitly, for the d=1 and d=3 scans added alongside the original d=2 one.
DIMENSION_DEFAULT_RADII = {
    1: (0, 2, 4, 6, 8, 10, 12, 15, 18, 20, 24, 28, 31),
    2: DEFAULT_RADII,
    3: (0, 1, 2, 3, 4, 5, 6, 7),
}
DIMENSION_PUBLISHED_RADIUS = {1: 20, 2: 6, 3: 6}

plt.rcParams.update(latex_font_params)
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def maximum_radius(lattice_size: int) -> int:
    """Return the largest radius whose support is not the whole lattice.

    At ``lattice_size // 2`` the periodic box covers every site, which is
    the global-orbit degeneracy described in this module's docstring.
    """
    return lattice_size // 2 - 1


def run_savename(radius: int, seed: int) -> Path:
    """Keep the scan's training artifacts outside the standard XY results."""
    return SCAN_DATA_DIRECTORY / (
        f"xy_dim{NDIM}_b{BETA:.2f}_L{LATTICE_SIZE}_"
        f"R{radius:02d}_seed{seed}_loss-variance_training.npy"
    )


def summary_savename() -> Path:
    """Return the collected scan results used by the plot."""
    return SCAN_DATA_DIRECTORY / (
        f"xy_mask_radius_scan_dim{NDIM}_b{BETA:.2f}_"
        f"L{LATTICE_SIZE}_summary.npy"
    )


def set_seed(seed: int) -> None:
    """Reset every torch RNG so each radius starts from the same network."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_namespace(
    args: argparse.Namespace, radius: int
) -> argparse.Namespace:
    """Build the argument namespace ``train_variance_loss`` expects."""
    return argparse.Namespace(
        mask_radius=int(radius),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_cooldown=args.lr_cooldown,
        min_learning_rate=args.min_learning_rate,
        max_training_time=args.max_training_time,
        plateau_window=args.plateau_window,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
        plateau_threshold_mode=args.plateau_threshold_mode,
        early_stopping=args.early_stopping,
        unet_hidden_dim=args.unet_hidden_dim,
    )


def protocol_tag(args: argparse.Namespace) -> str:
    """Summarize the optimization settings every run must share."""
    return (
        f"{args.lr_scheduler}_lr{args.learning_rate:g}_"
        f"bs{args.batch_size}_ep{args.epochs}_"
        f"es{int(bool(args.early_stopping))}"
    )


def annotate_run(
    artifact: Path, *, radius: int, seed: int, protocol: str
) -> None:
    """Record the scan coordinates that the shared saver does not store."""
    saved = np.load(artifact, allow_pickle=True)
    if saved.shape != () or not isinstance(saved.item(), dict):
        raise ValueError(f"invalid training artifact: {artifact}")
    data = saved.item()
    metadata = data.setdefault("metadata", {})
    metadata.update(
        study="xy_mask_radius_scan",
        mask_radius=int(radius),
        initialization_rng_seed=int(seed),
        optimization_protocol=protocol,
    )
    np.save(artifact, data, allow_pickle=True)


def read_run(
    artifact: Path, *, radius: int, seed: int, protocol: str
) -> dict[str, float]:
    """Load and validate one saved radius/seed training result."""
    saved = np.load(artifact, allow_pickle=True)
    if saved.shape != () or not isinstance(saved.item(), dict):
        raise ValueError(f"invalid training artifact: {artifact}")
    data = saved.item()
    metadata = data.get("metadata", {})
    expected = {
        "loss_function": "variance",
        "ndim": NDIM,
        "lattice_size": (LATTICE_SIZE,) * NDIM,
        "source": SOURCE,
        "mask_radius": int(radius),
        "initialization_rng_seed": int(seed),
        "optimization_protocol": protocol,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{artifact} has {key}={metadata.get(key)!r}; "
                f"expected {value!r}"
            )
    if not np.isclose(float(metadata.get("beta", np.nan)), BETA):
        raise ValueError(f"{artifact} was not trained at beta={BETA:g}")
    variance_ratio = float(data["variance_ratio"])
    if not math.isfinite(variance_ratio) or variance_ratio <= 0:
        raise ValueError(f"invalid variance ratio in {artifact}")
    return {
        "variance_ratio": variance_ratio,
        "field_at_source": float(metadata["field_at_source"]),
        "best_epoch": int(data.get("best_epoch", -1)),
        "epochs_run": int(len(np.asarray(data["train_var"]))),
        "stop_reason": str(data.get("stop_reason", "")),
    }


def train_run(
    fields: torch.Tensor,
    obs: XYOnePtFn,
    split_mask: torch.Tensor,
    *,
    radius: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Train one radius/seed combination and return its scan entry."""
    artifact = run_savename(radius, seed)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    protocol = protocol_tag(args)
    set_seed(seed)
    train_variance_loss(
        fields=fields,
        ndim=NDIM,
        beta=BETA,
        obs=obs,
        obs_str=f"OPF {SOURCE}, R={radius}, seed={seed}",
        args=training_namespace(args, radius),
        source=SOURCE,
        training_savename=artifact,
        split_mask=split_mask,
    )
    annotate_run(artifact, radius=radius, seed=seed, protocol=protocol)
    return read_run(artifact, radius=radius, seed=seed, protocol=protocol)


def approximation_three_scan(
    fields: torch.Tensor,
    radii: list[int],
    *,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Solve the masked lattice Laplace profile at every scanned radius."""
    ratios, sources, capacities = [], [], []
    for radius in radii:
        result = fit_lattice_laplace_profile(
            fields,
            BETA,
            SOURCE,
            mask_radius=int(radius),
            alpha_bounds=tuple(args.alpha_bounds),
            alpha_grid_points=args.alpha_grid_points,
            alpha_xatol=args.alpha_xatol,
        )
        diagnostics = result["optimization_diagnostics"]
        scaled_field = np.asarray(result["scaled_field"], dtype=float)
        ratios.append(float(result["variance_ratio"]))
        sources.append(float(scaled_field[*SOURCE, 0]))
        # The solver scales the quadratic penalty by beta; E[f] is the
        # lattice capacity of the support for a profile pinned to one.
        capacities.append(float(diagnostics["quadratic_penalty"]) / BETA)
        print(
            f"Approximation 3: R={radius:2d}, "
            f"E[f]={capacities[-1]:.6g}, "
            f"alpha={result['alpha_scale']:.6g}, "
            f"ratio={ratios[-1]:.6g}"
        )
    return {
        "variance_ratios": np.asarray(ratios, dtype=float),
        "field_at_source": np.asarray(sources, dtype=float),
        "capacity": np.asarray(capacities, dtype=float),
    }


def collect_scan(
    radii: list[int],
    seeds: list[int],
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Train or load every radius/seed run and build the summary payload."""
    fields, _ = xy.load_ensemble(
        str(ENSEMBLE), decorrelation_factor=args.decorrelation
    )
    latsize = tuple(int(length) for length in fields.shape[1:])
    if latsize != (LATTICE_SIZE,) * NDIM:
        raise ValueError(f"{ENSEMBLE} has lattice size {latsize}")
    obs = XYOnePtFn(latsize, SOURCE, XYAction(BETA, NDIM))

    set_seed(args.split_seed)
    split_mask = make_split_mask(fields)

    shape = (len(radii), len(seeds))
    variance_ratios = np.full(shape, np.nan)
    field_at_source = np.full(shape, np.nan)
    best_epochs = np.full(shape, -1, dtype=int)
    epochs_run = np.full(shape, -1, dtype=int)
    stop_reasons: list[list[str]] = []

    for radius_index, radius in enumerate(radii):
        radius_reasons = []
        for seed_index, seed in enumerate(seeds):
            artifact = run_savename(radius, seed)
            entry = None
            if artifact.exists() and not args.overwrite:
                try:
                    entry = read_run(
                        artifact,
                        radius=radius,
                        seed=seed,
                        protocol=protocol_tag(args),
                    )
                    print(f"Loaded R={radius}, seed={seed} from {artifact}")
                except (KeyError, ValueError) as error:
                    print(
                        f"Saved R={radius}, seed={seed} result is "
                        f"incompatible; retraining it ({error})"
                    )
            if entry is None:
                if not args.run:
                    raise FileNotFoundError(
                        f"no saved result for R={radius}, seed={seed} at "
                        f"{artifact}; rerun with --run"
                    )
                print(f"\nTraining R={radius}, seed={seed} ...")
                entry = train_run(
                    fields,
                    obs,
                    split_mask,
                    radius=radius,
                    seed=seed,
                    args=args,
                )
            variance_ratios[radius_index, seed_index] = entry["variance_ratio"]
            field_at_source[radius_index, seed_index] = entry["field_at_source"]
            best_epochs[radius_index, seed_index] = entry["best_epoch"]
            epochs_run[radius_index, seed_index] = entry["epochs_run"]
            radius_reasons.append(entry["stop_reason"])
        stop_reasons.append(radius_reasons)

    approximation_three = approximation_three_scan(fields, radii, args=args)
    return {
        "metadata": {
            "study": "xy_mask_radius_scan",
            "model": "XY",
            "ndim": NDIM,
            "beta": BETA,
            "lattice_size": LATTICE_SIZE,
            "source": SOURCE,
            "observable": "exp(i*theta_source)",
            "ensemble": str(ENSEMBLE),
            "configuration_count": int(len(fields)),
            "decorrelation": float(args.decorrelation),
            "split_seed": int(args.split_seed),
            "published_mask_radius": PUBLISHED_RADIUS,
            "maximum_radius": maximum_radius(LATTICE_SIZE),
            "optimization_protocol": protocol_tag(args),
            "training_config": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "lr_scheduler": args.lr_scheduler,
                "min_learning_rate": float(args.min_learning_rate),
                "early_stopping": bool(args.early_stopping),
            },
            "reported_statistic": "minimum variance ratio over restarts",
        },
        "radii": np.asarray(radii, dtype=int),
        "seeds": np.asarray(seeds, dtype=int),
        "variance_ratios": variance_ratios,
        "field_at_source": field_at_source,
        "best_epochs": best_epochs,
        "epochs_run": epochs_run,
        "stop_reasons": stop_reasons,
        "approx3_variance_ratios": approximation_three["variance_ratios"],
        "approx3_field_at_source": approximation_three["field_at_source"],
        "approx3_capacity": approximation_three["capacity"],
    }


def save_summary(summary: dict[str, object]) -> tuple[Path, Path]:
    """Write the scan summary as a numpy artifact and a readable JSON copy."""
    artifact = summary_savename()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.save(artifact, summary, allow_pickle=True)

    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    json_path = FIGURES_DIRECTORY / f"{OUTPUT_STEM}.json"
    serializable = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in summary.items()
    }
    json_path.write_text(json.dumps(serializable, indent=2) + "\n")
    return artifact, json_path


def load_summary() -> dict[str, object]:
    """Load a previously saved scan summary."""
    artifact = summary_savename()
    saved = np.load(artifact, allow_pickle=True)
    if saved.shape != () or not isinstance(saved.item(), dict):
        raise ValueError(f"invalid scan summary: {artifact}")
    return saved.item()


def make_plot(summary: dict[str, object]) -> plt.Figure:
    """Draw the variance ratio against the periodic support radius."""
    radii = np.asarray(summary["radii"], dtype=float)
    ratios = np.asarray(summary["variance_ratios"], dtype=float)
    approx3 = np.asarray(summary["approx3_variance_ratios"], dtype=float)
    if ratios.ndim != 2 or len(radii) != len(ratios):
        raise ValueError("variance ratios must be one row per scanned radius")
    if not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise ValueError("the scan contains invalid trained variance ratios")
    if not np.isfinite(approx3).all() or np.any(approx3 <= 0):
        raise ValueError("the scan contains invalid Approximation 3 ratios")

    # The ratio is the minimum of a non-convex objective, so the attainable
    # value is the best restart.  The remaining restarts are kept in the
    # saved summary rather than drawn.
    best = ratios.min(axis=1)

    figure, axis = plt.subplots(
        figsize=(0.52 * fig_width, 0.40 * fig_width)
    )
    axis.axhline(
        1.0,
        color="black",
        linewidth=0.8,
        zorder=1,
    )
    axis.axvline(
        PUBLISHED_RADIUS,
        color="0.45",
        linestyle=":",
        linewidth=0.8,
        zorder=1,
    )
    axis.plot(
        radii,
        best,
        color="tab:blue",
        marker="o",
        markersize=3.2,
        linestyle="-",
        linewidth=1.0,
        label="direct variance",
        zorder=4,
    )
    axis.plot(
        radii,
        approx3,
        color="tab:red",
        marker="D",
        markersize=3.4,
        markerfacecolor="none",
        markeredgewidth=0.9,
        linestyle="--",
        linewidth=0.9,
        label="Approx. 3 (smooth deformation)",
        zorder=3,
    )

    axis.set_yscale("log")
    axis.set_xlim(radii.min() - 0.6, radii.max() + 0.6)
    axis.set_ylim(0.6 * ratios.min(), 1.45)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    y_ticks = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    axis.set_yticks(y_ticks, [f"{tick:g}" for tick in y_ticks])
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel(r"periodic support radius $R/a$")
    axis.set_ylabel(
        r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$",
        labelpad=3,
    )
    axis.set_title(
        rf"$XY$, $d={NDIM}$, $\beta={BETA:g}$, "
        rf"${LATTICE_SIZE}^{{{NDIM}}}$",
        pad=3,
    )
    axis.grid(color="0.8", linewidth=0.5, alpha=0.5, which="both")
    axis.set_axisbelow(True)
    axis.tick_params(direction="out", length=3.5, width=0.8, pad=2)
    axis.annotate(
        rf"$R={PUBLISHED_RADIUS}$",
        xy=(PUBLISHED_RADIUS, 1.0),
        xytext=(2.5, -2.5),
        textcoords="offset points",
        fontsize=6.5,
        color="0.35",
        va="top",
        ha="left",
    )
    axis.legend(
        loc="lower left",
        frameon=False,
        handletextpad=0.4,
        labelspacing=0.25,
        borderaxespad=0.4,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
    figure.subplots_adjust(left=0.19, right=0.98, bottom=0.17, top=0.98)
    return figure


def parse_args() -> argparse.Namespace:
    """Parse the scan grid and the training controls it varies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dim",
        type=int,
        choices=(1, 2, 3),
        default=NDIM,
        help="spatial dimension of the XY ensemble to scan",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help=f"inverse temperature of the ensemble (default: {BETA:g} at --dim 2)",
    )
    parser.add_argument(
        "--lattice-size",
        type=int,
        default=None,
        help=f"lattice side length L (default: {LATTICE_SIZE} at --dim 2)",
    )
    parser.add_argument(
        "--radii",
        nargs="+",
        type=int,
        default=None,
        help=(
            "periodic support radii to scan (default: a per-dimension grid "
            "within [0, L/2-1])"
        ),
    )
    parser.add_argument(
        "--published-radius",
        type=int,
        default=None,
        help="radius used elsewhere in the manuscript (default: per-dimension)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="network-initialization seeds repeated at every radius",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=20251023,
        help="seed for the one train/test split shared by every run",
    )
    parser.add_argument(
        "--run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train missing runs; --no-run plots only saved results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="retrain every radius/seed even when a result is saved",
    )
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lr-scheduler",
        choices=("constant", "cosine", "plateau"),
        default="cosine",
        help=(
            "cosine anneals the fixed budget identically at every radius; "
            "the plateau schedule used elsewhere collapses the rate before "
            "the larger supports converge"
        ),
    )
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-cooldown", type=int, default=1)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-training-time", type=float, default=None)
    parser.add_argument("--plateau-window", type=int, default=20)
    parser.add_argument("--plateau-patience", type=int, default=40)
    parser.add_argument("--plateau-min-delta", type=float, default=1e-3)
    parser.add_argument(
        "--plateau-threshold-mode", choices=("abs", "rel"), default="rel"
    )
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "off by default: the relative plateau rule fires while the "
            "source deformation is still growing at the larger radii"
        ),
    )
    parser.add_argument("--unet-hidden-dim", type=int, default=None)
    parser.add_argument("--decorrelation", type=float, default=1.0)
    parser.add_argument(
        "--alpha-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(-5.0, 5.0),
        help="bounds for the Approximation 3 amplitude scan",
    )
    parser.add_argument("--alpha-grid-points", type=int, default=41)
    parser.add_argument("--alpha-xatol", type=float, default=1e-4)
    args = parser.parse_args()

    dimension_lattice_defaults = {1: 64, 2: LATTICE_SIZE, 3: 16}
    if args.lattice_size is None:
        args.lattice_size = dimension_lattice_defaults[args.dim]
    if args.beta is None:
        if args.dim != 2:
            parser.error(f"--beta is required for --dim {args.dim}")
        args.beta = BETA
    if args.radii is None:
        args.radii = list(DIMENSION_DEFAULT_RADII[args.dim])
    if args.published_radius is None:
        args.published_radius = DIMENSION_PUBLISHED_RADIUS[args.dim]

    limit = maximum_radius(args.lattice_size)
    if not args.radii:
        parser.error("--radii must contain at least one radius")
    if len(set(args.radii)) != len(args.radii):
        parser.error("--radii must not repeat a radius")
    if any(radius < 0 or radius > limit for radius in args.radii):
        parser.error(
            f"--radii must lie between 0 and {limit}; a radius of "
            f"{args.lattice_size // 2} covers the whole lattice, where the "
            "constant global-orbit mode makes the objective unbounded"
        )
    if args.published_radius < 0 or args.published_radius > limit:
        parser.error(f"--published-radius must lie between 0 and {limit}")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain distinct values")
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be positive and --batch-size at least 2")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive and finite")
    if not math.isfinite(args.min_learning_rate) or args.min_learning_rate < 0:
        parser.error("--min-learning-rate must be non-negative and finite")
    if (
        args.lr_scheduler != "constant"
        and args.min_learning_rate > args.learning_rate
    ):
        parser.error("--min-learning-rate cannot exceed --learning-rate")
    if not 0 < args.lr_factor < 1:
        parser.error("--lr-factor must lie strictly between 0 and 1")
    if args.lr_patience < 1 or args.lr_cooldown < 0:
        parser.error(
            "--lr-patience must be positive and --lr-cooldown non-negative"
        )
    if args.max_training_time is not None and (
        not math.isfinite(args.max_training_time)
        or args.max_training_time <= 0
    ):
        parser.error("--max-training-time must be positive and finite")
    if (
        args.plateau_window < 1
        or args.plateau_patience < 1
        or not math.isfinite(args.plateau_min_delta)
        or args.plateau_min_delta < 0
    ):
        parser.error("invalid plateau settings")
    if args.plateau_threshold_mode == "rel" and args.plateau_min_delta >= 1:
        parser.error("relative --plateau-min-delta must be smaller than 1")
    if not math.isfinite(args.decorrelation) or args.decorrelation <= 0:
        parser.error("--decorrelation must be positive and finite")
    if args.alpha_bounds[0] >= args.alpha_bounds[1]:
        parser.error("--alpha-bounds LOWER must be smaller than UPPER")
    if args.alpha_grid_points < 3:
        parser.error("--alpha-grid-points must be at least 3")
    if args.alpha_xatol <= 0:
        parser.error("--alpha-xatol must be positive")
    args.radii = sorted(args.radii)
    return args


def main() -> None:
    """Run or load the radius scan, save the results, and plot them."""
    args = parse_args()

    global NDIM, BETA, LATTICE_SIZE, SOURCE, ENSEMBLE, PUBLISHED_RADIUS
    global OUTPUT_STEM
    NDIM = args.dim
    BETA = args.beta
    LATTICE_SIZE = args.lattice_size
    SOURCE = (0,) * NDIM
    ENSEMBLE = Path(f"data/xy_dim{NDIM}_b{BETA:.2f}_L{LATTICE_SIZE}_ens.npy")
    PUBLISHED_RADIUS = args.published_radius
    if NDIM == 2 and math.isclose(BETA, 0.65) and LATTICE_SIZE == 32:
        OUTPUT_STEM = "xy_mask_radius"
    else:
        OUTPUT_STEM = f"xy_mask_radius_dim{NDIM}_b{BETA:.2f}_L{LATTICE_SIZE}"

    if not ENSEMBLE.exists():
        raise FileNotFoundError(ENSEMBLE)

    summary = collect_scan(args.radii, args.seeds, args=args)
    artifact, json_path = save_summary(summary)
    print(f"\nWrote {artifact}")
    print(f"Wrote {json_path}")

    radii = np.asarray(summary["radii"], dtype=int)
    ratios = np.asarray(summary["variance_ratios"], dtype=float)
    approx3 = np.asarray(summary["approx3_variance_ratios"], dtype=float)
    capacity = np.asarray(summary["approx3_capacity"], dtype=float)
    print("\n R   best    worst   Approx. 3   E[f]")
    for index, radius in enumerate(radii):
        row = ratios[index]
        print(
            f"{radius:2d}   {row.min():.4f}  {row.max():.4f}  "
            f"{approx3[index]:9.4f}   {capacity[index]:.4f}"
        )

    figure = make_plot(summary)
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pdf_output = FIGURES_DIRECTORY / f"{OUTPUT_STEM}.pdf"
    png_output = FIGURES_DIRECTORY / f"{OUTPUT_STEM}.png"
    figure.savefig(pdf_output)
    figure.savefig(png_output, dpi=240)
    plt.close(figure)
    print(f"Wrote {pdf_output}")
    print(f"Wrote {png_output}")


if __name__ == "__main__":
    sys.exit(main())
