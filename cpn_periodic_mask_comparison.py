"""Compare masked and unmasked CP(2) one-point-function deformations.

The masked result is stored separately from the standard CP(N) variance
artifacts because this locality comparison trains its deformation from a
fresh near-zero U-Net initialization. The unmasked baseline is loaded from its
comparison artifact when available.

Run, for example:

    python cpn_periodic_mask_comparison.py --mask-radius 6
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.optim as optim

from plot_settings import fig_width_pt, inches_per_pt, latex_font_params

from actions import CPnGaugeAction
from cpn_opf.loading import load_ensemble
from cpn_opf.objectives import init_variance_network
from cpn_opf.persistence import load_deformation_field
from cpn_opf.training import (
    make_deformation_mask,
    make_model,
    make_split_mask,
    save_training_result,
)
from losses import VarianceLoss
from networks import UNet
from observables import CPnOnePtFn
from training import Trainer


plt.rcParams.update(latex_font_params)
# Match the manuscript's AMS and bbold packages for CP^N titles.
plt.rcParams["text.latex.preamble"] = r"\usepackage{amssymb}\usepackage{bbold}"


N = 2
NDIM = 2
LATTICE_SIZE = 64
SOURCE = (0, 0)
COLOR_INDICES = (0, 1)
DEFAULT_ENSEMBLE = Path("data/cp2_gauge_dim2_b3.30_L64_ens.npy")
DEFAULT_OUTPUT = Path("paper/figs/locality_cpn.pdf")
COMPARISON_DATA_DIRECTORY = Path(
    "training_data/locality_comparison_near_zero"
)
MASKED_LABEL = "masked"
UNMASKED_LABEL = "not masked"


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device, or choose the best available backend."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Reset every torch RNG used by the two otherwise identical runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def beta_from_filename(ensemble: Path) -> float:
    """Read the beta token from a canonical CP(N) ensemble filename."""
    match = re.search(r"_b(?P<beta>[^_]+)_L", ensemble.name)
    if match is None:
        raise ValueError(
            "the ensemble filename must include a beta token like '_b3.30_L64'"
        )
    try:
        beta = float(match.group("beta"))
    except ValueError as error:
        raise ValueError(
            f"could not parse beta from ensemble filename: {ensemble.name}"
        ) from error
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("ensemble beta must be positive and finite")
    return beta


def masked_comparison_savename(beta: float) -> Path:
    """Keep the near-zero masked result outside standard CP(N) artifacts."""
    return COMPARISON_DATA_DIRECTORY / (
        f"cp{N}_dim{NDIM}_b{beta:g}_L{LATTICE_SIZE}_"
        f"i{COLOR_INDICES[0]}_j{COLOR_INDICES[1]}_"
        "masked_nearzero_outscale10_lr3e-3_bs256_seed1234_1000ep_training.npy"
    )


def unmasked_comparison_savename(beta: float) -> Path:
    """Keep the selected unmasked baseline in the comparison data folder."""
    return COMPARISON_DATA_DIRECTORY / (
        f"cp{N}_dim{NDIM}_b{beta:g}_L{LATTICE_SIZE}_"
        f"i{COLOR_INDICES[0]}_j{COLOR_INDICES[1]}_"
        "unmasked_baseline_source_1000ep_training.npy"
    )


def load_comparison_case(
    artifact: Path,
    *,
    beta: float,
) -> tuple[torch.Tensor, float, dict]:
    """Load and validate one CP(N) variance-training result."""
    if not artifact.exists():
        raise FileNotFoundError(
            f"no CP(N) variance-training artifact at {artifact}"
        )
    saved = np.load(artifact, allow_pickle=True)
    if saved.shape != () or not isinstance(saved.item(), dict):
        raise ValueError(f"invalid masked training artifact: {artifact}")
    data = saved.item()
    metadata = data.get("metadata", {})
    expected = {
        "loss_function": "variance",
        "cp_n": N,
        "ndim": NDIM,
        "lattice_size": (LATTICE_SIZE,) * NDIM,
        "source": SOURCE,
        "i": COLOR_INDICES[0],
        "j": COLOR_INDICES[1],
        "action_type": "gauge",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{artifact} has {key}={metadata.get(key)!r}; expected {
                    value!r}"
            )
    if not np.isclose(float(metadata.get("beta", np.nan)), beta):
        raise ValueError(
            f"{artifact} does not match the ensemble beta={beta:g}"
        )
    field, source = load_deformation_field(artifact)
    if source != SOURCE or field.shape != (LATTICE_SIZE,) * NDIM + (N + 1,):
        raise ValueError(f"masked field in {
                         artifact} has incompatible geometry")
    try:
        variance_ratio = float(data["variance_ratio"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"no variance ratio stored in {artifact}") from error
    if not math.isfinite(variance_ratio):
        raise ValueError(f"non-finite variance ratio in {artifact}")
    return torch.as_tensor(field, dtype=torch.float32), variance_ratio, metadata


def load_masked_case(
    artifact: Path,
    *,
    beta: float,
) -> tuple[torch.Tensor, float]:
    """Load and validate a saved masked comparison result."""
    field, variance_ratio, metadata = load_comparison_case(
        artifact, beta=beta
    )
    if (
        "periodic_box_mask" in metadata
        and metadata["periodic_box_mask"] is not True
    ):
        raise ValueError(f"{artifact} is not a saved masked comparison")
    return field, variance_ratio


def load_unmasked_case(
    artifact: Path,
    *,
    beta: float,
) -> tuple[torch.Tensor, float]:
    """Load a previously saved unmasked comparison result."""
    _field, _ratio, metadata = load_comparison_case(artifact, beta=beta)
    if metadata.get("periodic_box_mask") is not False:
        raise ValueError(f"{artifact} is not a saved unmasked comparison")
    return _field, _ratio


def normalize_training_curve(
    values: np.ndarray,
    initial_variance: float,
    *,
    label: str,
) -> np.ndarray:
    """Normalize a recorded variance-training history by its initial value."""
    curve = np.asarray(values, dtype=float)
    if curve.ndim != 1:
        raise ValueError(f"{label} training variance must be one-dimensional")
    if not math.isfinite(initial_variance) or initial_variance <= 0:
        raise ValueError(f"{label} has an invalid initial variance")
    finite = np.isfinite(curve)
    if not finite.any() or np.any(curve[finite] < 0):
        raise ValueError(f"{label} has no usable variance-training history")
    return curve / initial_variance


def minimum_training_epoch(values: np.ndarray) -> int:
    """Return the epoch with the lowest finite training variance."""
    curve = np.asarray(values, dtype=float)
    if curve.ndim != 1:
        raise ValueError("training variance must be one-dimensional")
    finite_epochs = np.flatnonzero(np.isfinite(curve))
    if finite_epochs.size == 0:
        raise ValueError("training history has no finite variance values")
    return int(finite_epochs[np.argmin(curve[finite_epochs])])


def load_training_minimum_field(
    artifact: Path,
    *,
    smooth_window: int | None = None,
) -> tuple[torch.Tensor, int, float]:
    """Load the field at the raw or smoothed training-history minimum."""
    saved = np.load(artifact, allow_pickle=True)
    if saved.shape != () or not isinstance(saved.item(), dict):
        raise ValueError(f"invalid training artifact: {artifact}")
    data = saved.item()
    if "train_var" not in data or "initial_var" not in data:
        raise ValueError(f"no variance-training history stored in {artifact}")
    train_variance = np.asarray(data["train_var"], dtype=float)
    selection_history = (
        train_variance
        if smooth_window is None
        else moving_median(train_variance, window=smooth_window)
    )
    epoch = minimum_training_epoch(selection_history)
    alphas = data.get("alphas")
    if alphas is None or len(alphas) != len(data["train_var"]):
        raise ValueError(f"no per-epoch deformation fields stored in {artifact}")
    initial_variance = float(data["initial_var"])
    if not math.isfinite(initial_variance) or initial_variance <= 0:
        raise ValueError(f"invalid initial variance stored in {artifact}")
    train_ratio = float(data["train_var"][epoch]) / initial_variance
    return torch.as_tensor(alphas[epoch], dtype=torch.float32), epoch, train_ratio


def load_training_curve(artifact: Path) -> np.ndarray:
    """Load the variance-only training history from a saved result artifact."""
    saved = np.load(artifact, allow_pickle=True)
    if saved.shape != () or not isinstance(saved.item(), dict):
        raise ValueError(f"invalid training artifact: {artifact}")
    data = saved.item()
    if "train_var" not in data or "initial_var" not in data:
        raise ValueError(f"no variance-training history stored in {artifact}")
    return normalize_training_curve(
        data["train_var"],
        float(data["initial_var"]),
        label=str(artifact),
    )


def moving_median(values: np.ndarray, window: int = 25) -> np.ndarray:
    """Reduce minibatch noise with a centered rolling median."""
    curve = np.asarray(values, dtype=float)
    if curve.ndim != 1 or window < 1:
        raise ValueError("a one-dimensional curve and positive window are required")
    radius = window // 2
    smoothed = np.full(curve.shape, np.nan, dtype=float)
    for index in range(len(curve)):
        if not np.isfinite(curve[index]):
            continue
        start = max(0, index - radius)
        stop = min(len(curve), index + radius + 1)
        values_in_window = curve[start:stop]
        values_in_window = values_in_window[np.isfinite(values_in_window)]
        if values_in_window.size:
            smoothed[index] = np.median(values_in_window)
    return smoothed


def train_case(
    fields: torch.Tensor,
    links: torch.Tensor,
    split_mask: torch.Tensor,
    *,
    beta: float,
    use_periodic_box_mask: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Trainer, torch.Tensor, float]:
    """Train one variance-reduction field, optionally with OPF support."""
    set_seed(args.seed)
    latsize = tuple(int(length) for length in fields.shape[1:-1])
    model = make_model(
        fields,
        links,
        n=N,
        action_type="gauge",
        split_mask=split_mask,
        device=device,
    )
    action = CPnGaugeAction(beta, N, latsize, device=device)
    observable = CPnOnePtFn(
        N,
        latsize,
        i=COLOR_INDICES[0],
        j=COLOR_INDICES[1],
        source=SOURCE,
        action=action,
        device=device,
    )
    loss = VarianceLoss(model, observable)
    network = UNet(
        input_dim=observable.grid.shape[-1],
        output_dim=model.DOF,
        ndim=NDIM,
        hd=args.unet_hidden_dim,
    ).to(device)
    init_variance_network(network)
    if args.initial_output_scale != 1.0:
        with torch.no_grad():
            network.output.weight.mul_(args.initial_output_scale)
            if network.output.bias is not None:
                network.output.bias.mul_(args.initial_output_scale)
    mask = (
        make_deformation_mask(
            latsize,
            SOURCE,
            args.mask_radius,
            model.DOF,
            device=device,
        )
        if use_periodic_box_mask
        else None
    )
    trainer = Trainer(
        model,
        loss,
        optim.Adam(network.parameters(), lr=args.learning_rate, eps=1e-6),
        args.epochs,
        args.batch_size,
    )
    trainer.train(
        network,
        mask=mask,
        plot=False,
        max_training_time=args.max_training_time,
        plateau_window=args.plateau_window,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
        plateau_threshold_mode=args.plateau_threshold_mode,
        early_stopping=args.early_stopping,
        lr_scheduler=args.lr_scheduler,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_cooldown=args.lr_cooldown,
        min_lr=args.min_learning_rate,
    )
    training_minimum_epoch = minimum_training_epoch(
        np.asarray(trainer.train_var, dtype=float)
    )
    # alpha_record and train_var are recorded from the same pre-update field.
    best_field = trainer.alpha_record[training_minimum_epoch].detach()
    trainer.training_minimum_epoch = training_minimum_epoch
    variance_ratio = float(
        (loss(alphas=best_field, sampletype="all") / loss.initial_var)
        .detach()
        .cpu()
    )
    return trainer, best_field, variance_ratio


def save_comparison_case(
    artifact: Path,
    trainer: Trainer,
    field: torch.Tensor,
    variance_ratio: float,
    *,
    beta: float,
    use_periodic_box_mask: bool,
    mask_radius: int,
    rng_seed: int,
    initial_output_scale: float = 1.0,
    selected_epoch: int | None = None,
    training_variance_ratio: float | None = None,
    selection_metric: str = "minimum_training_variance",
) -> None:
    """Persist one comparison run with explicit initialization provenance."""
    save_training_result(
        artifact,
        trainer,
        n=N,
        action_type="gauge",
        ndim=NDIM,
        beta=beta,
        latsize=(LATTICE_SIZE,) * NDIM,
        source=SOURCE,
        i=COLOR_INDICES[0],
        j=COLOR_INDICES[1],
        learned_field=field,
        variance_ratio=variance_ratio,
        loss_function="variance",
        extra_metadata={
            "comparison_study": "masked_vs_unmasked",
            "periodic_box_mask": bool(use_periodic_box_mask),
            "periodic_box_mask_radius": int(mask_radius),
            "field_initialization": "near_zero_random",
            "field_initialization_method": "init_variance_network",
            "initial_output_scale": float(initial_output_scale),
            "approx2_reference_field": False,
            "initialization_rng_seed": int(rng_seed),
            "selection_metric": selection_metric,
            "selected_epoch": selected_epoch,
            "selected_training_variance_ratio": training_variance_ratio,
        },
    )


def evaluate_variance_ratio(
    fields: torch.Tensor,
    links: torch.Tensor,
    split_mask: torch.Tensor,
    learned_field: torch.Tensor,
    *,
    beta: float,
    device: torch.device,
) -> float:
    """Evaluate one selected deformation against the full ensemble variance."""
    latsize = tuple(int(length) for length in fields.shape[1:-1])
    model = make_model(
        fields,
        links,
        n=N,
        action_type="gauge",
        split_mask=split_mask,
        device=device,
    )
    action = CPnGaugeAction(beta, N, latsize, device=device)
    observable = CPnOnePtFn(
        N,
        latsize,
        i=COLOR_INDICES[0],
        j=COLOR_INDICES[1],
        source=SOURCE,
        action=action,
        device=device,
    )
    loss = VarianceLoss(model, observable)
    with torch.no_grad():
        ratio = loss(
            alphas=learned_field.to(device), sampletype="all"
        ) / loss.initial_var
    return float(ratio.detach().cpu())


def save_selected_comparison_case(
    artifact: Path,
    learned_field: torch.Tensor,
    variance_ratio: float,
    *,
    selected_epoch: int,
    training_variance_ratio: float,
    selection_metric: str,
) -> None:
    """Update a saved run with its minimum-training-variance field."""
    saved = np.load(artifact, allow_pickle=True)
    if saved.shape != () or not isinstance(saved.item(), dict):
        raise ValueError(f"invalid training artifact: {artifact}")
    data = saved.item()
    field = learned_field.detach().cpu().numpy()
    data["learned_field"] = field
    alpha_scale = float(data.get("alpha_scale", 1.0))
    data["scaled_field"] = alpha_scale * field
    data["variance_ratio"] = float(variance_ratio)
    metadata = data.setdefault("metadata", {})
    metadata.update(
        field_at_source=float(field[*SOURCE, COLOR_INDICES[0]]),
        field_norm_at_source=float(np.linalg.norm(field[*SOURCE])),
        selection_metric=selection_metric,
        selected_epoch=int(selected_epoch),
        selected_training_variance_ratio=float(training_variance_ratio),
    )
    np.save(artifact, data, allow_pickle=True)


def plot_comparison(
    fields: dict[str, torch.Tensor],
    training_curves: dict[str, np.ndarray],
    beta: float,
    output: Path,
    cutoff_epochs: dict[str, int],
    *,
    mask_radius: int = 6,
) -> None:
    """Render separate masked and unmasked panels for the manuscript."""
    arrays = {name: field.detach().cpu().numpy()
              for name, field in fields.items()}
    if set(training_curves) != set(arrays):
        raise ValueError("each deformation field must have a training history")
    limits = {
        label: limit if (limit := float(np.abs(field).max())) > 0.0 else 1.0
        for label, field in arrays.items()
    }
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if any(len(curve) < 1 for curve in training_curves.values()):
        raise ValueError("training histories must contain at least one epoch")
    plotted_curves = {
        label: moving_median(np.asarray(curve, dtype=float))
        for label, curve in training_curves.items()
    }
    plotted_curves = {
        label: curve[: cutoff_epochs[label] + 1]
        for label, curve in plotted_curves.items()
    }
    finite_positive = np.concatenate(
        [curve[np.isfinite(curve) & (curve > 0)] for curve in plotted_curves.values()]
    )
    if finite_positive.size == 0:
        raise ValueError("training histories contain no positive finite values")
    curve_ymin = float(finite_positive.min()) * 0.75
    curve_ymax = float(finite_positive.max()) * 1.25

    # The training panel and each field map share one square axes size.  The
    # figure is slightly wider than the manuscript text block so the four
    # panels remain legible when included at 0.88\paperwidth in the draft.
    width = 1.16 * fig_width_pt * inches_per_pt
    height = 0.30 * width
    output_suffix = output.suffix or ".pdf"
    output_stem = output.stem if output.suffix else output.name
    output_paths = {
        MASKED_LABEL: output.with_name(
            f"{output_stem}_masked{output_suffix}"
        ),
        UNMASKED_LABEL: output.with_name(
            f"{output_stem}_not_masked{output_suffix}"
        ),
    }

    for label, field in arrays.items():
        fig = plt.figure(figsize=(width, height))
        grid_top = 0.84 if label == UNMASKED_LABEL else 0.98
        grid = fig.add_gridspec(
            nrows=1,
            ncols=N + 3,
            width_ratios=(1.0, 0.28, *([1.0] * (N + 1))),
            left=0.10,
            right=0.90,
            bottom=0.15,
            top=grid_top,
            wspace=0.04,
        )

        training_axis = fig.add_subplot(grid[0, 0])
        training_axis.set_box_aspect(1)
        curve = plotted_curves[label]
        training_axis.plot(
            np.arange(len(curve)),
            curve,
            color="black",
            linewidth=1.1,
        )
        training_axis.set_yscale("log")
        training_axis.set_ylim(curve_ymin, curve_ymax)
        training_axis.set_xlim(0, max(1, len(curve) - 1))
        panel_label = "(a)" if label == UNMASKED_LABEL else "(b)"
        panel_title = training_axis.set_title(
            rf"\textbf{{{panel_label} {label}}}"
        )
        panel_title.set_usetex(True)
        training_axis.set_ylabel(
            r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$",
            labelpad=2,
        )
        training_axis.set_xlabel("training epoch")
        training_axis.grid(True, which="both", linewidth=0.35, alpha=0.35)

        field_axes = []
        image = None
        half = LATTICE_SIZE // 2
        extent = (-half - 0.5, LATTICE_SIZE - half - 0.5) * 2
        for channel in range(N + 1):
            column = channel + 2
            if channel == 0:
                axis = fig.add_subplot(grid[0, column])
            else:
                axis = fig.add_subplot(
                    grid[0, column],
                    sharex=field_axes[0],
                    sharey=field_axes[0],
                )
            axis.set_box_aspect(1)
            field_axes.append(axis)
            centered_field = np.roll(
                field[..., channel], shift=(half, half), axis=(0, 1)
            )
            image = axis.imshow(
                centered_field,
                origin="lower",
                cmap="bwr",
                vmin=-limits[label],
                vmax=limits[label],
                extent=extent,
            )
            axis.set_title(rf"$F_{{x,{channel}}}$", pad=3)
            axis.set_xlabel(r"$x_0$")
            axis.set_xticks([-20, 0, 20])
            axis.set_yticks([-20, 0, 20])
            if label == MASKED_LABEL:
                axis.add_patch(
                    Rectangle(
                        (-mask_radius, -mask_radius),
                        2 * mask_radius,
                        2 * mask_radius,
                        fill=False,
                        edgecolor="black",
                        linestyle="--",
                        linewidth=1.0,
                        zorder=3,
                    )
                )
            if channel == 0:
                axis.set_ylabel(r"$x_1$")
            axis.tick_params(labelleft=(channel == 0))

        if label == UNMASKED_LABEL:
            suptitle = fig.suptitle(
                rf"$\mathbb{{CP}}^{{{N}}}$ gauge action, "
                rf"${LATTICE_SIZE}^{{{NDIM}}}$, $\beta={beta:g}$",
                y=0.96,
            )
            suptitle.set_usetex(True)
        # Match the colorbar to the active heatmap axes bounds so its ends
        # align with the map pixels even when the axes are constrained to be
        # square.  This axis sits outside the GridSpec so it cannot change the
        # plots' shared dimensions.
        fig.canvas.draw()
        map_position = field_axes[-1].get_position()
        colorbar_axis = fig.add_axes(
            [
                map_position.x1 + 0.02,
                map_position.y0,
                0.018,
                map_position.height,
            ]
        )
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.ax.tick_params(labelsize=7.5, pad=2)

        panel_output = output_paths[label]
        panel_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(panel_output)
        print(f"Saved {label} comparison plot to {panel_output}")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse only the training controls that affect this fixed comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "output basename; writes *_masked and *_not_masked PDFs "
            f"(default: {DEFAULT_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--masked-training-artifact",
        type=Path,
        default=None,
        help="comparison-specific masked result to load or create",
    )
    parser.add_argument(
        "--unmasked-training-artifact",
        type=Path,
        default=None,
        help="saved unmasked comparison result to load or create",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--initial-output-scale",
        type=float,
        default=1.0,
        help="scale only the near-zero U-Net output-layer initialization",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("constant", "cosine", "plateau"),
        default="cosine",
        help="learning-rate schedule for both training runs",
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
        default=True,
        help="stop both trainings after the validation objective plateaus",
    )
    parser.add_argument(
        "--mask-radius",
        type=int,
        default=6,
        help="periodic-mask radius if no saved masked result is found",
    )
    parser.add_argument("--unet-hidden-dim", type=int, default=None)
    parser.add_argument("--decorrelation", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be positive and --batch-size at least 2")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive and finite")
    if (
        not math.isfinite(args.initial_output_scale)
        or args.initial_output_scale <= 0
    ):
        parser.error("--initial-output-scale must be positive and finite")
    if not math.isfinite(args.min_learning_rate) or args.min_learning_rate < 0:
        parser.error("--min-learning-rate must be non-negative and finite")
    if args.lr_scheduler != "constant" and args.min_learning_rate > args.learning_rate:
        parser.error("--min-learning-rate cannot exceed --learning-rate")
    if (
        not math.isfinite(args.lr_factor)
        or not 0 < args.lr_factor < 1
        or args.lr_patience < 1
        or args.lr_cooldown < 0
    ):
        parser.error("invalid learning-rate scheduler settings")
    if args.max_training_time is not None and (
        not math.isfinite(
            args.max_training_time) or args.max_training_time <= 0
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
    if args.mask_radius < 0:
        parser.error("--mask-radius must be non-negative")
    if not math.isfinite(args.decorrelation) or args.decorrelation <= 0:
        parser.error("--decorrelation must be positive and finite")
    return args


def main() -> None:
    """Load or train both cases, then display their field comparison."""
    args = parse_args()
    beta = beta_from_filename(args.ensemble)
    masked_artifact = (
        args.masked_training_artifact or masked_comparison_savename(beta)
    )
    unmasked_artifact = (
        args.unmasked_training_artifact or unmasked_comparison_savename(beta)
    )
    print(f"Using beta={beta:g}, parsed from {args.ensemble.name}")
    cases = {
        MASKED_LABEL: {
            "artifact": masked_artifact,
            "use_mask": True,
            "selection_metric": "minimum_training_variance",
            "smooth_window": None,
            "loaded": None,
        },
        UNMASKED_LABEL: {
            "artifact": unmasked_artifact,
            "use_mask": False,
            "selection_metric": "minimum_25_epoch_rolling_median_training_variance",
            "smooth_window": 25,
            "loaded": None,
        },
    }
    for label, case in cases.items():
        artifact = case["artifact"]
        if not artifact.exists():
            print(f"No {label} result at {artifact}; training it")
            continue
        try:
            if case["use_mask"]:
                load_masked_case(artifact, beta=beta)
            else:
                load_unmasked_case(artifact, beta=beta)
            field, selected_epoch, training_ratio = load_training_minimum_field(
                artifact, smooth_window=case["smooth_window"]
            )
            case["loaded"] = {
                "field": field,
                "epoch": selected_epoch,
                "training_ratio": training_ratio,
                "curve": load_training_curve(artifact),
            }
            print(
                f"Loaded {label} result from {artifact}; selected training "
                f"minimum at epoch {selected_epoch}"
            )
        except ValueError as error:
            print(f"Saved {label} result is incompatible; retraining it ({error})")

    # The manuscript figure can be refreshed from its saved training histories
    # and selected fields even when the original gauge ensemble is unavailable.
    # Full-sample variance evaluation still requires that ensemble below.
    if not args.ensemble.exists():
        if any(case["loaded"] is None for case in cases.values()):
            raise FileNotFoundError(args.ensemble)
        plot_comparison(
            {label: case["loaded"]["field"] for label, case in cases.items()},
            {label: case["loaded"]["curve"] for label, case in cases.items()},
            beta,
            args.output,
            {label: case["loaded"]["epoch"] for label, case in cases.items()},
            mask_radius=args.mask_radius,
        )
        return

    device = resolve_device(args.device)
    set_seed(args.seed)
    fields, links, _ = load_ensemble(
        args.ensemble,
        n=N,
        ndim=NDIM,
        lattice_size=LATTICE_SIZE,
        beta=beta,
        action_type="gauge",
        i=COLOR_INDICES[0],
        j=COLOR_INDICES[1],
        decorrelation_factor=args.decorrelation,
    )
    if links is None:
        raise RuntimeError("the gauge-action comparison requires gauge links")
    fields, links = fields.to(device), links.to(device)
    split_mask = make_split_mask(fields)

    results: dict[str, torch.Tensor] = {}
    ratios: dict[str, float] = {}
    training_curves: dict[str, np.ndarray] = {}
    for label, case in cases.items():
        artifact = case["artifact"]
        use_mask = case["use_mask"]
        loaded = case["loaded"]
        if loaded is None:
            print(f"\nTraining {label} ...")
            trainer, field, _ = train_case(
                fields,
                links,
                split_mask,
                beta=beta,
                use_periodic_box_mask=use_mask,
                args=args,
                device=device,
            )
            if use_mask:
                selected_epoch = int(trainer.training_minimum_epoch)
            else:
                selected_epoch = minimum_training_epoch(
                    moving_median(np.asarray(trainer.train_var, dtype=float))
                )
            field = trainer.alpha_record[selected_epoch].detach()
            training_ratio = float(
                trainer.train_var[selected_epoch]
                / trainer.loss_fn.initial_var
            )
            curve = normalize_training_curve(
                np.asarray(trainer.train_var),
                float(trainer.loss_fn.initial_var),
                label=label,
            )
        else:
            field = loaded["field"]
            selected_epoch = loaded["epoch"]
            training_ratio = loaded["training_ratio"]
            curve = loaded["curve"]

        ratio = evaluate_variance_ratio(
            fields,
            links,
            split_mask,
            field,
            beta=beta,
            device=device,
        )
        if loaded is None:
            save_comparison_case(
                artifact,
                trainer,
                field,
                ratio,
                beta=beta,
                use_periodic_box_mask=use_mask,
                mask_radius=args.mask_radius,
                rng_seed=args.seed,
                initial_output_scale=args.initial_output_scale,
                selected_epoch=selected_epoch,
                training_variance_ratio=training_ratio,
                selection_metric=case["selection_metric"],
            )
        else:
            save_selected_comparison_case(
                artifact,
                field,
                ratio,
                selected_epoch=selected_epoch,
                training_variance_ratio=training_ratio,
                selection_metric=case["selection_metric"],
            )
        print(
            f"{label}: selected epoch {selected_epoch}, training variance "
            f"ratio {training_ratio:.6g}, full-sample ratio {ratio:.6g}"
        )
        results[label] = field
        ratios[label] = ratio
        training_curves[label] = curve
        case["selected_epoch"] = selected_epoch

    plot_comparison(
        results,
        training_curves,
        beta,
        args.output,
        {label: case["selected_epoch"] for label, case in cases.items()},
        mask_radius=args.mask_radius,
    )


if __name__ == "__main__":
    main()
