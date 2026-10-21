"""Compare masked and unmasked CP(2) one-point-function deformations.

The masked field is loaded from the matching CP(N) variance-training artifact
created by ``cpn_OPF_analysis.py`` when available. Otherwise, both masked and
unmasked fields are trained here.

Run, for example:

    python cpn_periodic_mask_comparison.py --epochs 200 --mask-radius 6
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from plot_settings import fig_width_pt, inches_per_pt, latex_font_params

from actions import CPnGaugeAction
from cpn_opf.loading import load_ensemble
from cpn_opf.objectives import init_variance_network
from cpn_opf.persistence import load_deformation_field, training_savename
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
DEFAULT_OUTPUT = Path("paper/figs/fig1_locality_cpn.pdf")
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


def load_masked_case(
    artifact: Path,
    *,
    beta: float,
) -> tuple[torch.Tensor, float]:
    """Load and validate the masked variance-training result from CP(N) OPF."""
    if not artifact.exists():
        raise FileNotFoundError(
            "no masked CP(N) variance-training artifact at "
            f"{artifact}; run cpn_OPF_analysis.py first or provide "
            "--masked-training-artifact"
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
    return torch.as_tensor(field, dtype=torch.float32), variance_ratio


def load_unmasked_case(
    artifact: Path,
    *,
    beta: float,
) -> tuple[torch.Tensor, float]:
    """Load a previously saved unmasked comparison result."""
    saved = np.load(artifact, allow_pickle=True)
    if saved.shape != () or not isinstance(saved.item(), dict):
        raise ValueError(f"invalid unmasked training artifact: {artifact}")
    metadata = saved.item().get("metadata", {})
    if metadata.get("periodic_box_mask") is not False:
        raise ValueError(f"{artifact} is not a saved unmasked comparison")
    return load_masked_case(artifact, beta=beta)


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
    best_field = trainer.best_alphas().detach()
    variance_ratio = float(
        (loss(alphas=best_field, sampletype="all") / loss.initial_var)
        .detach()
        .cpu()
    )
    return trainer, best_field, variance_ratio


def save_unmasked_case(
    artifact: Path,
    trainer: Trainer,
    field: torch.Tensor,
    variance_ratio: float,
    *,
    beta: float,
) -> None:
    """Persist the full unmasked training result for later reuse."""
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
    )
    saved = np.load(artifact, allow_pickle=True).item()
    saved["metadata"]["periodic_box_mask"] = False
    np.save(artifact, saved, allow_pickle=True)


def plot_comparison(
    fields: dict[str, torch.Tensor],
    training_curves: dict[str, np.ndarray],
    beta: float,
    output: Path,
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
    epoch_count = min(len(curve) for curve in training_curves.values())
    if epoch_count < 1:
        raise ValueError("training histories must contain at least one epoch")
    plotted_curves = {
        label: moving_median(np.asarray(curve[:epoch_count], dtype=float))
        for label, curve in training_curves.items()
    }
    unmasked_curve = plotted_curves.get(UNMASKED_LABEL)
    if unmasked_curve is not None:
        finite_epochs = np.flatnonzero(
            np.isfinite(unmasked_curve) & (unmasked_curve > 0)
        )
        if finite_epochs.size == 0:
            raise ValueError("unmasked history has no positive finite values")
        minimum_epoch = int(
            finite_epochs[np.argmin(unmasked_curve[finite_epochs])]
        )
        unmasked_curve[minimum_epoch + 1 :] = np.nan
        print(
            "Truncated the unmasked training curve at its minimum moving "
            f"median (epoch {minimum_epoch})"
        )
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
        training_axis.plot(
            np.arange(epoch_count),
            plotted_curves[label],
            color="C0",
            linewidth=1.1,
        )
        training_axis.set_yscale("log")
        training_axis.set_ylim(curve_ymin, curve_ymax)
        training_axis.set_xlim(0, epoch_count - 1)
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
            image = axis.imshow(
                field[..., channel],
                origin="lower",
                cmap="coolwarm",
                vmin=-limits[label],
                vmax=limits[label],
            )
            axis.set_title(rf"$F_{{x,{channel}}}$", pad=3)
            axis.set_xlabel(r"$x_0$")
            if channel == 0:
                axis.set_ylabel(r"$x_1$")
            axis.tick_params(labelleft=(channel == 0))

        if label == UNMASKED_LABEL:
            suptitle = fig.suptitle(
                rf"$\mathbb{{CP}}^{{{N}}}$ gauge action, $d={NDIM}$, "
                rf"$L={LATTICE_SIZE}$, $\beta={beta:g}$",
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
        help="saved masked variance result from cpn_OPF_analysis.py",
    )
    parser.add_argument(
        "--unmasked-training-artifact",
        type=Path,
        default=None,
        help="saved unmasked comparison result to load or create",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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
    if not args.ensemble.exists():
        raise FileNotFoundError(args.ensemble)
    beta = beta_from_filename(args.ensemble)
    masked_artifact = args.masked_training_artifact or training_savename(
        N,
        NDIM,
        beta,
        LATTICE_SIZE,
        COLOR_INDICES[0],
        COLOR_INDICES[1],
        loss_function="variance",
        action_type="gauge",
    )
    unmasked_artifact = args.unmasked_training_artifact or training_savename(
        N,
        NDIM,
        beta,
        LATTICE_SIZE,
        COLOR_INDICES[0],
        COLOR_INDICES[1],
        loss_function="variance-unmasked",
        action_type="gauge",
    )
    print(f"Using beta={beta:g}, parsed from {args.ensemble.name}")
    masked_result: tuple[torch.Tensor, float] | None = None
    if masked_artifact.exists():
        try:
            masked_result = load_masked_case(masked_artifact, beta=beta)
            print(f"Loaded masked field from {masked_artifact}")
        except ValueError as error:
            print(
                "Saved masked result is incompatible; retraining it "
                f"({error})"
            )
    else:
        print(f"No masked result at {masked_artifact}; training it")

    unmasked_result: tuple[torch.Tensor, float] | None = None
    if unmasked_artifact.exists():
        try:
            unmasked_result = load_unmasked_case(unmasked_artifact, beta=beta)
            print(f"Loaded unmasked field from {unmasked_artifact}")
        except ValueError as error:
            print(
                "Saved unmasked result is incompatible; retraining it "
                f"({error})"
            )
    else:
        print(f"No unmasked result at {unmasked_artifact}; training it")

    fields: torch.Tensor | None = None
    links: torch.Tensor | None = None
    split_mask: torch.Tensor | None = None
    device: torch.device | None = None
    if masked_result is None or unmasked_result is None:
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
            raise RuntimeError(
                "the gauge-action comparison requires gauge links")
        fields, links = fields.to(device), links.to(device)
        split_mask = make_split_mask(fields)

    masked_label = MASKED_LABEL
    unmasked_label = UNMASKED_LABEL
    results: dict[str, torch.Tensor] = {}
    ratios: dict[str, float] = {}
    training_curves: dict[str, np.ndarray] = {}
    if masked_result is None:
        print(f"\nTraining {masked_label} ...")
        assert fields is not None and links is not None and split_mask is not None
        assert device is not None
        trainer, results[masked_label], ratios[masked_label] = train_case(
            fields,
            links,
            split_mask,
            beta=beta,
            use_periodic_box_mask=True,
            args=args,
            device=device,
        )
        training_curves[masked_label] = normalize_training_curve(
            np.asarray(trainer.train_var),
            float(trainer.loss_fn.initial_var),
            label=masked_label,
        )
    else:
        results[masked_label], ratios[masked_label] = masked_result
        training_curves[masked_label] = load_training_curve(masked_artifact)
    if unmasked_result is None:
        print(f"\nTraining {unmasked_label} ...")
        assert fields is not None and links is not None and split_mask is not None
        assert device is not None
        trainer, field, variance_ratio = train_case(
            fields,
            links,
            split_mask,
            beta=beta,
            use_periodic_box_mask=False,
            args=args,
            device=device,
        )
        save_unmasked_case(
            unmasked_artifact,
            trainer,
            field,
            variance_ratio,
            beta=beta,
        )
        training_curves[unmasked_label] = normalize_training_curve(
            np.asarray(trainer.train_var),
            float(trainer.loss_fn.initial_var),
            label=unmasked_label,
        )
        results[unmasked_label] = field
        ratios[unmasked_label] = variance_ratio
        print(f"Saved unmasked field to {unmasked_artifact}")
    else:
        results[unmasked_label], ratios[unmasked_label] = unmasked_result
        training_curves[unmasked_label] = load_training_curve(unmasked_artifact)
    print(f"{masked_label}: Var[Q]/Var[O] = {ratios[masked_label]:.6g}")
    print(f"{unmasked_label}: Var[Q]/Var[O] = {ratios[unmasked_label]:.6g}")

    plot_comparison(results, training_curves, beta, args.output)


if __name__ == "__main__":
    main()
