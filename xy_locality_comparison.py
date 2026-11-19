"""Train (or load) the masked and unmasked XY one-point deformations and
render the combined locality figure for the paper.

The masked case reuses the saved direct-variance-minimization result from
``xy_OPF_analysis.py``.  The unmasked case has no saved artifact (variance
training without a support mask has no obstruction to a global, unbounded
shift and is not part of the standard pipeline), so it is trained here and
cached under ``training_data/`` for reuse.

Run, for example:

    python xy_locality_comparison.py --epochs 4000
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import tempfile

cache_dir = Path(tempfile.gettempdir()) / "lattice-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lattice-mplconfig")
)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import XY_utils as xy
from actions import XYAction
from deformations import ThetaDefs
from losses import VarianceLoss
from networks import UNet
from observables import XYOnePtFn
from plot_settings import fig_width_pt, inches_per_pt, latex_font_params
from training import Trainer, init_weights
from xy_opf.training import make_split_mask, save_training_result

NDIM = 2
LATTICE_SIZE = 32
SOURCE = (0, 0)
SINK_SEPARATION = 10
MASK_RADIUS = 6
BETA = 0.65
DEFAULT_ENSEMBLE = Path(f"data/xy_dim{NDIM}_b{BETA:.2f}_L{LATTICE_SIZE}_ens.npy")
MASKED_ARTIFACT = Path(
    f"training_data/xy_scaling_dim{NDIM}_b{BETA:.2f}_loss-variance_training.npy"
)
UNMASKED_ARTIFACT = Path(
    f"training_data/xy_scaling_dim{NDIM}_b{BETA:.2f}_"
    "loss-variance-unmasked_training.npy"
)
TWO_POINT_ARTIFACT = Path(
    f"training_data/xy_two_point/xy_twopt_dim{NDIM}_b{BETA:.2f}_"
    f"L{LATTICE_SIZE}_t{SINK_SEPARATION:02d}_src0-0_loss-variance_training.npy"
)
DEFAULT_OUTPUT_ONEPT = Path("paper/figs/xy_opf_locality_onept.pdf")
DEFAULT_OUTPUT_TWOPT = Path("paper/figs/xy_opf_locality_twopt.pdf")


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
    """Reset every torch RNG used for a reproducible training run."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_result(artifact: Path) -> dict:
    """Load and sanity-check a saved training-result dictionary."""
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    saved = np.load(artifact, allow_pickle=True)
    if saved.shape != () or not isinstance(saved.item(), dict):
        raise ValueError(f"invalid training artifact: {artifact}")
    data = saved.item()
    metadata = data.get("metadata", {})
    expected = {
        "ndim": NDIM,
        "beta": BETA,
        "lattice_size": (LATTICE_SIZE, LATTICE_SIZE),
        "source": SOURCE,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{artifact} has {key}={metadata.get(key)!r}; expected {value!r}"
            )
    return data


def normalize_curve(data: dict) -> np.ndarray:
    """Return the training-variance history normalized by its initial value."""
    curve = np.asarray(data["train_var"], dtype=float)
    initial_var = float(data["initial_var"])
    if not math.isfinite(initial_var) or initial_var <= 0:
        raise ValueError("invalid initial variance in training artifact")
    return curve / initial_var


def moving_median(values: np.ndarray, window: int = 25) -> np.ndarray:
    """Reduce minibatch noise with a centered rolling median."""
    curve = np.asarray(values, dtype=float)
    radius = window // 2
    smoothed = np.full(curve.shape, np.nan, dtype=float)
    for index in range(len(curve)):
        if not np.isfinite(curve[index]):
            continue
        start = max(0, index - radius)
        stop = min(len(curve), index + radius + 1)
        window_values = curve[start:stop]
        window_values = window_values[np.isfinite(window_values)]
        if window_values.size:
            smoothed[index] = np.median(window_values)
    return smoothed


def train_unmasked(
    fields: torch.Tensor,
    split_mask: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Trainer, torch.Tensor, float]:
    """Train the XY one-point deformation with no periodic support mask."""
    set_seed(args.seed)
    latsize = fields.shape[1:]
    model = ThetaDefs(ndims=NDIM)
    model.samples = {
        "all": fields,
        "train": fields[split_mask],
        "test": fields[~split_mask],
    }
    action = XYAction(BETA, NDIM)
    observable = XYOnePtFn(latsize, SOURCE, action)
    loss_fn = VarianceLoss(model, observable)

    alphanet = UNet(
        input_dim=observable.grid.shape[-1],
        output_dim=1,
        ndim=NDIM,
        hd=args.unet_hidden_dim,
    ).to(device)
    alphanet.apply(init_weights)
    nn.init.zeros_(alphanet.output.weight)
    nn.init.zeros_(alphanet.output.bias)

    trainer = Trainer(
        model,
        loss_fn,
        optim.Adam(alphanet.parameters(), lr=args.learning_rate),
        args.epochs,
        args.batch_size,
    )
    trainer.train(
        alphanet,
        mask=None,
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
    best_field = trainer.best_alphas(args.best_window).detach()
    variance_ratio = float(
        (loss_fn(alphas=best_field, sampletype="all") / loss_fn.initial_var)
        .detach()
        .cpu()
    )
    return trainer, best_field, variance_ratio


def truncate_before_instability(curve: np.ndarray, label: str) -> np.ndarray:
    """NaN-out everything after a curve's smoothed minimum.

    Unmasked variance training has an unbounded, zero-gradient-penalty
    direction (a field that is constant away from the lattice boundary
    leaves the nearest-neighbour action unchanged while shrinking the
    observable); continuing to train past the point where this is resolved
    to machine precision only amplifies numerical noise.  Truncating at the
    smoothed minimum keeps the plotted curve honest about where training
    actually stopped being informative.
    """
    truncated = curve.copy()
    finite = np.flatnonzero(np.isfinite(truncated) & (truncated > 0))
    if finite.size == 0:
        raise ValueError(f"{label} has no usable positive, finite values")
    minimum_epoch = int(finite[np.argmin(truncated[finite])])
    truncated[minimum_epoch + 1 :] = np.nan
    print(f"Truncated {label} at its smoothed minimum (epoch {minimum_epoch})")
    return truncated


def centered_extent(latsize: tuple[int, int]) -> tuple[np.ndarray, list[float]]:
    """Return the roll shift and imshow extent centering (0, 0)."""
    half = [length // 2 for length in latsize]
    extent = [-half[1] - 0.5, latsize[1] - half[1] - 0.5,
              -half[0] - 0.5, latsize[0] - half[0] - 0.5]
    return np.array(half), extent


def plot_heatmap(
    axis: plt.Axes,
    field: np.ndarray,
    *,
    vlim: float,
    title: str,
    xlabel: bool,
    ylabel: bool,
    anchor: str | None = None,
    x_shift: int = 0,
) -> object:
    """Draw a periodic deformation field, optionally shifting its x origin."""
    latsize = field.shape
    half, extent = centered_extent(latsize)
    centered = np.roll(field, shift=(half[0], half[1] + x_shift), axis=(0, 1))
    image = axis.imshow(
        centered,
        origin="lower",
        cmap="bwr",
        vmin=-vlim,
        vmax=vlim,
        extent=extent,
    )
    axis.set_box_aspect(1)
    if anchor is not None:
        axis.set_anchor(anchor)
    axis.set_title(title, pad=3)
    if xlabel:
        axis.set_xlabel(r"$x_0$")
    else:
        # Fully remove the tick marks (not just their labels) so a stacked
        # panel above another heatmap never draws into the title below it.
        axis.tick_params(axis="x", which="both", bottom=False, top=False,
                          labelbottom=False)
    if ylabel:
        axis.set_ylabel(r"$x_1$", labelpad=2)
    axis.tick_params(labelleft=ylabel)
    return image


def add_mask_box(
    axis: plt.Axes,
    center: tuple[float, float],
    *,
    radius: int,
    linewidth: float,
) -> None:
    """Outline a square support region centered on an operator insertion."""
    x0, x1 = center
    axis.add_patch(
        Rectangle(
            (x0 - radius, x1 - radius),
            2 * radius,
            2 * radius,
            fill=False,
            edgecolor="black",
            linestyle="--",
            linewidth=linewidth,
            zorder=3,
        )
    )


def add_colorbar(
    fig: plt.Figure,
    axis: plt.Axes,
    image: object,
    *,
    fig_width: float,
    gap_in: float,
    width_in: float,
    tick_labelsize: float,
) -> plt.Axes:
    """Attach a colorbar to ``axis``'s right edge at an explicit inch size.

    Using an explicit axis instead of ``make_axes_locatable`` keeps the
    heatmap's own size exactly whatever it was already placed at: the
    divider-based colorbar shrinks its parent axes to make room for itself,
    which fights any attempt to size heatmaps precisely.
    """
    position = axis.get_position()
    cax = fig.add_axes(
        [
            position.x1 + gap_in / fig_width,
            position.y0,
            width_in / fig_width,
            position.height,
        ]
    )
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.ax.tick_params(labelsize=tick_labelsize, pad=1.5)
    return cax


def save_tight_fixed_height(
    fig: plt.Figure,
    path: Path,
    *,
    fig_height: float,
    pad_inches: float = 0.02,
) -> None:
    """Save ``fig`` cropped tightly in width but at a fixed, shared height.

    Two figures saved this way end up with identical page heights (so they
    line up as matched-height LaTeX subfigures) while each is cropped to use
    no more horizontal space than its own content actually needs, instead of
    whatever dead margin its GridSpec happened to leave.
    """
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    cropped = Bbox.from_extents(tight.x0, 0.0, tight.x1, fig_height)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches=cropped, pad_inches=pad_inches)
    saved_width = cropped.width + 2 * pad_inches
    saved_height = cropped.height + 2 * pad_inches
    print(f"  {path}: saved size {saved_width:.3f}in x {saved_height:.3f}in")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument("--masked-artifact", type=Path, default=MASKED_ARTIFACT)
    parser.add_argument("--unmasked-artifact", type=Path, default=UNMASKED_ARTIFACT)
    parser.add_argument("--two-point-artifact", type=Path, default=TWO_POINT_ARTIFACT)
    parser.add_argument("--output-onept", type=Path, default=DEFAULT_OUTPUT_ONEPT)
    parser.add_argument("--output-twopt", type=Path, default=DEFAULT_OUTPUT_TWOPT)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lr-scheduler", choices=("constant", "cosine", "plateau"), default="plateau"
    )
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-cooldown", type=int, default=1)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-training-time", type=float, default=None)
    parser.add_argument("--plateau-window", type=int, default=20)
    parser.add_argument("--plateau-patience", type=int, default=400)
    parser.add_argument("--plateau-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--plateau-threshold-mode", choices=("abs", "rel"), default="rel"
    )
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="unmasked training has no plateau to stop at before instability",
    )
    parser.add_argument("--best-window", type=int, default=20)
    parser.add_argument(
        "--plot-epochs",
        type=int,
        default=None,
        help="maximum training epoch shown (default: full saved history)",
    )
    parser.add_argument(
        "--floor-decades",
        type=float,
        default=3.0,
        help="decades below the masked plateau where the shared y-axis is cut",
    )
    parser.add_argument("--unet-hidden-dim", type=int, default=None)
    parser.add_argument("--decorrelation", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--retrain-unmasked", action="store_true")
    args = parser.parse_args()

    masked_data = load_result(args.masked_artifact)
    masked_curve = normalize_curve(masked_data)
    masked_field = np.asarray(masked_data["scaled_field"])[..., 0]
    masked_ratio = float(masked_data["variance_ratio"])

    if args.unmasked_artifact.exists() and not args.retrain_unmasked:
        unmasked_data = load_result(args.unmasked_artifact)
        print(f"Loaded unmasked field from {args.unmasked_artifact}")
    else:
        device = resolve_device(args.device)
        set_seed(args.seed)
        fields, _ = xy.load_ensemble(
            str(args.ensemble), decorrelation_factor=args.decorrelation
        )
        fields = fields.to(device)
        split_mask = make_split_mask(fields)
        print("Training unmasked XY one-point deformation ...")
        trainer, field, variance_ratio = train_unmasked(
            fields, split_mask, args=args, device=device
        )
        save_training_result(
            args.unmasked_artifact,
            trainer,
            loss_function="variance",
            ndim=NDIM,
            beta=BETA,
            latsize=(LATTICE_SIZE, LATTICE_SIZE),
            source=SOURCE,
            learned_field=field,
            alpha_scale=1.0,
            variance_ratio=variance_ratio,
            extra_metadata={"periodic_box_mask": False},
        )
        print(f"Saved unmasked result to {args.unmasked_artifact}")
        unmasked_data = load_result(args.unmasked_artifact)

    unmasked_curve = normalize_curve(unmasked_data)
    unmasked_field = np.asarray(unmasked_data["scaled_field"])[..., 0]
    unmasked_ratio_raw = float(unmasked_data["variance_ratio"])

    two_point_data = load_result(args.two_point_artifact)
    glued_field = np.asarray(two_point_data["glued_field"])[..., 0]
    trained_field = np.asarray(two_point_data["scaled_field"])[..., 0]
    metrics = two_point_data["metadata"]["metrics"]

    epoch_count = min(len(masked_curve), len(unmasked_curve))
    if args.plot_epochs is not None:
        epoch_count = min(epoch_count, args.plot_epochs)
    masked_smoothed = moving_median(masked_curve[:epoch_count])
    unmasked_smoothed = moving_median(unmasked_curve[:epoch_count])
    unmasked_smoothed = truncate_before_instability(unmasked_smoothed, "unmasked")
    unmasked_ratio = float(np.nanmin(unmasked_smoothed))
    print(f"masked: Var[Q]/Var[O] -> {masked_ratio:.4g} (plateau)")
    print(
        f"unmasked: Var[Q]/Var[O] -> {unmasked_ratio:.4g} at truncation "
        f"(best_alphas ratio {unmasked_ratio_raw:.4g})"
    )

    # These figures are included in the paper at well under their native
    # size (e.g. at 0.42\linewidth against a ~4.9in native width), so a 9pt
    # matplotlib font prints far smaller than the caption's own \small (9pt
    # at the document's 10pt base).  FONT_SCALE is the inverse of that
    # shrink factor (target LaTeX width / native pdf width): multiplying
    # every font size by it, so the *native* font is bigger than 9pt, makes
    # the *printed* font come out at roughly 9pt once LaTeX scales the whole
    # figure down.  fig_height is deliberately left unscaled -- only the
    # margins below grow with FONT_SCALE, so the bigger text eats a larger
    # share of the same fixed vertical budget, which is exactly the point.
    linewidth_in = fig_width_pt * inches_per_pt
    onept_frac = 0.42
    approx_native_width = 4.9
    latex_shrink = onept_frac * linewidth_in / approx_native_width
    FONT_SCALE = 1.0 / latex_shrink

    plt.rcParams.update(latex_font_params)
    plt.rcParams.update(
        {
            "font.size": 9 * FONT_SCALE,
            "axes.labelsize": 9 * FONT_SCALE,
            "axes.titlesize": 9 * FONT_SCALE,
            "xtick.labelsize": 8 * FONT_SCALE,
            "ytick.labelsize": 8 * FONT_SCALE,
            "legend.fontsize": 9 * FONT_SCALE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7 * FONT_SCALE,
        }
    )
    cbar_tick_labelsize = 8 * FONT_SCALE
    cbar_label_fontsize = 9 * FONT_SCALE

    # Both figures share the same overall height (margins included) so that,
    # placed side by side in LaTeX as two subfigures of appropriately chosen
    # widths, their content rows line up.  All panel axes below are placed
    # with explicit, inches-based rectangles (rather than left to GridSpec's
    # width_ratios plus set_box_aspect) so that the training panel and the
    # two-point heatmaps come out at the exact same square size, and the
    # stacked one-point heatmaps use whatever height is left over.  Every
    # inches-based margin/gap constant here is scaled by FONT_SCALE along
    # with the fonts, so spacing stays proportional to the (now larger) text.
    fig_height = 3.45
    title_h = 0.28 * FONT_SCALE    # vertical room for a panel title
    xlabel_h = 0.46 * FONT_SCALE   # vertical room for x ticks + an x-label
    top_pad = 0.05 * FONT_SCALE
    bottom_pad = 0.06 * FONT_SCALE
    gap_bc = 0.015 * FONT_SCALE    # vertical gap between stacked (b) and (c)

    # Side length shared by the training panel and each two-point heatmap:
    # the full figure height, minus one title and one x-axis label/tick row.
    square_size = fig_height - top_pad - title_h - xlabel_h - bottom_pad
    # Side length of each stacked one-point heatmap: half of whatever is left
    # once two titles, one x-axis label/tick row, and the inter-panel gap
    # are taken out of the same total height.
    stacked_size = (
        fig_height - top_pad - bottom_pad - 2 * title_h - xlabel_h - gap_bc
    ) / 2

    cbar_gap_in = 0.12 * FONT_SCALE
    cbar_width_in = 0.11 * FONT_SCALE

    # --- Training history + stacked one-point heatmaps --------------------
    # The raw canvas is made generously wide so nothing clips against its
    # hard edge; ``save_tight_fixed_height`` below then crops away whatever
    # dead margin is left, so only the actual content sets the final width.
    onept_width = 9.0
    onept_figure = plt.figure(figsize=(onept_width, fig_height))

    curve_left_in = 0.68 * FONT_SCALE
    heatmap_left_in = curve_left_in + square_size + 0.58 * FONT_SCALE

    def axes_rect(fig_width_in, left_in, bottom_in, side_in):
        return [
            left_in / fig_width_in,
            bottom_in / fig_height,
            side_in / fig_width_in,
            side_in / fig_height,
        ]

    # (a) combined training curve, square and the same size as a two-point
    # heatmap panel in the two-point plot, top-aligned with panel (b).
    curve_bottom_in = fig_height - top_pad - title_h - square_size
    curve_axis = onept_figure.add_axes(
        axes_rect(onept_width, curve_left_in, curve_bottom_in, square_size)
    )
    epochs = np.arange(epoch_count)
    curve_axis.plot(
        epochs, unmasked_smoothed, color="tab:red", linewidth=1.2,
        label="unmasked",
    )
    curve_axis.plot(
        epochs, masked_smoothed, color="tab:blue", linewidth=1.2,
        label="masked",
    )
    curve_axis.set_yscale("log")
    curve_axis.set_xscale("log")
    finite_positive = np.concatenate(
        [
            masked_smoothed[np.isfinite(masked_smoothed) & (masked_smoothed > 0)],
            unmasked_smoothed[
                np.isfinite(unmasked_smoothed) & (unmasked_smoothed > 0)
            ],
        ]
    )
    # The unmasked curve collapses many decades below the masked plateau as
    # it rides the unbounded, zero-penalty constant-shift direction (see
    # ``truncate_before_instability``); showing that full range would
    # compress the masked plateau to an imperceptible sliver, so the visible
    # floor is set a fixed number of decades below the masked plateau and the
    # unmasked curve is simply allowed to run off the bottom of the frame.
    floor = masked_ratio * 10**(-args.floor_decades)
    curve_axis.set_ylim(floor, float(finite_positive.max()) * 1.5)
    curve_axis.set_xlim(1, epoch_count - 1)
    curve_axis.set_xlabel("training epoch")
    curve_axis.set_ylabel(
        r"$\operatorname{Var}[\mathcal{Q}]/\operatorname{Var}[\mathcal{O}]$",
        labelpad=2,
    )
    curve_axis.set_title("(a) training history", pad=3)
    curve_axis.grid(True, which="both", linewidth=0.35, alpha=0.35)
    curve_axis.legend(
        loc="lower left",
        ncol=1,
        frameon=True,
        framealpha=0.85,
        handlelength=0.9,
        handletextpad=0.4,
        borderpad=0.25,
        borderaxespad=0.3,
        fontsize=6 * FONT_SCALE,
    )

    # (b)-(c) one-point deformation heatmaps, stacked with a shared x-axis.
    # Each is as large as it can be while fitting, with panel (c)'s, in the
    # same total height as the training/two-point panels above.
    c_bottom_in = bottom_pad + xlabel_h
    b_bottom_in = c_bottom_in + stacked_size + title_h + gap_bc
    axis_b = onept_figure.add_axes(
        axes_rect(onept_width, heatmap_left_in, b_bottom_in, stacked_size)
    )
    image_b = plot_heatmap(
        axis_b, unmasked_field, vlim=float(np.abs(unmasked_field).max()),
        title="(b) unmasked $F_x$", xlabel=False, ylabel=True,
    )
    add_colorbar(
        onept_figure, axis_b, image_b, fig_width=onept_width, gap_in=cbar_gap_in,
        width_in=cbar_width_in, tick_labelsize=cbar_tick_labelsize,
    )
    axis_c = onept_figure.add_axes(
        axes_rect(onept_width, heatmap_left_in, c_bottom_in, stacked_size),
        sharex=axis_b, sharey=axis_b,
    )
    image_c = plot_heatmap(
        axis_c, masked_field, vlim=float(np.abs(masked_field).max()),
        title="(c) masked $F_x$", xlabel=True, ylabel=True,
    )
    add_mask_box(
        axis_c, SOURCE, radius=MASK_RADIUS, linewidth=0.8 * FONT_SCALE
    )
    # These panels are small enough, relative to the caption-matched font
    # size, that the usual three x tick labels collide; two comfortably
    # fit instead (the shared x-axis means this also fixes panel (b)).
    axis_c.set_xticks([-10, 10])
    add_colorbar(
        onept_figure, axis_c, image_c, fig_width=onept_width, gap_in=cbar_gap_in,
        width_in=cbar_width_in, tick_labelsize=cbar_tick_labelsize,
    )

    save_tight_fixed_height(onept_figure, args.output_onept, fig_height=fig_height)
    print(f"Saved one-point locality figure to {args.output_onept}")
    plt.close(onept_figure)

    # --- Glued vs. directly-trained two-point heatmaps --------------------
    two_point_limit = max(
        float(np.abs(glued_field).max()), float(np.abs(trained_field).max())
    )
    twopt_width = 11.0
    twopt_figure = plt.figure(figsize=(twopt_width, fig_height))

    d_left_in = 0.60 * FONT_SCALE
    e_left_in = d_left_in + square_size + 0.35 * FONT_SCALE
    d_bottom_in = bottom_pad + xlabel_h

    def twopt_axes_rect(left_in, bottom_in, side_in):
        return [
            left_in / twopt_width,
            bottom_in / fig_height,
            side_in / twopt_width,
            side_in / fig_height,
        ]

    axis_d = twopt_figure.add_axes(twopt_axes_rect(d_left_in, d_bottom_in, square_size))
    image_d = plot_heatmap(
        axis_d, glued_field, vlim=two_point_limit,
        title=r"(d) glued, $\Delta t=10$", xlabel=True, ylabel=True,
        x_shift=-(SINK_SEPARATION // 2),
    )
    midpoint = SINK_SEPARATION / 2
    for x0 in (-midpoint, midpoint):
        add_mask_box(
            axis_d, (x0, SOURCE[1]), radius=MASK_RADIUS,
            linewidth=0.8 * FONT_SCALE,
        )
    axis_e = twopt_figure.add_axes(
        twopt_axes_rect(e_left_in, d_bottom_in, square_size),
        sharex=axis_d, sharey=axis_d,
    )
    plot_heatmap(
        axis_e, trained_field, vlim=two_point_limit,
        title=r"(e) trained, $\Delta t=10$", xlabel=True, ylabel=False,
        x_shift=-(SINK_SEPARATION // 2),
    )

    cax_de = add_colorbar(
        twopt_figure, axis_e, image_d, fig_width=twopt_width, gap_in=cbar_gap_in,
        width_in=cbar_width_in, tick_labelsize=cbar_tick_labelsize,
    )
    cax_de.set_ylabel(r"$F_x$", labelpad=2, fontsize=cbar_label_fontsize)

    save_tight_fixed_height(twopt_figure, args.output_twopt, fig_height=fig_height)
    print(f"Saved two-point locality figure to {args.output_twopt}")
    plt.close(twopt_figure)

    print(
        "two-point variance ratios: glued="
        f"{metrics['glued_variance_ratio']:.4g}, trained="
        f"{metrics['variance_ratio']:.4g}"
    )


if __name__ == "__main__":
    main()
