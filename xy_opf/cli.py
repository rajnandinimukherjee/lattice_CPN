"""Command-line interface for the XY OPF scaling study."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import LOSS_FUNCTION_CHOICES
from .persistence import beta_key


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without executing the study."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[],
        help=(
            "beta values to use for --run, e.g. 1.2 1.3; "
            "omit to discover matching ensemble files"
        ),
    )
    parser.add_argument(
        "--delete-betas",
        nargs="+",
        type=float,
        default=[],
        metavar="BETA",
        help=(
            "delete only these saved beta artifacts and variance-ratio "
            "entries, then exit"
        ),
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        required=True,
        help="dimensions, e.g. 1 2 3",
    )
    parser.add_argument(
        "--Ls",
        nargs="+",
        default=[],
        help=(
            "lattice sizes in each dimension, e.g. 64 64 32; not needed "
            "with --delete-betas"
        ),
    )
    parser.add_argument(
        "--mask-radius",
        type=int,
        help="OPF localisation mask radius",
        default=6,
    )
    parser.add_argument("--epochs", help="epochs", type=int, default=200)
    parser.add_argument(
        "--batch-size", help="batch size", type=int, default=64
    )
    parser.add_argument(
        "--learning-rate",
        help="initial learning rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("constant", "cosine", "plateau"),
        default="plateau",
        help="learning-rate schedule",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="multiplicative plateau-scheduler reduction",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help=(
            "non-improving validation blocks before reducing the learning "
            "rate"
        ),
    )
    parser.add_argument(
        "--lr-cooldown",
        type=int,
        default=1,
        help="validation blocks to wait after reducing the learning rate",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-5,
        help="minimum learning rate for plateau or cosine scheduling",
    )
    parser.add_argument(
        "--max-training-time",
        type=float,
        default=None,
        help="maximum training time per beta/dimension in seconds",
    )
    parser.add_argument(
        "--plateau-window",
        type=int,
        default=20,
        help="optimizer steps per non-overlapping validation block",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=40,
        help="number of non-improving validation blocks before stopping",
    )
    parser.add_argument(
        "--plateau-min-delta",
        type=float,
        default=1e-3,
        help="minimum validation-objective improvement",
    )
    parser.add_argument(
        "--plateau-threshold-mode",
        choices=("abs", "rel"),
        default="rel",
        help="interpret --plateau-min-delta as absolute or relative",
    )
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop when validation exceeds --plateau-patience",
    )
    parser.add_argument(
        "--unet-hidden-dim",
        type=int,
        default=None,
        help="UNet hidden dim size",
    )
    parser.add_argument(
        "--decorrelation",
        type=float,
        default=1.0,
        help="decorrelation factor for MC configurations",
    )
    parser.add_argument(
        "--loss-function",
        choices=LOSS_FUNCTION_CHOICES,
        default=None,
        help="operate on only one objective; omit to use all objectives",
    )
    parser.add_argument(
        "--field-dependent-init",
        choices=("zero", "fixed-profile"),
        default="zero",
        help=(
            "initialize field-dependent XY training from a zero profile "
            "or the saved field-independent variance profile"
        ),
    )
    parser.add_argument(
        "--alpha-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(-5.0, 5.0),
        help="bounds for optimizing the penalty-trained field scale",
    )
    parser.add_argument(
        "--alpha-grid-points",
        type=int,
        default=41,
        help="coarse grid size used before bounded scalar alpha refinement",
    )
    parser.add_argument(
        "--alpha-xatol",
        type=float,
        default=1e-4,
        help="absolute tolerance for scalar alpha refinement",
    )
    parser.add_argument(
        "--saddle-maxiter",
        type=int,
        default=10_000,
        help="maximum L-BFGS-B iterations for the SaddlePoint field",
    )
    parser.add_argument(
        "--saddle-gtol",
        type=float,
        default=1e-8,
        help="projected-gradient tolerance for the SaddlePoint solve",
    )
    parser.add_argument(
        "--saddle-ftol",
        type=float,
        default=1e-12,
        help="relative objective tolerance for the SaddlePoint solve",
    )
    parser.add_argument(
        "--run",
        help="run training instead of loading saved losses",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--plot-training",
        help=(
            "plot individual training histories; defaults to enabled with "
            "--run and disabled with --no-run"
        ),
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--save-plot",
        metavar="PATH",
        help="save the cross-dimension scaling plot as a PDF or image",
    )
    return parser


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """Validate combinations and ranges that argparse cannot express."""
    if args.plot_training is None:
        args.plot_training = args.run

    if not args.delete_betas and len(args.dims) != len(args.Ls):
        parser.error("--dims and --Ls must contain the same number of values")
    if args.delete_betas and args.run:
        parser.error("--delete-betas cannot be combined with --run")
    if args.delete_betas and args.betas:
        parser.error("use --delete-betas without --betas")
    if any(beta <= 0 for beta in [*args.betas, *args.delete_betas]):
        parser.error("beta values must be positive")
    if any(
        beta != beta_key(beta) for beta in [*args.betas, *args.delete_betas]
    ):
        parser.error("beta values may have at most two decimal places")
    if any(int(dim) not in (1, 2, 3) for dim in args.dims):
        parser.error("--dims values must be 1, 2, or 3")
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be positive and --batch-size at least 2")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 < args.lr_factor < 1:
        parser.error("--lr-factor must be between 0 and 1")
    if args.lr_patience < 1 or args.lr_cooldown < 0:
        parser.error(
            "--lr-patience must be positive and --lr-cooldown non-negative"
        )
    if args.min_learning_rate < 0:
        parser.error("--min-learning-rate must be non-negative")
    if (
        args.lr_scheduler != "constant"
        and args.min_learning_rate > args.learning_rate
    ):
        parser.error("--min-learning-rate cannot exceed --learning-rate")
    if args.plateau_window < 1 or args.plateau_patience < 1:
        parser.error(
            "--plateau-window and --plateau-patience must be positive"
        )
    if args.plateau_min_delta < 0:
        parser.error("--plateau-min-delta must be non-negative")
    if (
        args.plateau_threshold_mode == "rel"
        and args.plateau_min_delta >= 1
    ):
        parser.error("relative --plateau-min-delta must be smaller than 1")
    if args.alpha_bounds[0] >= args.alpha_bounds[1]:
        parser.error("--alpha-bounds LOWER must be smaller than UPPER")
    if args.alpha_grid_points < 3:
        parser.error("--alpha-grid-points must be at least 3")
    if args.alpha_xatol <= 0:
        parser.error("--alpha-xatol must be positive")
    if args.saddle_maxiter < 1:
        parser.error("--saddle-maxiter must be positive")
    if args.saddle_gtol <= 0 or args.saddle_ftol <= 0:
        parser.error("--saddle-gtol and --saddle-ftol must be positive")


def main(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and execute the study."""
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    from .study import run_analysis

    run_analysis(args)
