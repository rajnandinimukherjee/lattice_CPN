"""Command-line interface for CP(N) OPF scaling studies."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import math

from .config import LOSS_FUNCTIONS


def build_parser() -> argparse.ArgumentParser:
    """Build the parser without executing a CP(N) study."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", "--n", dest="N", type=int,
                        default=2, help="CP(N) index")
    parser.add_argument(
        "--action",
        type=str.lower,
        choices=("gauge", "quartic"),
        default="gauge",
        help="CP(N) action to analyse (default: Gauge)",
    )
    parser.add_argument("--i", type=int, default=0,
                        help="first OPF component index")
    parser.add_argument("--j", type=int, default=1,
                        help="second OPF component index")
    parser.add_argument("--betas", nargs="+", type=float, default=[],
                        help="betas to train; omit to discover ensembles")
    parser.add_argument("--delete-betas", nargs="+", type=float, default=[],
                        metavar="BETA", help="delete only these saved beta results")
    parser.add_argument("--dims", nargs="+", type=int,
                        required=True, help="lattice dimensions, e.g. 2 3")
    parser.add_argument("--Ls", nargs="+", type=int, required=True,
                        help="common linear lattice size for each dimension")
    parser.add_argument("--mask-radius", type=int, default=6,
                        help="OPF localisation mask radius")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-scheduler", choices=("constant",
                        "cosine", "plateau"), default="cosine")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-cooldown", type=int, default=1)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-training-time", type=float, default=None)
    parser.add_argument("--plateau-window", type=int, default=20)
    parser.add_argument("--plateau-patience", type=int, default=40)
    parser.add_argument("--plateau-min-delta", type=float, default=1e-3)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--validation-samples", type=int, default=None)
    parser.add_argument("--plateau-threshold-mode",
                        choices=("abs", "rel"), default="rel")
    parser.add_argument("--early-stopping",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unet-hidden-dim", type=int, default=None)
    parser.add_argument("--decorrelation", type=float, default=1.0,
                        help="multiple of g=2*tau_int used to thin configurations")
    parser.add_argument(
        "--loss-function",
        choices=LOSS_FUNCTIONS,
        default=None,
        help="operate on only one objective; omit to use all objectives",
    )
    parser.add_argument(
        "--variance-init-from-approx2",
        action="store_true",
        help=(
            "initialize direct variance training from the saved, scaled "
            "Approximation 2 field"
        ),
    )
    parser.add_argument(
        "--penalty-init-from-approx2",
        action="store_true",
        help=(
            "initialize penalty training from the saved, normalized "
            "Approximation 2 profile and learn a U-Net residual"
        ),
    )
    parser.add_argument(
        "--alpha-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(-5.0, 5.0),
        help="bounds for scaling a penalty-trained deformation field",
    )
    parser.add_argument(
        "--alpha-grid-points",
        type=int,
        default=41,
        help="coarse grid size before bounded scalar-alpha refinement",
    )
    parser.add_argument(
        "--alpha-xatol",
        type=float,
        default=1e-4,
        help="absolute tolerance for scalar-alpha refinement",
    )
    parser.add_argument("--approx2-maxiter", type=int, default=2000)
    parser.add_argument("--approx2-gtol", type=float, default=1e-9)
    parser.add_argument("--approx2-ftol", type=float, default=1e-12)
    parser.add_argument(
        "--approx3-max-configurations",
        type=int,
        default=1280,
        help="maximum decorrelated configurations used for Approximation 3 fits",
    )
    parser.add_argument(
        "--recompute-approx3",
        action="store_true",
        help="recompute Approximation 3 results even when a matching result is saved",
    )
    parser.add_argument("--device", choices=("auto", "cpu",
                        "cuda", "mps"), default="auto")
    parser.add_argument("--run", action=argparse.BooleanOptionalAction,
                        default=False, help="train rather than load saved artifacts")
    parser.add_argument("--plot-training",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--plot-field",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "plot the saved deformation field for every beta and CP(N) "
            "component (3D fields use a source-centred 2D slice)"
        ),
    )
    parser.add_argument("--save-plot", action=argparse.BooleanOptionalAction,
                        default=True, help="save training and scaling figures")
    parser.add_argument(
        "--save-legend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save a standalone scaling-plot legend",
    )
    parser.add_argument("--plot-file", default=None,
                        help="override the scaling-plot filename")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction,
                        default=False, help="display figures interactively")
    return parser


def validate_args(
        parser: argparse.ArgumentParser,
        args: argparse.Namespace
) -> None:
    """Validate argument combinations and numerical ranges."""
    if args.plot_training is None:
        args.plot_training = args.run
    if len(args.dims) != len(args.Ls):
        parser.error("--dims and --Ls must contain the same number of values")
    if args.delete_betas and args.run:
        parser.error("--delete-betas cannot be combined with --run")
    if args.delete_betas and args.betas:
        parser.error("use --delete-betas without --betas")
    if args.N < 1:
        parser.error("--N must be positive")
    if not 0 <= args.i <= args.N or not 0 <= args.j <= args.N:
        parser.error("--i and --j must lie between 0 and N")
    if any(dim not in (1, 2, 3) for dim in args.dims):
        parser.error("--dims values must be 1, 2, or 3")
    if any(length < 1 for length in args.Ls):
        parser.error("--Ls values must be positive")
    if any(not math.isfinite(beta) or beta <= 0 for beta in [*args.betas, *args.delete_betas]):
        parser.error("beta values must be positive and finite")
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be positive and --batch-size at least 2")
    if args.mask_radius < 0 or args.learning_rate <= 0:
        parser.error(
            "--mask-radius must be non-negative and --learning-rate positive")
    if not math.isfinite(args.decorrelation) or args.decorrelation <= 0:
        parser.error("--decorrelation must be positive and finite")
    if not 0 < args.lr_factor < 1 or args.lr_patience < 1 or args.lr_cooldown < 0:
        parser.error("invalid learning-rate scheduler settings")
    if args.min_learning_rate < 0 or (args.lr_scheduler != "constant" and args.min_learning_rate > args.learning_rate):
        parser.error(
            "--min-learning-rate must be non-negative and no larger than --learning-rate")
    if args.max_training_time is not None and args.max_training_time <= 0:
        parser.error("--max-training-time must be positive")
    if args.plateau_window < 1 or args.plateau_patience < 1 or args.plateau_min_delta < 0:
        parser.error("invalid plateau settings")
    if args.validation_interval < 1:
        parser.error("--validation-interval must be positive")
    if args.validation_samples is not None and args.validation_samples < 2:
        parser.error("--validation-samples must be at least 2")
    if args.plateau_threshold_mode == "rel" and args.plateau_min_delta >= 1:
        parser.error("relative --plateau-min-delta must be smaller than 1")
    if args.alpha_bounds[0] >= args.alpha_bounds[1]:
        parser.error("--alpha-bounds LOWER must be smaller than UPPER")
    if args.alpha_grid_points < 3:
        parser.error("--alpha-grid-points must be at least 3")
    if not math.isfinite(args.alpha_xatol) or args.alpha_xatol <= 0:
        parser.error("--alpha-xatol must be positive and finite")
    if args.approx2_maxiter < 1:
        parser.error("--approx2-maxiter must be positive")
    if args.approx3_max_configurations < 2:
        parser.error("--approx3-max-configurations must be at least two")
    if not math.isfinite(args.approx2_gtol) or args.approx2_gtol <= 0:
        parser.error("--approx2-gtol must be positive and finite")
    if not math.isfinite(args.approx2_ftol) or args.approx2_ftol <= 0:
        parser.error("--approx2-ftol must be positive and finite")
    if (
        args.loss_function in {"approx2", "approx3"}
        and args.action != "gauge"
    ):
        parser.error(
            "Approximations 2 and 3 are available only for the gauge action"
        )
    if (
        args.variance_init_from_approx2 or args.penalty_init_from_approx2
    ) and args.action != "gauge":
        parser.error("Approximation 2 initialization requires the gauge action")


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and execute the CP(N) OPF study."""
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    from .study import run_analysis

    run_analysis(args)
