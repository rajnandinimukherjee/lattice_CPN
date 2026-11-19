"""Command-line interface for the complex-scalar OPF study."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import LOSS_FUNCTION_CHOICES
from .persistence import am_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ams",
        nargs="+",
        type=float,
        default=[],
        help="bare masses to run; omit to discover matching ensemble files",
    )
    parser.add_argument(
        "--delete-ams", nargs="+", type=float, default=[], metavar="AM"
    )
    parser.add_argument("--dims", nargs="+", required=True)
    parser.add_argument("--Ls", nargs="+", default=[])
    parser.add_argument("--mask-radius", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
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
    parser.add_argument("--plateau-patience", type=int, default=40)
    parser.add_argument("--plateau-min-delta", type=float, default=1e-3)
    parser.add_argument(
        "--plateau-threshold-mode", choices=("abs", "rel"), default="rel"
    )
    parser.add_argument(
        "--early-stopping", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--unet-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--loss-function", choices=LOSS_FUNCTION_CHOICES, default=None
    )
    parser.add_argument(
        "--field-dependent-init",
        choices=("zero", "fixed-profile"),
        default="zero",
    )
    parser.add_argument(
        "--alpha-bounds", nargs=2, type=float, default=(-5.0, 5.0)
    )
    parser.add_argument("--alpha-grid-points", type=int, default=41)
    parser.add_argument("--alpha-xatol", type=float, default=1e-4)
    parser.add_argument("--saddle-maxiter", type=int, default=10_000)
    parser.add_argument("--saddle-gtol", type=float, default=1e-8)
    parser.add_argument("--saddle-ftol", type=float, default=1e-12)
    parser.add_argument(
        "--run", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--plot-training", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--save-plot", metavar="PATH")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.plot_training is None:
        args.plot_training = args.run
    if not args.delete_ams and len(args.dims) != len(args.Ls):
        parser.error("--dims and --Ls must contain the same number of values")
    if args.delete_ams and args.run:
        parser.error("--delete-ams cannot be combined with --run")
    if args.delete_ams and args.ams:
        parser.error("use --delete-ams without --ams")
    masses = [*args.ams, *args.delete_ams]
    if any(am <= 0 for am in masses):
        parser.error("am values must be positive")
    if any(am != am_key(am) for am in masses):
        parser.error("am values may have at most three decimal places")
    if any(int(dim) not in (1, 2, 3) for dim in args.dims):
        parser.error("--dims values must be 1, 2, or 3")
    if args.mask_radius < 0:
        parser.error("--mask-radius must be non-negative")
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be positive and --batch-size at least 2")
    if args.learning_rate <= 0 or args.min_learning_rate < 0:
        parser.error("learning rates must be non-negative and initial rate positive")
    if args.lr_scheduler != "constant" and args.min_learning_rate > args.learning_rate:
        parser.error("--min-learning-rate cannot exceed --learning-rate")
    if not 0 < args.lr_factor < 1:
        parser.error("--lr-factor must be between zero and one")
    if args.lr_patience < 1 or args.lr_cooldown < 0:
        parser.error("invalid learning-rate patience/cooldown")
    if args.plateau_window < 1 or args.plateau_patience < 1:
        parser.error("plateau window and patience must be positive")
    if args.plateau_min_delta < 0:
        parser.error("--plateau-min-delta must be non-negative")
    if args.plateau_threshold_mode == "rel" and args.plateau_min_delta >= 1:
        parser.error("relative --plateau-min-delta must be smaller than one")
    if args.alpha_bounds[0] >= args.alpha_bounds[1]:
        parser.error("alpha bounds must be strictly increasing")
    if args.alpha_grid_points < 3 or args.alpha_xatol <= 0:
        parser.error("invalid scalar alpha optimization settings")
    if args.saddle_maxiter < 1 or args.saddle_gtol <= 0 or args.saddle_ftol <= 0:
        parser.error("invalid Approximation-2 optimizer settings")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    from .study import run_analysis

    run_analysis(args)
