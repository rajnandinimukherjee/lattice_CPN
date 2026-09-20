from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Sequence
from typing import Any

import h5py
import torch
import torch.nn as nn
import torch.optim as optim

import deformations as defs
from actions import CPnAction
from CPN_utils import CNtoR2N
from losses import VarianceLoss
from networks import UNet
from observables import CPnOnePtFn, CPnTwoPtFn
from training import Trainer, init_weights, periodic_box_mask

LEARNING_RATE = 1e-3
DEFORMATIONS: dict[str, Callable[..., nn.Module]] = {
    "HomogDef": defs.HomogenousDeformations,
}


def parse_int_tuple(value: str) -> tuple[int, ...]:
    """Parse an integer tuple supplied on the command line."""
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (tuple, list)):
        raise argparse.ArgumentTypeError("expected a tuple such as '(64, 64)'")
    try:
        return tuple(int(item) for item in parsed)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("tuple values must be integers") from error


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatically selected PyTorch device."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_samples(args: argparse.Namespace) -> torch.Tensor:
    """Load CPN configurations and return interleaved real coordinates."""
    if args.source == "Raj":
        lattice_label = "x".join(str(length) for length in args.latsize)
        filename = f"data/CP{args.N}_lat{lattice_label}_b{args.beta:.1f}.h5"
        with h5py.File(filename, "r") as data:
            vectors = torch.as_tensor(
                data["configs/vectors"][2000::20], dtype=torch.complex128
            )
        return CNtoR2N(vectors)

    expected_n = 2
    expected_lattice = (64, 64)
    if args.N != expected_n or tuple(args.latsize) != expected_lattice:
        raise ValueError("the Tej ensemble is fixed to CP(2) on a 64x64 lattice")
    vectors = torch.from_file(
        "data/cpn_b4.0_L64_Nc3_big_ens.dat",
        shared=False,
        size=2000 * 64 * 64 * 3,
        dtype=torch.complex128,
    ).reshape(2000, 64, 64, 3)[50:]
    return CNtoR2N(vectors)


def main(
    args: argparse.Namespace,
    deformation: Callable[..., nn.Module] | None = None,
) -> Trainer | None:
    """Run a CPN variance-reduction analysis from parsed arguments."""
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    samples = load_samples(args).to(dtype=torch.float32, device=device)
    number_of_configurations = len(samples)

    action = CPnAction(args.beta, args.N, args.latsize, device=device)
    volume = torch.tensor(args.latsize).prod().item()
    print(
        "\n".join(
            [
                "Ensemble details:",
                f"  source : {args.source}",
                f"  size   : {args.latsize}",
                f"  beta   : {args.beta}",
                f"  N_conf : {number_of_configurations}",
                f"  E/V    : {action.energy(samples).mean()/volume:.6f}",
            ]
        )
    )

    train_indices = torch.randperm(number_of_configurations, device=device)[
        : int(args.split * number_of_configurations)
    ]
    train_mask = torch.zeros(number_of_configurations, dtype=torch.bool, device=device)
    train_mask[train_indices] = True

    observable_type = CPnOnePtFn if args.obs == "OPF" else CPnTwoPtFn
    observable_kwargs: dict[str, Any] = {
        "N": args.N,
        "latsize": args.latsize,
        "i": args.i,
        "j": args.j,
        "source": (0,) * len(args.latsize),
        "action": action,
        "device": device,
    }
    if args.obs == "TPF":
        observable_kwargs["delta"] = args.delta
    observable = observable_type(**observable_kwargs)

    deformation_type = deformation or DEFORMATIONS[args.deformation]
    model = deformation_type(args.N, device=device).to(device)
    model.samples = {
        "all": samples,
        "train": samples[train_mask],
        "test": samples[~train_mask],
    }

    loss_fn = VarianceLoss(model, observable)
    alphanet = UNet(
        input_dim=observable.grid.shape[-1],
        output_dim=model.DOF,
        ndim=len(args.latsize),
        hd=args.hidden_dim,
    ).to(device)
    alphanet.apply(init_weights)
    alpha_mask = (
        periodic_box_mask(
            args.latsize,
            (0,) * len(args.latsize),
            radius=args.mask_radius,
        )
        .unsqueeze(-1)
        .to(device)
        if args.obs == "OPF"
        else None
    )

    optimizer = optim.Adam(alphanet.parameters(), lr=args.learning_rate, eps=1e-6)
    trainer = Trainer(model, loss_fn, optimizer, args.epochs, args.batch_size)

    print(f"<O> = {loss_fn.target_exp:.6f},\t" f"Var(O) = {loss_fn.initial_var:.6f}")
    undeformed = observable(model)
    print(f"|<O>|^2 = {torch.abs(undeformed.mean()).item() ** 2:.6f}")
    print(f"<|O|>^2 = {undeformed.abs().mean().item() ** 2:.6f}")
    print(f"<|O|^2> = {undeformed.abs().square().mean().item():.6f}")

    if not args.run:
        return None

    trainer.train(
        alphanet,
        alpha_mask,
        plot=args.plot,
        max_training_time=args.max_training_time,
        plateau_window=args.plateau_window,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
        extension=args.training_extension,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_cooldown=args.lr_cooldown,
        min_lr=args.min_lr,
    )
    return trainer


def build_parser(default_device: str = "cpu") -> argparse.ArgumentParser:
    """Build the command-line parser shared by CPN analysis scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="CP(N) dimension", type=int, default=2)
    parser.add_argument(
        "--obs", choices=("OPF", "TPF"), default="OPF", help="observable"
    )
    parser.add_argument(
        "--latsize",
        help='lattice dimensions, e.g. "(64, 64)"',
        type=parse_int_tuple,
        default=(64, 64),
    )
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument("--j", type=int, default=0)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--delta", type=parse_int_tuple, default=(0, 1))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-training-time", type=float, default=None)
    parser.add_argument("--plateau-window", type=int, default=20)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-min-delta", type=float, default=1e-4)
    parser.add_argument("--training-extension", type=int, default=100)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--lr-cooldown", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1000)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument(
        "--deformation", choices=tuple(DEFORMATIONS), default="HomogDef"
    )
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--mask-radius", type=int, default=6)
    parser.add_argument("--run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--source", choices=("Raj", "Tej"), default="Tej")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=default_device,
    )
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
    default_device: str = "cpu",
) -> argparse.Namespace:
    """Parse and validate CPN command-line arguments."""
    args = build_parser(default_device=default_device).parse_args(argv)
    if not 0 < args.split < 1:
        raise ValueError("split must lie strictly between zero and one")
    if len(args.delta) != len(args.latsize):
        raise ValueError("delta and latsize must have the same dimension")
    return args


def cli(default_device: str = "cpu") -> None:
    """Run the CPN command-line entry point."""
    main(parse_args(default_device=default_device))


if __name__ == "__main__":
    cli()
