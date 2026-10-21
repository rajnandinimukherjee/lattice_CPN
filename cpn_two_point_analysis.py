"""Train localized CP(2) deformations for two-point correlators."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cpn-two-point-mplconfig")

import numpy as np
import torch

from actions import CPnGaugeAction
from CPN_utils import ComplexPair
from cpn_opf.objectives import init_variance_network
from cpn_opf.persistence import load_deformation_field
from cpn_opf.training import (
    make_model,
    make_split_mask,
    run_training,
    save_training_result,
)
from losses import ComplexVarianceLoss
from networks import UNet
from observables import CPnTwoPtFn
from cpn_opf.loading import load_ensemble
from cpn_opf.config import BEST_WINDOW


class GluedResidualUNet(torch.nn.Module):
    """A localized two-point field initialized by a glued OPF profile."""

    def __init__(
        self,
        grid: torch.Tensor,
        initial_field: torch.Tensor,
        *,
        output_dim: int,
        hidden_dim: int | None,
    ) -> None:
        super().__init__()
        self.register_buffer("initial_field", initial_field.detach().clone())
        self.residual = UNet(
            input_dim=grid.shape[-1],
            output_dim=output_dim,
            ndim=grid.ndim - 1,
            hd=hidden_dim,
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.initial_field + self.residual(grid)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=2, help="CP(N) index")
    parser.add_argument("--beta", type=float, default=2.5)
    parser.add_argument("--lattice-size", type=int, default=64)
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument("--j", type=int, default=1)
    parser.add_argument(
        "--time-separations",
        type=int,
        nargs="+",
        default=None,
        help="sink times; default is 2, 4, ..., L/2",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--one-point-profile",
        type=Path,
        default=None,
        help=(
            "masked variance-trained CP(N) OPF artifact used for the glued "
            "field and training initialization"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("training_data/cpn_two_point")
    )
    parser.add_argument("--mask-radius", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--unet-hidden-dim", type=int, default=None)
    parser.add_argument("--decorrelation", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-training-time", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--force", action="store_true", help="replace saved separations")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--plateau-window", type=int, default=20)
    parser.add_argument("--plateau-patience", type=int, default=40)
    parser.add_argument("--plateau-min-delta", type=float, default=1e-3)
    parser.add_argument("--plateau-threshold-mode", choices=("abs", "rel"), default="rel")
    parser.add_argument("--lr-scheduler", choices=("constant", "cosine", "plateau"), default="cosine")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-cooldown", type=int, default=1)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--early-stopping", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.N < 1 or not (0 <= args.i <= args.N and 0 <= args.j <= args.N):
        parser.error("N must be positive and i, j must be between 0 and N")
    if args.beta <= 0 or args.lattice_size < 2 or args.lattice_size % 2:
        parser.error("beta must be positive and lattice-size must be even")
    if args.epochs < 1 or args.batch_size < 2 or args.mask_radius < 0:
        parser.error("epochs/batch-size must be positive and mask-radius non-negative")
    if args.chunk_size < 1:
        parser.error("chunk-size must be positive")
    if args.time_separations is None:
        args.time_separations = list(range(2, args.lattice_size // 2 + 1, 2))
    args.time_separations = sorted(set(args.time_separations))
    if any(t < 1 or t > args.lattice_size // 2 for t in args.time_separations):
        parser.error("time separations must be in [1, L/2]")


def _artifact_path(args: argparse.Namespace, separation: int) -> Path:
    beta = f"{args.beta:g}"
    return args.output_dir / (
        f"cp{args.N}_twopt_dim2_b{beta}_L{args.lattice_size}_"
        f"i{args.i}_j{args.j}_t{separation:02d}_"
        "src0-0_action-gauge_loss-variance_training.npy"
    )


def _summary_path(args: argparse.Namespace) -> Path:
    return args.output_dir / (
        f"cp{args.N}_twopt_dim2_b{args.beta:g}_L{args.lattice_size}_"
        f"i{args.i}_j{args.j}_summary.npy"
    )


def load_masked_opf_profile(
    filename: Path,
    *,
    n: int,
    beta: float,
    latsize: tuple[int, int],
    i: int,
    j: int,
) -> np.ndarray:
    """Load and validate the masked one-point OPF field used for gluing."""
    if not filename.is_file():
        raise FileNotFoundError(
            "the glued two-point result requires the masked one-point OPF "
            f"artifact at {filename}; train it with cpn_OPF_analysis.py first"
        )
    payload = np.load(filename, allow_pickle=True)
    if payload.shape != () or not isinstance(payload.item(), dict):
        raise ValueError(f"invalid one-point OPF artifact in {filename}")
    data = payload.item()
    metadata = data.get("metadata", {})
    expected = {
        "cp_n": n,
        "ndim": 2,
        "lattice_size": latsize,
        "i": i,
        "j": j,
        "action_type": "gauge",
    }
    for key, value in expected.items():
        saved = metadata.get(key)
        if key == "lattice_size" and saved is not None:
            saved = tuple(saved)
        if saved != value:
            raise ValueError(
                f"one-point OPF artifact {filename} has {key}={saved!r}; "
                f"expected {value!r}"
            )
    saved_beta = metadata.get("beta")
    if saved_beta is None or not np.isclose(float(saved_beta), beta):
        raise ValueError(
            f"one-point OPF artifact {filename} has beta={saved_beta!r}; "
            f"expected {beta:g}"
        )
    saved_loss = data.get("loss_function", metadata.get("loss_function"))
    if saved_loss != "variance":
        raise ValueError(
            f"one-point OPF artifact {filename} uses loss={saved_loss!r}; "
            "expected the variance-trained masked OPF field"
        )

    field, source = load_deformation_field(filename)
    if source != (0, 0):
        raise ValueError(
            f"one-point profile source {source} does not match required source (0, 0)"
        )
    if field.shape != (*latsize, n + 1):
        raise ValueError(
            f"one-point profile has shape {field.shape}; "
            f"expected {(*latsize, n + 1)}"
        )
    if not np.isfinite(field).all():
        raise ValueError(f"one-point OPF artifact {filename} contains non-finite values")
    return field


def _complex_numpy(values: torch.Tensor | ComplexPair) -> np.ndarray:
    if isinstance(values, ComplexPair):
        return (
            values.real.detach().cpu().numpy().astype(np.float64)
            + 1j * values.imag.detach().cpu().numpy().astype(np.float64)
        )
    return values.detach().cpu().numpy().astype(np.complex128)


def complex_statistics(values: np.ndarray) -> dict[str, float | complex]:
    values = np.asarray(values, dtype=np.complex128)
    mean = values.mean()
    variance = float(np.mean(np.abs(values - mean) ** 2))
    standard_error = float(np.sqrt(variance / len(values)))
    snr = float(abs(mean) / standard_error) if standard_error > 0 else float("inf")
    return {
        "mean": complex(mean),
        "variance": variance,
        "standard_error": standard_error,
        "snr": snr,
    }


def evaluate_deformed_in_chunks(
    model: torch.nn.Module,
    obs: CPnTwoPtFn,
    alphas: torch.Tensor,
    chunk_size: int,
) -> np.ndarray:
    """Evaluate the full deformed estimator without large lattice work arrays."""
    original_fields = model.samples["all"]
    original_links = model.link_samples["all"]
    output: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, len(original_fields), chunk_size):
                stop = min(start + chunk_size, len(original_fields))
                model.samples["all"] = original_fields[start:stop]
                model.link_samples["all"] = original_links[start:stop]
                values = obs(
                    model=model,
                    deform=True,
                    alphas=alphas,
                    sampletype="all",
                )
                output.append(_complex_numpy(values))
    finally:
        model.samples["all"] = original_fields
        model.link_samples["all"] = original_links
    return np.concatenate(output)


def save_summary(
    args: argparse.Namespace,
    records: dict[int, dict[str, float | complex | int]],
    *,
    n_cfgs: int,
    stride: int,
    tau_int: float,
    limited_by: str,
) -> None:
    separations = np.asarray(sorted(records), dtype=np.int64)
    keys = (
        "undeformed_mean",
        "glued_mean",
        "deformed_mean",
        "undeformed_variance",
        "glued_variance",
        "deformed_variance",
        "variance_ratio",
        "glued_variance_ratio",
        "undeformed_snr",
        "glued_snr",
        "deformed_snr",
        "n_cfgs",
    )
    payload: dict[str, object] = {
        "metadata": {
            "cp_n": args.N,
            "ndim": 2,
            "beta": args.beta,
            "lattice_size": args.lattice_size,
            "i": args.i,
            "j": args.j,
            "source": (0, 0),
            "action_type": "gauge",
            "time_axis": 1,
            "thinning_stride": stride,
            "tau_int": tau_int,
            "autocorrelation_limited_by": limited_by,
            "configuration_count": n_cfgs,
            "observable": "P_ij(0) * P_ji((0,t))",
            "variance": "E[|C - E(C)|^2]",
        },
        "time_separations": separations,
    }
    for key in keys:
        payload[key] = np.asarray([records[t][key] for t in separations])
    destination = _summary_path(args)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, payload, allow_pickle=True)


def read_record(filename: Path) -> dict[str, float | complex | int]:
    data = np.load(filename, allow_pickle=True).item()
    return data["metrics"]


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    source = (0, 0)
    latsize = (args.lattice_size, args.lattice_size)
    profile_path = args.one_point_profile or Path(
        "training_data"
    ) / (
        f"cp{args.N}_scaling_dim2_b{args.beta:g}_L{args.lattice_size}_"
        f"i{args.i}_j{args.j}_action-gauge_loss-variance_training.npy"
    )
    one_point_field = load_masked_opf_profile(
        profile_path,
        n=args.N,
        beta=args.beta,
        latsize=latsize,
        i=args.i,
        j=args.j,
    )
    seed_description = f"masked OPF variance artifact: {profile_path}"
    ensemble = args.data_dir / (
        f"cp{args.N}_gauge_dim2_b{args.beta:.2f}_L{args.lattice_size}_ens.npy"
    )
    if not ensemble.exists():
        raise FileNotFoundError(ensemble)

    load_diagnostics: dict[str, object] = {}
    fields, links, saved_beta = load_ensemble(
        ensemble,
        n=args.N,
        ndim=2,
        lattice_size=args.lattice_size,
        beta=args.beta,
        action_type="gauge",
        i=args.i,
        j=args.j,
        decorrelation_factor=args.decorrelation,
        diagnostics=load_diagnostics,
    )
    if saved_beta is not None and not np.isclose(saved_beta, args.beta):
        raise ValueError(f"ensemble stores beta={saved_beta:g}, requested {args.beta:g}")
    fields = fields.to(device)
    if links is None:
        raise ValueError("the selected gauge ensemble has no auxiliary links")
    links = links.to(device)
    split_mask = make_split_mask(fields)
    action = CPnGaugeAction(args.beta, args.N, latsize, device=device)
    args.ndim = 2
    args.thinning_stride = int(load_diagnostics["stride"])
    args.tau_int = float(load_diagnostics["tau_int"])
    args.autocorrelation_limited_by = str(load_diagnostics["limited_by"])

    records: dict[int, dict[str, float | complex | int]] = {}
    for separation in args.time_separations:
        destination = _artifact_path(args, separation)
        if destination.exists() and not args.force:
            saved_record = read_record(destination)
            if "glued_variance_ratio" in saved_record:
                print(f"Using saved t={separation}: {destination}")
                records[separation] = saved_record
                save_summary(
                    args,
                    records,
                    n_cfgs=len(fields),
                    stride=args.thinning_stride,
                    tau_int=args.tau_int,
                    limited_by=args.autocorrelation_limited_by,
                )
                continue
            print(f"Replacing incomplete t={separation}: {destination}")

        started = time.monotonic()
        sink = (0, separation)
        obs = CPnTwoPtFn(
            args.N,
            latsize,
            i=args.i,
            j=args.j,
            source=source,
            delta=(0, separation),
            action=action,
            device=device,
        )
        model = make_model(
            fields,
            links,
            n=args.N,
            action_type="gauge",
            split_mask=split_mask,
            device=device,
        )
        loss_fn = ComplexVarianceLoss(model, obs)
        glued_field = one_point_field - np.roll(
            one_point_field, shift=sink, axis=(0, 1)
        )
        glued_tensor = torch.as_tensor(
            glued_field, dtype=torch.float32, device=device
        )
        alphanet = GluedResidualUNet(
            obs.grid,
            glued_tensor,
            output_dim=model.DOF,
            hidden_dim=args.unet_hidden_dim,
        ).to(device)
        init_variance_network(alphanet.residual)

        with torch.no_grad():
            undeformed_values = _complex_numpy(
                obs(model=model, deform=False, sampletype="all")
            )
        glued_values = evaluate_deformed_in_chunks(
            model, obs, glued_tensor, args.chunk_size
        )
        undeformed = complex_statistics(undeformed_values)
        glued = complex_statistics(glued_values)
        trainer = run_training(
            model,
            loss_fn,
            alphanet,
            latsize,
            source,
            args,
            support_centers=(source, sink),
        )
        best_alphas = trainer.best_alphas(BEST_WINDOW)

        deformed_values = evaluate_deformed_in_chunks(
            model, obs, best_alphas, args.chunk_size
        )
        deformed = complex_statistics(deformed_values)
        if undeformed["variance"] <= 0:
            raise ValueError(f"undeformed variance is not positive at t={separation}")
        ratio = float(deformed["variance"] / undeformed["variance"])
        metrics: dict[str, float | complex | int] = {
            "separation": separation,
            "undeformed_mean": undeformed["mean"],
            "glued_mean": glued["mean"],
            "deformed_mean": deformed["mean"],
            "undeformed_variance": undeformed["variance"],
            "glued_variance": glued["variance"],
            "deformed_variance": deformed["variance"],
            "variance_ratio": ratio,
            "glued_variance_ratio": float(glued["variance"] / undeformed["variance"]),
            "undeformed_snr": undeformed["snr"],
            "glued_snr": glued["snr"],
            "deformed_snr": deformed["snr"],
            "n_cfgs": len(fields),
            "best_epoch": int(trainer.best_epoch or 0),
        }
        extra_metadata = {
            "separation": separation,
            "sink": sink,
            "time_axis": 1,
            "observable": "P_ij(source) * P_ji(sink)",
            "operator_definition": "P_ij(x)=z_i(x) zbar_j(x)-delta_ij/(N+1)",
            "complex_variance_definition": "E[|C-E(C)|^2]",
            "checkpoint_loss": "E[|C-mean_train(C_undeformed)|^2]",
            "initial_field_seed": seed_description,
            "glued_field_source": str(profile_path),
            "gluing_rule": "f_OPF(x) - f_OPF(x-sink), using the masked one-point OPF field",
            "metrics": metrics,
            "thinning_stride": args.thinning_stride,
            "tau_int": args.tau_int,
            "autocorrelation_limited_by": args.autocorrelation_limited_by,
        }
        save_training_result(
            destination,
            trainer,
            n=args.N,
            action_type="gauge",
            ndim=2,
            beta=args.beta,
            latsize=latsize,
            source=source,
            i=args.i,
            j=args.j,
            learned_field=best_alphas,
            variance_ratio=ratio,
            loss_function="variance",
            extra_metadata=extra_metadata,
            extra_data={
                "metrics": metrics,
                "glued_field": glued_tensor.detach().cpu().numpy(),
                "glued_variance_ratio": metrics["glued_variance_ratio"],
            },
        )
        records[separation] = metrics
        save_summary(
            args,
            records,
            n_cfgs=len(fields),
            stride=args.thinning_stride,
            tau_int=args.tau_int,
            limited_by=args.autocorrelation_limited_by,
        )
        print(
            f"t={separation:2d}: variance ratio={ratio:.4g}, "
            f"SNR {undeformed['snr']:.4g} -> glued {glued['snr']:.4g} "
            f"-> trained {deformed['snr']:.4g}, "
            f"best epoch={trainer.best_epoch}, "
            f"elapsed={time.monotonic() - started:.1f}s"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    run(args)


if __name__ == "__main__":
    main()
