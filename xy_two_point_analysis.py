"""Train localized XY deformations for two-point correlators."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import time

cache_dir = Path(tempfile.gettempdir()) / "lattice-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lattice-mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn as nn

import XY_utils as xy
from actions import XYAction
from deformations import ThetaDefs
from losses import ComplexVarianceLoss
from networks import UNet
from observables import XYTwoPtFn
from training import init_weights
from xy_opf.config import BEST_WINDOW
from xy_opf.training import (
    make_split_mask,
    run_training,
    save_training_result,
)


class GluedResidualUNet(nn.Module):
    """A two-point XY field initialized by a glued one-point profile."""

    def __init__(
        self,
        grid: torch.Tensor,
        initial_field: torch.Tensor,
        *,
        hidden_dim: int | None,
    ) -> None:
        super().__init__()
        self.register_buffer("initial_field", initial_field.detach().clone())
        self.residual = UNet(
            input_dim=grid.shape[-1],
            output_dim=1,
            ndim=grid.ndim - 1,
            hd=hidden_dim,
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.initial_field + self.residual(grid)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=0.65)
    parser.add_argument("--lattice-size", type=int, default=32)
    parser.add_argument(
        "--time-separations",
        type=int,
        nargs="+",
        default=None,
        help="sink times; default is 1, 2, ..., L/2",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/xy_dim2_b0.65_L32_ens.npy"),
    )
    parser.add_argument(
        "--one-point-profile",
        type=Path,
        default=Path(
            "training_data/xy_scaling_dim2_b0.65_loss-variance_training.npy"
        ),
        help="masked variance-trained XY OPF artifact used for gluing",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("training_data/xy_two_point")
    )
    parser.add_argument("--mask-radius", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--unet-hidden-dim", type=int, default=None)
    parser.add_argument("--decorrelation", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-training-time", type=float, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--plateau-window", type=int, default=20)
    parser.add_argument("--plateau-patience", type=int, default=40)
    parser.add_argument("--plateau-min-delta", type=float, default=1e-3)
    parser.add_argument(
        "--plateau-threshold-mode", choices=("abs", "rel"), default="rel"
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("constant", "cosine", "plateau"),
        default="plateau",
    )
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-cooldown", type=int, default=1)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.beta <= 0 or args.lattice_size < 2 or args.lattice_size % 2:
        parser.error("beta must be positive and lattice-size must be even")
    if args.epochs < 1 or args.batch_size < 2 or args.mask_radius < 0:
        parser.error("epochs/batch-size must be positive and mask-radius non-negative")
    if args.chunk_size < 1:
        parser.error("chunk-size must be positive")
    if args.time_separations is None:
        args.time_separations = list(range(1, args.lattice_size // 2 + 1))
    args.time_separations = sorted(set(args.time_separations))
    if any(t < 1 or t > args.lattice_size // 2 for t in args.time_separations):
        parser.error("time separations must be in [1, L/2]")
    if not np.isfinite(args.decorrelation) or args.decorrelation <= 0:
        parser.error("decorrelation must be positive and finite")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if args.max_training_time is not None and args.max_training_time <= 0:
        parser.error("max-training-time must be positive")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _artifact_path(args: argparse.Namespace, separation: int) -> Path:
    return args.output_dir / (
        f"xy_twopt_dim2_b{args.beta:g}_L{args.lattice_size}_"
        f"t{separation:02d}_src0-0_loss-variance_training.npy"
    )


def _summary_path(args: argparse.Namespace) -> Path:
    return args.output_dir / (
        f"xy_twopt_dim2_b{args.beta:g}_L{args.lattice_size}_summary.npy"
    )


def load_masked_opf_profile(
    filename: Path,
    *,
    beta: float,
    latsize: tuple[int, int],
) -> np.ndarray:
    """Load and validate the masked one-point XY variance-training field."""
    if not filename.is_file():
        raise FileNotFoundError(
            "the glued two-point result requires the masked XY OPF artifact "
            f"at {filename}; train it with xy_OPF_analysis.py first"
        )
    payload = np.load(filename, allow_pickle=True)
    if payload.shape != () or not isinstance(payload.item(), dict):
        raise ValueError(f"invalid one-point XY OPF artifact in {filename}")
    data = payload.item()
    metadata = data.get("metadata", {})
    if metadata.get("loss_function", data.get("loss_function")) != "variance":
        raise ValueError(f"{filename} is not a variance-trained XY OPF artifact")
    if int(metadata.get("ndim", -1)) != 2:
        raise ValueError(f"{filename} is not a two-dimensional XY OPF artifact")
    if tuple(metadata.get("lattice_size", ())) != latsize:
        raise ValueError(
            f"{filename} has lattice_size={metadata.get('lattice_size')!r}; "
            f"expected {latsize!r}"
        )
    if tuple(metadata.get("source", ())) != (0, 0):
        raise ValueError(
            f"{filename} has source={metadata.get('source')!r}; expected (0, 0)"
        )
    saved_beta = metadata.get("beta")
    if saved_beta is None or not np.isclose(float(saved_beta), beta):
        raise ValueError(
            f"{filename} has beta={saved_beta!r}; expected {beta:g}"
        )
    field = np.asarray(
        data.get("scaled_field", data.get("learned_field")), dtype=float
    )
    if field.shape != (*latsize, 1):
        raise ValueError(
            f"one-point profile has shape {field.shape}; expected {(*latsize, 1)}"
        )
    if not np.isfinite(field).all():
        raise ValueError(f"{filename} contains non-finite field values")
    return field


def _complex_numpy(values: torch.Tensor) -> np.ndarray:
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


def fixed_signal_snr(signal_mean: complex, variance: float, n_cfgs: int) -> float:
    """Use the undeformed signal with a method-specific variance estimate."""
    if variance <= 0 or n_cfgs < 1:
        raise ValueError("variance and configuration count must be positive")
    return float(abs(signal_mean) * np.sqrt(n_cfgs / variance))


def evaluate_deformed_in_chunks(
    model: ThetaDefs,
    obs: XYTwoPtFn,
    alphas: torch.Tensor,
    chunk_size: int,
) -> np.ndarray:
    """Evaluate the deformed estimator without full-ensemble action arrays."""
    original_fields = model.samples["all"]
    output: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, len(original_fields), chunk_size):
                stop = min(start + chunk_size, len(original_fields))
                model.samples["all"] = original_fields[start:stop]
                values = obs(
                    model=model,
                    deform=True,
                    alphas=alphas,
                    sampletype="all",
                )
                output.append(_complex_numpy(values))
    finally:
        model.samples["all"] = original_fields
    return np.concatenate(output)


def read_record(filename: Path, profile_path: Path) -> dict | None:
    payload = np.load(filename, allow_pickle=True)
    if payload.shape != () or not isinstance(payload.item(), dict):
        return None
    data = payload.item()
    metadata = data.get("metadata", {})
    if metadata.get("glued_field_source") != str(profile_path):
        return None
    return data.get("metrics")


def save_summary(
    args: argparse.Namespace,
    records: dict[int, dict],
    *,
    n_cfgs: int,
    diagnostics: dict[str, object],
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
    summary = {
        "metadata": {
            "model": "XY",
            "ndim": 2,
            "beta": float(args.beta),
            "lattice_size": args.lattice_size,
            "source": (0, 0),
            "action": "nearest-neighbor XY",
            "observable": "exp(i*(theta_source-theta_sink))",
            "variance": "E[|C-E(C)|^2]",
            "stn_definition": (
                "abs(mean(C_undeformed)) / "
                "sqrt(variance(C_method) / configuration_count)"
            ),
            "one_point_profile": str(args.one_point_profile),
            "gluing_rule": "f(x-source) - f(x-sink)",
            "thinning_stride": diagnostics["stride"],
            "tau_int": diagnostics["tau_int"],
            "effective_sample_size": diagnostics["effective_sample_size"],
            "limiting_diagnostic": diagnostics["limiting_diagnostic"],
            "raw_configuration_count": diagnostics["raw_configuration_count"],
            "configuration_count": n_cfgs,
            "decorrelation_factor": args.decorrelation,
        },
        "time_separations": separations,
    }
    for key in keys:
        dtype = np.complex128 if key.endswith("_mean") else (
            np.int64 if key == "n_cfgs" else np.float64
        )
        summary[key] = np.asarray([records[t][key] for t in separations], dtype=dtype)
    for method in ("undeformed", "glued", "deformed"):
        summary[f"fixed_signal_{method}_snr"] = np.asarray(
            [
                fixed_signal_snr(
                    summary["undeformed_mean"][index],
                    summary[f"{method}_variance"][index],
                    int(summary["n_cfgs"][index]),
                )
                for index in range(len(separations))
            ],
            dtype=np.float64,
        )
    output = _summary_path(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, summary, allow_pickle=True)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    latsize = (args.lattice_size, args.lattice_size)
    source = (0, 0)
    one_point_field = load_masked_opf_profile(
        args.one_point_profile, beta=args.beta, latsize=latsize
    )

    diagnostics: dict[str, object] = {}
    fields, saved_beta = xy.load_ensemble(
        args.data_file,
        decorrelation_factor=args.decorrelation,
        diagnostics=diagnostics,
    )
    if not np.isclose(saved_beta, args.beta):
        raise ValueError(
            f"ensemble stores beta={saved_beta:g}, requested {args.beta:g}"
        )
    if tuple(fields.shape[1:]) != latsize:
        raise ValueError(
            f"ensemble has lattice shape {tuple(fields.shape[1:])}; expected {latsize}"
        )
    fields = fields.to(device)
    split_mask = make_split_mask(fields)
    action = XYAction(args.beta, ndims=2)

    records: dict[int, dict] = {}
    for separation in args.time_separations:
        destination = _artifact_path(args, separation)
        if destination.exists() and not args.force:
            saved_record = read_record(destination, args.one_point_profile)
            if saved_record is not None and "glued_variance_ratio" in saved_record:
                print(f"Using saved t={separation}: {destination}")
                records[separation] = saved_record
                save_summary(
                    args,
                    records,
                    n_cfgs=len(fields),
                    diagnostics=diagnostics,
                )
                continue
            print(f"Replacing incomplete or stale t={separation}: {destination}")

        started = time.monotonic()
        sink = (0, separation)
        obs = XYTwoPtFn(
            latsize,
            source=source,
            sink=sink,
            action=action,
        )
        obs.grid = obs.grid.to(device)
        model = ThetaDefs(ndims=2)
        model.samples = {
            "all": fields,
            "train": fields[split_mask],
            "test": fields[~split_mask],
        }
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
            hidden_dim=args.unet_hidden_dim,
        ).to(device)
        alphanet.residual.apply(init_weights)
        with torch.no_grad():
            nn.init.zeros_(alphanet.residual.output.weight)
            nn.init.zeros_(alphanet.residual.output.bias)

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
            raise ValueError(
                f"undeformed variance is not positive at t={separation}"
            )
        ratio = float(deformed["variance"] / undeformed["variance"])
        metrics = {
            "separation": separation,
            "undeformed_mean": undeformed["mean"],
            "glued_mean": glued["mean"],
            "deformed_mean": deformed["mean"],
            "undeformed_variance": undeformed["variance"],
            "glued_variance": glued["variance"],
            "deformed_variance": deformed["variance"],
            "variance_ratio": ratio,
            "glued_variance_ratio": float(
                glued["variance"] / undeformed["variance"]
            ),
            "undeformed_snr": undeformed["snr"],
            "glued_snr": glued["snr"],
            "deformed_snr": deformed["snr"],
            "n_cfgs": len(fields),
            "best_epoch": int(trainer.best_epoch or 0),
        }
        for method, statistics in (
            ("undeformed", undeformed),
            ("glued", glued),
            ("deformed", deformed),
        ):
            metrics[f"fixed_signal_{method}_snr"] = fixed_signal_snr(
                undeformed["mean"], statistics["variance"], len(fields)
            )
        save_training_result(
            destination,
            trainer,
            loss_function="variance",
            ndim=2,
            beta=args.beta,
            latsize=latsize,
            source=source,
            learned_field=best_alphas,
            alpha_scale=1.0,
            variance_ratio=ratio,
            extra_metadata={
                "sink": sink,
                "observable": "exp(i*(theta_source-theta_sink))",
                "variance_definition": "E[|C-E(C)|^2]",
                "initial_field_seed": f"masked XY OPF variance artifact: {args.one_point_profile}",
                "glued_field_source": str(args.one_point_profile),
                "gluing_rule": "f(x-source) - f(x-sink)",
                "metrics": metrics,
                "thinning_stride": diagnostics["stride"],
                "tau_int": diagnostics["tau_int"],
                "limiting_diagnostic": diagnostics["limiting_diagnostic"],
            },
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
            diagnostics=diagnostics,
        )
        print(
            f"t={separation:2d}: variance ratio={ratio:.4g}, "
            f"fixed-signal SNR {metrics['fixed_signal_undeformed_snr']:.4g} "
            f"-> glued {metrics['fixed_signal_glued_snr']:.4g} "
            f"-> trained {metrics['fixed_signal_deformed_snr']:.4g}, "
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
