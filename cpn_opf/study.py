"""Orchestration for CP(N) OPF scaling studies."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from actions import CPnGaugeAction, CPnQuarticAction
from observables import CPnOnePtFn
from training import plot_exported_training

from .config import LOSS_FUNCTIONS, TRAINING_DATA_DIRECTORY
from .loading import load_approx3_ensemble, load_ensemble
from .objectives import (
    solve_approx2_loss,
    solve_approx3_loss,
    train_penalty_loss,
    train_variance_loss,
)
from .persistence import (
    beta_key,
    delete_beta_training_data,
    ensemble_files_for_dimension,
    ensemble_savename,
    load_deformations_at_source,
    load_variance_ratios,
    ratio_savename,
    training_files_for_dimension,
    training_savename,
    upsert_variance_ratios,
)
from .plotting import plot_deformation_fields, plot_scaling
from .training import make_split_mask


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device or select CUDA/MPS/CPU automatically."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _plot_savename(args: argparse.Namespace) -> Path:
    if args.plot_file is not None:
        return Path(args.plot_file)
    return TRAINING_DATA_DIRECTORY / (
        f"cp{args.N}_opf_i{args.i}_j{args.j}_action-{args.action}_scaling.png"
    )


def _field_plot_savename(
    args: argparse.Namespace, ndim: int, lattice_size: int, loss_function: str
) -> Path:
    """Return a distinct field-plot name for one dimension and objective."""
    stem = (
        f"cp{args.N}_opf_i{args.i}_j{args.j}_action-{args.action}_"
        f"dim{ndim}_L{lattice_size}_loss-{loss_function}_field.png"
    )
    if args.plot_file is not None:
        requested = Path(args.plot_file)
        return requested.with_name(f"{requested.stem}_{stem}")
    return TRAINING_DATA_DIRECTORY / stem


def _ensemble_file_for_beta(
    n: int,
    action_type: str,
    ndim: int,
    beta: float,
    lattice_size: int,
) -> Path:
    """Find a requested action-specific ensemble for a selected beta.

    Canonical filenames use ``b{beta:.2f}``. Discovery also accepts harmless
    differences in a pre-existing beta spelling.
    """
    canonical = ensemble_savename(n, ndim, beta, lattice_size, action_type)
    if canonical.exists():
        return canonical
    matches = [
        path
        for saved_beta, path in ensemble_files_for_dimension(
            n, ndim, lattice_size, action_type
        )
        if np.isclose(saved_beta, beta, rtol=1e-9, atol=1e-12)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"multiple ensembles match beta={beta:g}: "
            + ", ".join(str(path) for path in matches)
        )
    return canonical


def _approx3_signature(
    args: argparse.Namespace,
    *,
    ndim: int,
    lattice_size: int,
    beta: float,
    ensemble_file: Path,
) -> dict[str, object]:
    """Fingerprint inputs that affect a saved Approximation 3 fit."""
    stat = ensemble_file.stat()
    return {
        "cp_n": int(args.N),
        "ndim": int(ndim),
        "beta": float(beta),
        "lattice_size": [int(lattice_size)] * ndim,
        "source": [0] * ndim,
        "i": int(args.i),
        "j": int(args.j),
        "mask_radius": int(args.mask_radius),
        "decorrelation_factor": float(args.decorrelation),
        "max_configurations": int(args.approx3_max_configurations),
        "alpha_bounds": [float(value) for value in args.alpha_bounds],
        "alpha_grid_points": int(args.alpha_grid_points),
        "alpha_xatol": float(args.alpha_xatol),
        "ensemble": {
            "path": str(ensemble_file.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
    }


def _cached_approx3_ratio(
    result_file: Path,
    signature: dict[str, object],
) -> float | None:
    """Return a matching finite saved ratio, or ``None`` for stale results."""
    if not result_file.exists():
        return None
    try:
        payload = np.load(result_file, allow_pickle=True).item()
        if payload.get("metadata", {}).get("approx3_signature") != signature:
            return None
        ratio = float(payload["variance_ratio"])
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return ratio if np.isfinite(ratio) else None


def run_analysis(args: argparse.Namespace) -> None:
    """Train or load CP(N) OPF scaling results and produce the summary plot."""
    loss_functions = (
        [args.loss_function]
        if args.loss_function is not None
        else [
            loss
            for loss in LOSS_FUNCTIONS
            if args.action == "gauge" or loss not in {"approx2", "approx3"}
        ]
    )
    requested_betas = sorted({beta_key(beta) for beta in args.betas})
    dimension_specs = sorted(zip(args.dims, args.Ls))
    if args.delete_betas:
        _delete_results(args, dimension_specs, loss_functions)
        return

    variance_ratios = {loss: [] for loss in loss_functions}
    deformations_at_source = {loss: [] for loss in loss_functions}
    betas_by_loss = {loss: [] for loss in loss_functions}
    for ndim, lattice_size in dimension_specs:
        print("\n" + "=" * 50)
        print(
            f"CP({args.N}), {args.action} action, "
            f"dimension {ndim}, L={lattice_size}"
        )
        if args.run:
            _run_dimension(
                args,
                ndim,
                lattice_size,
                requested_betas,
                loss_functions,
                variance_ratios,
                deformations_at_source,
                betas_by_loss,
            )
        else:
            _load_dimension(
                args,
                ndim,
                lattice_size,
                loss_functions,
                variance_ratios,
                deformations_at_source,
                betas_by_loss,
            )

    if args.save_plot or args.show:
        plot_scaling(
            [spec[0] for spec in dimension_specs],
            [spec[1] for spec in dimension_specs],
            betas_by_loss,
            variance_ratios,
            deformations_at_source,
            loss_functions,
            n=args.N,
            i=args.i,
            j=args.j,
            action_type=args.action,
            filename=_plot_savename(args) if args.save_plot else None,
            save_legend=args.save_legend,
            show=False,
        )
    if args.plot_field:
        _plot_fields(args, dimension_specs, loss_functions)
    # Figures must all exist before the GUI event loop starts.  Calling
    # ``show`` from each plotting helper blocks after the first window.
    if args.show:
        plt.show()


def _plot_fields(
    args: argparse.Namespace,
    dimension_specs: list[tuple[int, int]],
    loss_functions: list[str],
) -> None:
    """Render saved full deformation fields, grouped by dimension and loss."""
    for ndim, lattice_size in dimension_specs:
        for loss_function in loss_functions:
            artifacts = training_files_for_dimension(
                args.N, ndim, lattice_size, args.i, args.j,
                loss_function, args.action,
            )
            plot_deformation_fields(
                artifacts,
                n=args.N,
                ndim=ndim,
                lattice_size=lattice_size,
                i=args.i,
                j=args.j,
                loss_function=loss_function,
                action_type=args.action,
                filename=(
                    _field_plot_savename(args, ndim, lattice_size, loss_function)
                    if args.save_plot else None
                ),
                show=False,
            )


def _delete_results(
    args: argparse.Namespace,
    dimension_specs: list[tuple[int, int]],
    loss_functions: list[str],
) -> None:
    """Delete exactly the requested per-beta CP(N) artifacts."""
    betas = sorted({beta_key(beta) for beta in args.delete_betas})
    for ndim, lattice_size in dimension_specs:
        for loss_function in loss_functions:
            deleted, removed = delete_beta_training_data(
                args.N,
                ndim,
                lattice_size,
                args.i,
                args.j,
                betas,
                loss_function,
                args.action,
            )
            for path in deleted:
                print(f"deleted {path}")
            for beta in removed:
                print(f"removed beta={beta:g} from CP(N) variance-ratio index")
            if not deleted and not removed:
                print(
                    f"no saved CP(N) {args.action} data for "
                    f"d={ndim}, L={lattice_size}"
                )


def _load_dimension(
    args: argparse.Namespace,
    ndim: int,
    lattice_size: int,
    loss_functions: list[str],
    variance_ratios: dict[str, list[np.ndarray]],
    deformations_at_source: dict[str, list[np.ndarray]],
    betas_by_loss: dict[str, list[list[float]]],
) -> None:
    """Load one saved CP(N) scaling series."""
    for loss_function in loss_functions:
        artifacts = training_files_for_dimension(
            args.N,
            ndim,
            lattice_size,
            args.i,
            args.j,
            loss_function,
            args.action,
        )
        if not artifacts:
            raise FileNotFoundError(
                "no saved CP(N) OPF artifacts for "
                f"N={args.N}, d={ndim}, L={lattice_size}, i={args.i}, j={args.j}"
            )
        betas = [beta for beta, _ in artifacts]
        betas_by_loss[loss_function].append(betas)
        variance_ratios[loss_function].append(
            load_variance_ratios(
                ratio_savename(
                    args.N,
                    ndim,
                    lattice_size,
                    args.i,
                    args.j,
                    loss_function,
                    args.action,
                ),
                betas,
            )
        )
        deformations_at_source[loss_function].append(
            load_deformations_at_source(
                args.N,
                ndim,
                lattice_size,
                args.i,
                args.j,
                betas,
                loss_function,
                args.action,
            )
        )
        if args.plot_training and loss_function not in {"approx2", "approx3"}:
            for beta, artifact in artifacts:
                plot_exported_training(
                    artifact,
                    fname=artifact.with_suffix(".png") if args.save_plot else None,
                    show=False,
                    plot_label=(
                        rf"$\mathrm{{CP}}({args.N}),\ d={ndim},\ \beta={beta:g}$"
                        + f"\n$i={args.i},\\ j={args.j}$, loss={loss_function}"
                    ),
                )


def _run_dimension(
    args: argparse.Namespace,
    ndim: int,
    lattice_size: int,
    requested_betas: list[float],
    loss_functions: list[str],
    variance_ratios: dict[str, list[np.ndarray]],
    deformations_at_source: dict[str, list[np.ndarray]],
    betas_by_loss: dict[str, list[list[float]]],
) -> None:
    """Train every selected ensemble for one CP(N) lattice geometry."""
    if requested_betas:
        ensembles = [
            (
                beta,
                _ensemble_file_for_beta(
                    args.N, args.action, ndim, beta, lattice_size
                ),
            )
            for beta in requested_betas
        ]
    else:
        ensembles = ensemble_files_for_dimension(
            args.N, ndim, lattice_size, args.action
        )
        if not ensembles:
            raise FileNotFoundError(
                "no CP(N) ensembles found matching "
                f"cp{args.N}_{args.action}_dim{ndim}_b*_L{lattice_size}_ens.npy"
            )

    device = resolve_device(args.device)
    source = (0,) * ndim
    stored: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for beta, ensemble_file in ensembles:
        if not ensemble_file.exists():
            raise FileNotFoundError(ensemble_file)
        approx3_signature = (
            _approx3_signature(
                args,
                ndim=ndim,
                lattice_size=lattice_size,
                beta=beta,
                ensemble_file=ensemble_file,
            )
            if "approx3" in loss_functions
            else None
        )
        if loss_functions == ["approx3"]:
            approx3_result = training_savename(
                args.N,
                ndim,
                beta,
                lattice_size,
                args.i,
                args.j,
                "approx3",
                args.action,
            )
            cached_ratio = (
                None
                if getattr(args, "recompute_approx3", False)
                else _cached_approx3_ratio(
                    approx3_result, approx3_signature
                )
            )
            if cached_ratio is not None:
                stored["approx3"] = upsert_variance_ratios(
                    ratio_savename(
                        args.N,
                        ndim,
                        lattice_size,
                        args.i,
                        args.j,
                        "approx3",
                        args.action,
                    ),
                    n=args.N,
                    ndim=ndim,
                    lattice_size=lattice_size,
                    i=args.i,
                    j=args.j,
                    loss_function="approx3",
                    updates={beta: cached_ratio},
                    action_type=args.action,
                )
                print(
                    f"reusing saved Approximation 3 result for beta={beta:g}: "
                    f"variance ratio={cached_ratio:.8g}"
                )
                continue

        if loss_functions == ["approx3"]:
            fields, links, _ = load_approx3_ensemble(
                ensemble_file,
                n=args.N,
                ndim=ndim,
                lattice_size=lattice_size,
                beta=beta,
                i=args.i,
                j=args.j,
                decorrelation_factor=args.decorrelation,
                max_configurations=args.approx3_max_configurations,
            )
            saved_beta = None
        else:
            fields, links, saved_beta = load_ensemble(
                ensemble_file,
                n=args.N,
                ndim=ndim,
                lattice_size=lattice_size,
                beta=beta,
                action_type=args.action,
                i=args.i,
                j=args.j,
                decorrelation_factor=args.decorrelation,
            )
        if saved_beta is not None and not np.isclose(saved_beta, beta):
            warnings.warn(
                f"{ensemble_file} stores beta={saved_beta:g}, but its filename "
                f"selects beta={beta:g}; using the filename value",
                stacklevel=2,
        )
        fields = fields.to(device)
        if links is not None:
            links = links.to(device)
        latsize = tuple(int(length) for length in fields.shape[1:-1])
        action_class = (
            CPnGaugeAction if args.action == "gauge" else CPnQuarticAction
        )
        action = action_class(beta, args.N, latsize, device=device)
        obs = CPnOnePtFn(
            args.N,
            latsize,
            i=args.i,
            j=args.j,
            source=source,
            action=action,
            device=device,
        )
        split_mask = make_split_mask(fields)
        for loss_function in loss_functions:
            result = training_savename(
                args.N,
                ndim,
                beta,
                lattice_size,
                args.i,
                args.j,
                loss_function,
                args.action,
            )
            training_kwargs = {}
            if loss_function == "approx3":
                training_kwargs["approx3_signature"] = approx3_signature
            initialize_variance = (
                loss_function == "variance"
                and args.variance_init_from_approx2
            )
            initialize_penalty = (
                loss_function == "penalty"
                and args.penalty_init_from_approx2
            )
            if initialize_variance or initialize_penalty:
                reference_path = training_savename(
                    args.N,
                    ndim,
                    beta,
                    lattice_size,
                    args.i,
                    args.j,
                    "approx2",
                    args.action,
                )
                if not reference_path.exists():
                    raise FileNotFoundError(
                        "Approximation 2 initialization requested but no "
                        f"saved result exists: {reference_path}"
                    )
                reference_data = np.load(
                    reference_path, allow_pickle=True
                ).item()
                reference_metadata = reference_data.get("metadata", {})
                saved_beta = reference_metadata.get("beta")
                if saved_beta is not None and not np.isclose(saved_beta, beta):
                    raise ValueError(
                        f"{reference_path} stores beta={saved_beta:g}, "
                        f"expected {beta:g}"
                    )
                field_key = "scaled_field" if initialize_variance else "learned_field"
                if field_key not in reference_data:
                    raise ValueError(
                        f"{reference_path} has no saved {field_key}"
                    )
                training_kwargs["initial_field"] = np.asarray(
                    reference_data[field_key], dtype=np.float32
                )
            train_function = {
                "variance": train_variance_loss,
                "penalty": train_penalty_loss,
                "approx2": solve_approx2_loss,
                "approx3": solve_approx3_loss,
            }[loss_function]
            ratio = train_function(
                fields,
                links,
                n=args.N,
                action_type=args.action,
                ndim=ndim,
                beta=beta,
                obs=obs,
                args=args,
                source=source,
                i=args.i,
                j=args.j,
                training_savename=result,
                device=device,
                split_mask=split_mask,
                **training_kwargs,
            )
            stored[loss_function] = upsert_variance_ratios(
                ratio_savename(
                    args.N,
                    ndim,
                    lattice_size,
                    args.i,
                    args.j,
                    loss_function,
                    args.action,
                ),
                n=args.N,
                ndim=ndim,
                lattice_size=lattice_size,
                i=args.i,
                j=args.j,
                loss_function=loss_function,
                updates={beta: ratio},
                action_type=args.action,
            )
            if args.plot_training and loss_function not in {"approx2", "approx3"}:
                plot_exported_training(
                    result,
                    fname=result.with_suffix(".png") if args.save_plot else None,
                    show=False,
                    plot_label=(
                        rf"$\mathrm{{CP}}({args.N}),\ d={ndim},\ \beta={beta:g}$"
                        + f"\n$i={args.i},\\ j={args.j}$, loss={loss_function}"
                    ),
                )

    for loss_function in loss_functions:
        stored_betas, stored_ratios = stored[loss_function]
        betas_by_loss[loss_function].append(stored_betas.tolist())
        variance_ratios[loss_function].append(stored_ratios)
        deformations_at_source[loss_function].append(
            load_deformations_at_source(
                args.N,
                ndim,
                lattice_size,
                args.i,
                args.j,
                stored_betas,
                loss_function,
                args.action,
            )
        )
        print(
            f"updated CP({args.N}) d={ndim} variance ratios "
            f"({len(stored_betas)} betas stored)"
        )
