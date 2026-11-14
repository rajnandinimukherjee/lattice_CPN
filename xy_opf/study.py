"""High-level orchestration for XY OPF training and saved-result analysis."""

from __future__ import annotations

import argparse

import numpy as np

import XY_utils as xy
from actions import XYAction
from observables import XYOnePtFn
from training import plot_exported_training

from .config import DATA_DIRECTORY, LOSS_FUNCTIONS
from .objectives import (
    train_field_dependent_variance_loss,
    train_penalty_loss,
    train_variance_loss,
)
from .optimization import fit_lattice_laplace_profile, run_saddle_point
from .persistence import (
    beta_key,
    delete_beta_training_data,
    ensemble_files_for_dimension,
    load_deformations_at_source,
    load_variance_ratios,
    ratio_savename,
    training_files_for_dimension,
    training_savename,
    upsert_variance_ratios,
)
from .plotting import plot_scaling
from .training import make_split_mask


def run_analysis(args: argparse.Namespace) -> None:
    """Train or plot XY one-point deformations across dimensions."""
    loss_functions = (
        [args.loss_function]
        if args.loss_function is not None
        else list(LOSS_FUNCTIONS)
    )
    requested_betas = sorted({beta_key(beta) for beta in args.betas})
    delete_betas = sorted(
        {beta_key(beta) for beta in getattr(args, "delete_betas", [])}
    )
    if delete_betas:
        _delete_results(args, loss_functions, delete_betas)
        return

    dimension_specs = sorted(
        (int(dim), int(length)) for dim, length in zip(args.dims, args.Ls)
    )
    dims = [dimension for dimension, _ in dimension_specs]
    lattice_sizes = [str(length) for _, length in dimension_specs]

    variance_ratios = {loss: [] for loss in loss_functions}
    deformations_at_source = {loss: [] for loss in loss_functions}
    betas_by_loss = {loss: [] for loss in loss_functions}
    training_figures = []

    for dim_idx, ndim in enumerate(dims):
        print("\n" + "=" * 50)
        print(f"dimension {ndim}")
        if args.run:
            _run_dimension(
                args=args,
                ndim=ndim,
                lattice_size=lattice_sizes[dim_idx],
                requested_betas=requested_betas,
                loss_functions=loss_functions,
                variance_ratios=variance_ratios,
                deformations_at_source=deformations_at_source,
                betas_by_loss=betas_by_loss,
                training_figures=training_figures,
            )
        else:
            _load_dimension(
                args=args,
                ndim=ndim,
                loss_functions=loss_functions,
                variance_ratios=variance_ratios,
                deformations_at_source=deformations_at_source,
                betas_by_loss=betas_by_loss,
                training_figures=training_figures,
            )

    output_path = getattr(args, "save_plot", None)
    plot_scaling(
        dims,
        lattice_sizes,
        betas_by_loss,
        variance_ratios,
        deformations_at_source,
        loss_functions,
        output_path=output_path,
        show=output_path is None,
    )


def _delete_results(
    args: argparse.Namespace,
    loss_functions: list[str],
    delete_betas: list[float],
) -> None:
    """Delete selected result artifacts and ratio-index entries."""
    dims = sorted({int(dim) for dim in args.dims})
    for ndim in dims:
        for loss_function in loss_functions:
            deleted_files, removed_betas = delete_beta_training_data(
                ndim, loss_function, delete_betas
            )
            for path in deleted_files:
                print(f"deleted {path}")
            for beta in removed_betas:
                print(
                    f"removed beta={beta:g} from "
                    f"{ratio_savename(ndim, loss_function)}"
                )
            if not deleted_files and not removed_betas:
                print(
                    "no saved data found for "
                    f"dimension {ndim}, loss={loss_function}, "
                    f"betas={delete_betas}"
                )


def _load_dimension(
    *,
    args: argparse.Namespace,
    ndim: int,
    loss_functions: list[str],
    variance_ratios: dict[str, list[np.ndarray]],
    deformations_at_source: dict[str, list[np.ndarray]],
    betas_by_loss: dict[str, list[list[float]]],
    training_figures: list,
) -> None:
    """Load one dimension's previously saved objective artifacts."""
    for loss_function in loss_functions:
        training_files = training_files_for_dimension(ndim, loss_function)
        if not training_files:
            raise FileNotFoundError(
                "no saved objective artifacts found for "
                f"dimension {ndim}, loss={loss_function}"
            )

        dimension_betas = [beta for beta, _ in training_files]
        betas_by_loss[loss_function].append(dimension_betas)
        variance_ratios[loss_function].append(
            load_variance_ratios(
                ratio_savename(ndim, loss_function),
                ndim,
                dimension_betas,
                loss_function,
            )
        )
        deformations_at_source[loss_function].append(
            load_deformations_at_source(
                ndim,
                loss_function,
                dimension_betas,
            )
        )
        if args.plot_training and loss_function != "saddle-point":
            for beta, saved_training in training_files:
                training_figures.append(
                    plot_exported_training(
                        saved_training,
                        show=False,
                        plot_label=(
                            rf"$d={ndim},\ \beta={beta:g}$"
                            + f"\nloss={loss_function}"
                        ),
                    )
                )


def _run_dimension(
    *,
    args: argparse.Namespace,
    ndim: int,
    lattice_size: str,
    requested_betas: list[float],
    loss_functions: list[str],
    variance_ratios: dict[str, list[np.ndarray]],
    deformations_at_source: dict[str, list[np.ndarray]],
    betas_by_loss: dict[str, list[list[float]]],
    training_figures: list,
) -> None:
    """Run every requested objective for one dimension."""
    if requested_betas:
        ensemble_files = [
            (
                beta,
                DATA_DIRECTORY / (
                    f"xy_dim{ndim}_b{beta:.2f}_L{lattice_size}_ens.npy"
                ),
            )
            for beta in requested_betas
        ]
    else:
        ensemble_files = ensemble_files_for_dimension(ndim, lattice_size)
        if not ensemble_files:
            raise FileNotFoundError(
                "no ensemble files found for "
                f"dimension {ndim}, L={lattice_size}: "
                f"{DATA_DIRECTORY}/xy_dim{ndim}_b*_L{lattice_size}_ens.npy"
            )

    source = (0,) * ndim
    stored_ratio_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for beta, ensemble_file in ensemble_files:
        fields, _ = xy.load_ensemble(
            str(ensemble_file),
            decorrelation_factor=args.decorrelation,
        )
        latsize = fields.shape[-ndim:]
        action = XYAction(beta, ndim)
        obs = XYOnePtFn(latsize, source, action)
        obs_str = f"OPF {source}"
        split_mask = (
            make_split_mask(fields)
            if any(
                loss not in {"saddle-point", "approx3"}
                for loss in loss_functions
            )
            else None
        )

        for loss_function in loss_functions:
            saved_training = training_savename(
                ndim, beta, loss_function
            )
            common_arguments = dict(
                fields=fields,
                ndim=ndim,
                beta=beta,
                obs=obs,
                obs_str=obs_str,
                args=args,
                source=source,
                split_mask=split_mask,
            )
            if loss_function == "approx3":
                result = fit_lattice_laplace_profile(
                    fields,
                    beta,
                    source,
                    mask_radius=args.mask_radius,
                    alpha_bounds=tuple(args.alpha_bounds),
                    alpha_grid_points=args.alpha_grid_points,
                    alpha_xatol=args.alpha_xatol,
                )
                variance_ratio = float(result["variance_ratio"])
                scaled_field = np.asarray(result["scaled_field"])
                diagnostics = dict(result["optimization_diagnostics"])
                metadata = {
                    "loss_function": "approx3",
                    "objective_name": "Approximation 3 lattice Laplace",
                    "ndim": int(ndim),
                    "beta": float(beta),
                    "lattice_size": tuple(int(length) for length in latsize),
                    "source": tuple(int(coordinate) for coordinate in source),
                    "field_at_source": float(
                        scaled_field[*source, 0]
                    ),
                    "mask_radius": int(args.mask_radius),
                    "configuration_count": int(len(fields)),
                    "alpha_bounds": tuple(
                        float(value) for value in args.alpha_bounds
                    ),
                    "alpha_grid_points": int(args.alpha_grid_points),
                    "alpha_xatol": float(args.alpha_xatol),
                    "optimizer": "coarse grid plus bounded scalar refinement",
                }
                saved_training.parent.mkdir(parents=True, exist_ok=True)
                np.save(
                    saved_training,
                    {
                        "metadata": metadata,
                        "loss_function": "approx3",
                        "objective_name": metadata["objective_name"],
                        "optimization_diagnostics": diagnostics,
                        "f_x": np.asarray(result["f_x"]),
                        "learned_field": np.asarray(result["f_x"]),
                        "alpha_scale": float(result["alpha_scale"]),
                        "scaled_field": scaled_field,
                        "variance_ratio": variance_ratio,
                    },
                    allow_pickle=True,
                )
                print(
                    f"Approximation 3: alpha={result['alpha_scale']:.8g}, "
                    f"variance ratio={variance_ratio:.8g}, "
                    f"Laplace residual="
                    f"{diagnostics['max_abs_laplace_residual']:.3g}; "
                    f"saved {saved_training}"
                )
            elif loss_function == "saddle-point":
                variance_ratio = run_saddle_point(
                    **common_arguments,
                    result_savename=saved_training,
                )
            else:
                train_functions = {
                    "variance": train_variance_loss,
                    "penalty": train_penalty_loss,
                    "field-dependent": train_field_dependent_variance_loss,
                }
                train_function = train_functions[loss_function]
                variance_ratio = train_function(
                    **common_arguments,
                    training_savename=saved_training,
                )
                if args.plot_training:
                    training_figures.append(
                        plot_exported_training(
                            saved_training,
                            show=False,
                            plot_label=(
                                rf"$d={ndim},\ \beta={beta:g}$"
                                + f"\nloss={loss_function}"
                            ),
                        )
                    )
            stored_ratio_data[loss_function] = upsert_variance_ratios(
                ratio_savename(ndim, loss_function),
                ndim=ndim,
                lattice_size=int(lattice_size),
                loss_function=loss_function,
                updates={beta: variance_ratio},
            )

    for loss_function in loss_functions:
        saved_ratios = ratio_savename(ndim, loss_function)
        stored_betas, stored_ratios = stored_ratio_data[loss_function]
        betas_by_loss[loss_function].append(stored_betas.tolist())
        variance_ratios[loss_function].append(stored_ratios)
        deformations_at_source[loss_function].append(
            load_deformations_at_source(
                ndim,
                loss_function,
                stored_betas,
            )
        )
        print(
            f"updated {ndim}-dim {loss_function} variance ratios "
            f"in {saved_ratios} ({len(stored_betas)} betas stored)"
        )
