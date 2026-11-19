"""High-level orchestration for complex-scalar OPF analyses."""

from __future__ import annotations

import argparse

import numpy as np

import CSFT_utils as csft
from actions import ComplexScalarAction
from observables import ComplexScalarOnePtFn
from training import plot_exported_training

from .config import DATA_DIRECTORY, LOSS_FUNCTIONS
from .objectives import (
    train_field_dependent_variance_loss,
    train_penalty_loss,
    train_variance_loss,
)
from .optimization import fit_lattice_laplace_profile, run_saddle_point
from .persistence import (
    am_key,
    delete_am_training_data,
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
    """Train or plot complex-scalar one-point deformations across dimensions."""
    loss_functions = (
        [args.loss_function]
        if args.loss_function is not None
        else list(LOSS_FUNCTIONS)
    )
    requested_masses = sorted({am_key(am) for am in args.ams})
    delete_masses = sorted(
        {am_key(am) for am in getattr(args, "delete_ams", [])}
    )
    if delete_masses:
        _delete_results(args, loss_functions, delete_masses)
        return

    dimension_specs = sorted(
        (int(dim), int(length)) for dim, length in zip(args.dims, args.Ls)
    )
    dims = [dim for dim, _ in dimension_specs]
    lattice_sizes = [str(length) for _, length in dimension_specs]
    variance_ratios = {loss: [] for loss in loss_functions}
    source_fields = {loss: [] for loss in loss_functions}
    masses_by_loss = {loss: [] for loss in loss_functions}

    for index, ndim in enumerate(dims):
        print("\n" + "=" * 50)
        print(f"complex scalar dimension {ndim}")
        if args.run:
            _run_dimension(
                args=args,
                ndim=ndim,
                lattice_size=lattice_sizes[index],
                requested_masses=requested_masses,
                loss_functions=loss_functions,
                variance_ratios=variance_ratios,
                source_fields=source_fields,
                masses_by_loss=masses_by_loss,
            )
        else:
            _load_dimension(
                args=args,
                ndim=ndim,
                loss_functions=loss_functions,
                variance_ratios=variance_ratios,
                source_fields=source_fields,
                masses_by_loss=masses_by_loss,
            )

    output_path = getattr(args, "save_plot", None)
    plot_scaling(
        dims,
        lattice_sizes,
        masses_by_loss,
        variance_ratios,
        source_fields,
        loss_functions,
        output_path=output_path,
        show=output_path is None,
    )


def _delete_results(
    args: argparse.Namespace,
    loss_functions: list[str],
    masses: list[float],
) -> None:
    for ndim in sorted({int(dim) for dim in args.dims}):
        for loss in loss_functions:
            deleted, removed = delete_am_training_data(ndim, loss, masses)
            for path in deleted:
                print(f"deleted {path}")
            for am in removed:
                print(f"removed am={am:g} from {ratio_savename(ndim, loss)}")


def _load_dimension(
    *,
    args: argparse.Namespace,
    ndim: int,
    loss_functions: list[str],
    variance_ratios: dict[str, list[np.ndarray]],
    source_fields: dict[str, list[np.ndarray]],
    masses_by_loss: dict[str, list[list[float]]],
) -> None:
    for loss in loss_functions:
        artifacts = training_files_for_dimension(ndim, loss)
        if not artifacts:
            raise FileNotFoundError(
                f"no saved complex-scalar artifacts for dimension {ndim}, loss={loss}"
            )
        masses = [am for am, _ in artifacts]
        masses_by_loss[loss].append(masses)
        variance_ratios[loss].append(
            load_variance_ratios(
                ratio_savename(ndim, loss), ndim, masses, loss
            )
        )
        source_fields[loss].append(
            load_deformations_at_source(ndim, loss, masses)
        )
        if args.plot_training and loss not in {"saddle-point", "approx3"}:
            for am, artifact in artifacts:
                plot_exported_training(
                    artifact,
                    show=False,
                    plot_label=rf"$d={ndim},\ am={am:g}$" + f"\nloss={loss}",
                )


def _run_dimension(
    *,
    args: argparse.Namespace,
    ndim: int,
    lattice_size: str,
    requested_masses: list[float],
    loss_functions: list[str],
    variance_ratios: dict[str, list[np.ndarray]],
    source_fields: dict[str, list[np.ndarray]],
    masses_by_loss: dict[str, list[list[float]]],
) -> None:
    if requested_masses:
        ensembles = [
            (
                am,
                DATA_DIRECTORY
                / f"csft_dim{ndim}_am{am:.3f}_L{lattice_size}_ens.npy",
            )
            for am in requested_masses
        ]
    else:
        ensembles = ensemble_files_for_dimension(ndim, lattice_size)
    if not ensembles:
        raise FileNotFoundError(
            f"no csft_dim{ndim}_am*_L{lattice_size}_ens.npy files in "
            f"{DATA_DIRECTORY}"
        )

    source = (0,) * ndim
    stored: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for am, ensemble_file in ensembles:
        if not ensemble_file.exists():
            raise FileNotFoundError(ensemble_file)
        fields, _ = csft.load_ensemble(ensemble_file, expected_am=am)
        latsize = fields.shape[1:]
        action = ComplexScalarAction(am, ndim)
        obs = ComplexScalarOnePtFn(latsize, source, action)
        obs_str = f"scalar OPF {source}, am={am:g}"
        split_mask = (
            make_split_mask(fields)
            if any(loss not in {"saddle-point", "approx3"} for loss in loss_functions)
            else None
        )
        for loss in loss_functions:
            saved = training_savename(ndim, am, loss)
            common = dict(
                fields=fields,
                ndim=ndim,
                am=am,
                obs=obs,
                obs_str=obs_str,
                args=args,
                source=source,
                split_mask=split_mask,
            )
            if loss == "approx3":
                result = fit_lattice_laplace_profile(
                    fields,
                    am,
                    source,
                    mask_radius=args.mask_radius,
                    alpha_bounds=tuple(args.alpha_bounds),
                    alpha_grid_points=args.alpha_grid_points,
                    alpha_xatol=args.alpha_xatol,
                )
                ratio = float(result["variance_ratio"])
                scaled = np.asarray(result["scaled_field"])
                metadata = {
                    "loss_function": "approx3",
                    "objective_name": "complex-scalar Approximation 3 lattice Laplace",
                    "ndim": int(ndim),
                    "am": float(am),
                    "m2": float(am) ** 2,
                    "kappa": 0.5,
                    "lattice_size": tuple(int(length) for length in latsize),
                    "source": source,
                    "field_at_source": float(scaled[*source, 0]),
                    "mask_radius": int(args.mask_radius),
                    "configuration_count": int(len(fields)),
                }
                saved.parent.mkdir(parents=True, exist_ok=True)
                np.save(
                    saved,
                    {
                        "metadata": metadata,
                        "loss_function": "approx3",
                        "objective_name": metadata["objective_name"],
                        "optimization_diagnostics": result[
                            "optimization_diagnostics"
                        ],
                        "f_x": np.asarray(result["f_x"]),
                        "learned_field": np.asarray(result["f_x"]),
                        "alpha_scale": float(result["alpha_scale"]),
                        "scaled_field": scaled,
                        "variance_ratio": ratio,
                    },
                    allow_pickle=True,
                )
            elif loss == "saddle-point":
                ratio = run_saddle_point(**common, result_savename=saved)
            else:
                function = {
                    "variance": train_variance_loss,
                    "penalty": train_penalty_loss,
                    "field-dependent": train_field_dependent_variance_loss,
                }[loss]
                ratio = function(**common, training_savename=saved)
                if args.plot_training:
                    plot_exported_training(
                        saved,
                        show=False,
                        plot_label=rf"$d={ndim},\ am={am:g}$" + f"\nloss={loss}",
                    )
            stored[loss] = upsert_variance_ratios(
                ratio_savename(ndim, loss),
                ndim=ndim,
                lattice_size=int(lattice_size),
                loss_function=loss,
                updates={am: ratio},
            )

    for loss in loss_functions:
        masses, ratios = stored[loss]
        masses_by_loss[loss].append(masses.tolist())
        variance_ratios[loss].append(ratios)
        source_fields[loss].append(
            load_deformations_at_source(ndim, loss, masses)
        )
