from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize_scalar

import XY_utils as xy
from actions import XYAction
from deformations import ThetaDefs
from losses import PenaltyLoss, VarianceLoss
from networks import UNet
from observables import XYOnePtFn
from training import (Trainer,
                      init_weights,
                      periodic_box_mask,
                      plot_exported_training)

SPLIT = 0.8  # training split
BEST_WINDOW = 20
LOSS_FUNCTIONS = ("variance", "penalty")
LOSS_LABELS = {
    "variance": r"trained on $\mathrm{Var}(O_{\mathrm{def}})$",
    "penalty": (
        r"trained on $\langle e^{-2\,\mathrm{Re}\,\Delta S}\rangle$; "
        r"scaled by optimized $\alpha$"
    ),
}


def training_savename(ndim: int, beta: float, loss_function: str) -> Path:
    """Return an objective-specific training-data filename."""
    return Path(
        f"data/xy_scaling_dim{ndim}_b{beta:.2f}_"
        f"loss-{loss_function}_training.npy"
    )


def ratio_savename(ndim: int, loss_function: str) -> Path:
    """Return an objective-specific variance-ratio filename."""
    return Path(
        f"data/xy_scaling_dim{ndim}_loss-{loss_function}_var_ratios.npy"
    )


def load_variance_ratios(
    filename: str | Path,
    ndim: int,
    betas: list[float],
    loss_function: str = "variance",
) -> np.ndarray:
    """Load current or legacy saved variance ratios for selected betas."""
    loaded = np.load(filename, allow_pickle=True)
    if loaded.shape == () and isinstance(loaded.item(), dict):
        data = loaded.item()
        saved_betas = np.asarray(data["betas"], dtype=float)
        saved_ratios = np.asarray(data["variance_ratios"], dtype=float)
    else:
        saved_ratios = np.asarray(loaded, dtype=float)
        training_files = training_files_for_dimension(ndim, loss_function)
        saved_betas = np.asarray(
            [beta for beta, _ in training_files]
        )
        if len(saved_betas) != len(saved_ratios):
            if len(saved_ratios) == len(betas):
                return saved_ratios
            raise ValueError(
                f"cannot associate legacy ratios in {
                    filename} with beta values"
            )

    selected = []
    for beta in betas:
        matches = np.flatnonzero(np.isclose(saved_betas, beta))
        if not len(matches):
            raise ValueError(f"no saved variance ratio for beta={beta:g}")
        selected.append(saved_ratios[matches[0]])
    return np.asarray(selected)


def training_files_for_dimension(
    ndim: int,
    loss_function: str,
) -> list[tuple[float, Path]]:
    """Return objective-specific training files for one dimension."""
    files = []
    pattern = f"xy_scaling_dim{ndim}_b*_loss-{loss_function}_training.npy"
    for path in Path("data").glob(pattern):
        beta_string = path.stem.split("_b", 1)[1].split("_loss-", 1)[0]
        files.append((float(beta_string), path))

    # Variance-only runs made before objective names were added remain
    # readable, but every new run writes the unambiguous filename above.
    if not files and loss_function == "variance":
        for path in Path("data").glob(
            f"xy_scaling_dim{ndim}_b*_training.npy"
        ):
            if "_loss-" in path.stem:
                continue
            beta_string = path.stem.split("_b", 1)[1].rsplit(
                "_training", 1
            )[0]
            files.append((float(beta_string), path))
    return sorted(files)


def ensemble_files_for_dimension(
    ndim: int, lattice_size: str
) -> list[tuple[float, Path]]:
    """Return ensemble files and their beta values for one lattice size."""
    files = []
    for path in Path("data").glob(
        f"xy_dim{ndim}_b*_L{lattice_size}_ens.npy"
    ):
        beta_string = path.stem.split("_b", 1)[1].rsplit("_L", 1)[0]
        files.append((float(beta_string), path))
    return sorted(files)


def plot_scaling(
    dims: list[int],
    Ls: list[str],
    betas_by_loss: dict[str, list[list[float]]],
    variance_ratios: dict[str, list[np.ndarray]],
    loss_functions: list[str],
) -> None:
    """Plot one or both training objectives on each dimension's axes."""

    fig, axes = plt.subplots(nrows=len(dims), sharex=False)
    ax = np.atleast_1d(axes)
    for dim_idx, dim in enumerate(dims):
        all_betas = []
        for loss_function in loss_functions:
            betas = betas_by_loss[loss_function][dim_idx]
            all_betas.extend(betas)
            ax[dim_idx].plot(
                betas,
                variance_ratios[loss_function][dim_idx],
                marker="o",
                ls="--",
                label=LOSS_LABELS[loss_function],
            )
        ax[dim_idx].set_ylabel(r"$d$=" + str(dim) + r", $V/V_0$")
        ax[dim_idx].set_title(r"$L=$" + Ls[dim_idx])
        ax[dim_idx].legend()
        ticks = sorted(set(all_betas))
        ax[dim_idx].set_xticks(ticks)
        ax[dim_idx].set_xticklabels([f"{beta:.2f}" for beta in ticks])

    ax[-1].set_xlabel(r"$\beta$")
    plt.tight_layout()
    plt.show()


def train_variance_loss(
    fields: torch.Tensor,
    ndim: int,
    beta: float,
    obs: XYOnePtFn,
    obs_str: str,
    args: argparse.Namespace,
    source: tuple[int, ...],
    training_savename: str | Path,
    split_mask: torch.Tensor | None = None,
) -> float:
    """Train a deformation by directly minimizing observable variance."""
    split_mask = make_split_mask(fields, split_mask)
    latsize = fields.shape[1:]

    model = ThetaDefs(ndims=ndim)
    model.samples = {
        "all": fields,
        "train": fields[split_mask],
        "test": fields[~split_mask],
    }

    loss_fn = VarianceLoss(model, obs)
    alphanet = UNet(
        input_dim=obs.grid.shape[-1],
        output_dim=1,
        ndim=ndim,
        hd=args.unet_hidden_dim,
    )
    alphanet.apply(init_weights)

    nn.init.zeros_(alphanet.output.weight)
    nn.init.zeros_(alphanet.output.bias)

    mean, err = loss_fn.target_exp, loss_fn.exp_err
    StN_undef = mean / err
    print("---undeformed " + obs_str + "----------------")
    print(f"exp = {mean:3g}," f"std={
          err:3g}," f"\nStN = {StN_undef:3g}")

    trainer = run_training(
        model,
        loss_fn,
        alphanet,
        latsize,
        source,
        args)
    best_alphas = trainer.best_alphas(BEST_WINDOW)

    mean, err = loss_fn.get_stats(
        alphas=best_alphas, sampletype="all")
    StN_def = mean / err
    variance_ratio = float(
        (
            loss_fn(alphas=best_alphas, sampletype="all")
            / loss_fn.initial_var
        ).item()
    )
    print("-----deformed " + obs_str + "----------------")
    print(
        f"exp = {mean:3g}, std={err:3g},"
        f"\nStN = {StN_def:3g}"
        f"\nVar: def/undef = {variance_ratio:3g}"
    )
    save_training_result(
        training_savename,
        trainer,
        loss_function="variance",
        ndim=ndim,
        beta=beta,
        latsize=latsize,
        source=source,
        learned_field=best_alphas,
        alpha_scale=1.0,
        variance_ratio=variance_ratio,
    )
    return variance_ratio


def make_split_mask(
    fields: torch.Tensor,
    split_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a validated train/test mask, generating one when absent."""
    n_cfgs = fields.shape[0]
    if n_cfgs < 2:
        raise ValueError("at least two configurations are required")
    if split_mask is None:
        n_train = min(max(int(SPLIT * n_cfgs), 1), n_cfgs - 1)
        train_idx = torch.randperm(n_cfgs, device=fields.device)[:n_train]
        split_mask = torch.zeros(
            n_cfgs, dtype=torch.bool, device=fields.device
        )
        split_mask[train_idx] = True
    if split_mask.dtype != torch.bool or split_mask.shape != (n_cfgs,):
        raise ValueError(
            "split_mask must be a boolean vector over configurations")
    if not split_mask.any() or split_mask.all():
        raise ValueError("split_mask must contain both train and test samples")
    return split_mask


def run_training(
    model: ThetaDefs,
    loss_fn: nn.Module,
    alphanet: UNet,
    latsize: tuple[int, ...] | torch.Size,
    source: tuple[int, ...],
    args: argparse.Namespace,
    source_constraint: float | None = None,
    learning_rate: float | None = None,
) -> Trainer:
    """Train one UNet with the common optimizer, mask, and stopping rules."""

    learning_rate = learning_rate or args.learning_rate
    optimizer = optim.Adam(alphanet.parameters(), lr=learning_rate)
    trainer = Trainer(
        model, loss_fn, optimizer, args.epochs, args.batch_size
    )
    alpha_mask = periodic_box_mask(
        latsize, source, radius=5
    ).unsqueeze(-1)
    fixed_alphas = (
        {(*source, 0): source_constraint}
        if source_constraint is not None
        else None
    )
    trainer.train(
        alphanet,
        mask=alpha_mask,
        fixed_alphas=fixed_alphas,
        plot=False,
        max_training_time=args.max_training_time,
        plateau_window=args.plateau_window,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
    )
    return trainer


def save_training_result(
    filename: str | Path,
    trainer: Trainer,
    *,
    loss_function: str,
    ndim: int,
    beta: float,
    latsize: tuple[int, ...] | torch.Size,
    source: tuple[int, ...],
    learned_field: torch.Tensor,
    alpha_scale: float,
    variance_ratio: float,
) -> None:
    """Save histories, final fields, and identifying run metadata."""
    data = trainer.export_training_data()
    data.update(
        metadata={
            "loss_function": loss_function,
            "ndim": ndim,
            "beta": beta,
            "lattice_size": tuple(int(length) for length in latsize),
            "source": tuple(int(coordinate) for coordinate in source),
            "field_at_source": float(learned_field[*source, 0].item()),
        },
        loss_function=loss_function,
        learned_field=learned_field.detach().cpu().numpy(),
        alpha_scale=float(alpha_scale),
        scaled_field=(alpha_scale * learned_field).detach().cpu().numpy(),
        variance_ratio=float(variance_ratio),
    )
    if loss_function == "penalty":
        data["f_x"] = learned_field.detach().cpu().numpy()
        data["f_source_constraint"] = 1.0
    np.save(filename, data, allow_pickle=True)


def optimize_field_scale(
    f_x: torch.Tensor,
    variance_loss: VarianceLoss,
    bounds: tuple[float, float],
    grid_points: int = 41,
    xatol: float = 1e-4,
) -> tuple[float, float]:
    """Numerically minimize full observable variance over ``alpha * f_x``."""
    lower, upper = (float(value) for value in bounds)
    if not lower < upper:
        raise ValueError("alpha bounds must be strictly increasing")
    if grid_points < 3:
        raise ValueError("alpha_grid_points must be at least 3")

    cache: dict[float, float] = {}

    def objective(alpha: float) -> float:
        alpha = float(alpha)
        if alpha not in cache:
            with torch.no_grad():
                value = variance_loss(
                    alphas=alpha * f_x, sampletype="all"
                ).item()
            cache[alpha] = value if np.isfinite(value) else np.inf
        return cache[alpha]

    candidates = list(np.linspace(lower, upper, grid_points))
    candidates.extend(
        alpha for alpha in (0.0, 1.0) if lower <= alpha <= upper
    )
    candidates = sorted(set(float(alpha) for alpha in candidates))
    values = [objective(alpha) for alpha in candidates]
    finite_indices = [
        index for index, value in enumerate(values) if np.isfinite(value)
    ]
    if not finite_indices:
        raise RuntimeError(
            "observable variance was non-finite over all alpha values")

    best_index = min(
        finite_indices,
        key=lambda index: (values[index], abs(candidates[index] - 1.0)),
    )
    if 0 < best_index < len(candidates) - 1:
        result = minimize_scalar(
            objective,
            bounds=(candidates[best_index - 1], candidates[best_index + 1]),
            method="bounded",
            options={"xatol": xatol},
        )
        if result.success and np.isfinite(result.fun):
            candidates.append(float(result.x))
            values.append(float(result.fun))

    best_index = min(
        range(len(candidates)),
        key=lambda index: (values[index], abs(candidates[index] - 1.0)),
    )
    return candidates[best_index], values[best_index]


def train_penalty_loss(
    fields: torch.Tensor,
    ndim: int,
    beta: float,
    obs: XYOnePtFn,
    obs_str: str,
    args: argparse.Namespace,
    source: tuple[int, ...],
    training_savename: str | Path,
    split_mask: torch.Tensor | None = None,
) -> float:
    """Learn ``f_x`` with PenaltyLoss, then optimize its scalar multiplier."""
    split_mask = make_split_mask(fields, split_mask)
    latsize = fields.shape[1:]
    model = ThetaDefs(ndims=ndim)
    model.samples = {
        "all": fields,
        "train": fields[split_mask],
        "test": fields[~split_mask],
    }

    penalty_loss = PenaltyLoss(model, obs.action, obs=obs)
    alphanet = UNet(
        input_dim=obs.grid.shape[-1],
        output_dim=1,
        ndim=ndim,
        hd=args.unet_hidden_dim,
    )
    # The hard f(source)=1 constraint below breaks PenaltyLoss's stationary
    # zero-field direction. Keep the remaining unconstrained network output
    # small so the extensive exponential stays finite on large lattices.
    with torch.no_grad():
        alphanet.output.weight.mul_(1e-2)
        nn.init.zeros_(alphanet.output.bias)

    trainer = run_training(
        model,
        penalty_loss,
        alphanet,
        latsize,
        source,
        args,
        source_constraint=1.0,
        learning_rate=1e-3
    )
    f_x = trainer.best_alphas(BEST_WINDOW)

    variance_loss = VarianceLoss(model, obs)
    alpha, optimized_variance = optimize_field_scale(
        f_x,
        variance_loss,
        bounds=tuple(args.alpha_bounds),
        grid_points=args.alpha_grid_points,
        xatol=args.alpha_xatol,
    )
    scaled_field = alpha * f_x
    variance_ratio = optimized_variance / variance_loss.initial_var
    penalty_value = penalty_loss(alphas=f_x, sampletype="all").item()
    mean, err = variance_loss.get_stats(
        alphas=scaled_field, sampletype="all"
    )

    print("-----penalty-trained " + obs_str + "----------------")
    print(
        f"<exp(-2 Re S_diff)> = {penalty_value:3g}"
        f"\nalpha = {alpha:3g}"
        f"\nexp = {mean:3g}, std={err:3g}"
        f"\nVar: def/undef = {variance_ratio:3g}"
    )
    save_training_result(
        training_savename,
        trainer,
        loss_function="penalty",
        ndim=ndim,
        beta=beta,
        latsize=latsize,
        source=source,
        learned_field=f_x,
        alpha_scale=alpha,
        variance_ratio=variance_ratio,
    )
    return float(variance_ratio)


def main(args: argparse.Namespace) -> None:
    """Train or plot XY one-point deformations across dimensions."""

    loss_functions = (
        [args.loss_function]
        if args.loss_function is not None
        else list(LOSS_FUNCTIONS)
    )
    requested_betas = sorted(float(beta) for beta in args.betas)
    dimension_specs = sorted(
        (int(dim), int(length)) for dim, length in zip(args.dims, args.Ls)
    )
    dims = [dimension for dimension, _ in dimension_specs]
    lattice_sizes = [str(length) for _, length in dimension_specs]

    variance_ratios = {loss: [] for loss in loss_functions}
    betas_by_loss = {loss: [] for loss in loss_functions}
    training_figures = []
    for dim_idx, ndim in enumerate(dims):
        print("\n" + "=" * 50)
        print(f"dimension {ndim}")
        if not args.run:
            for loss_function in loss_functions:
                training_files = training_files_for_dimension(
                    ndim, loss_function
                )
                if not training_files:
                    raise FileNotFoundError(
                        "no saved training files found for "
                        f"dimension {ndim}, loss={loss_function}"
                    )

                dimension_betas = [beta for beta, _ in training_files]
                betas_by_loss[loss_function].append(dimension_betas)
                saved_ratios = ratio_savename(ndim, loss_function)
                if not saved_ratios.exists() and loss_function == "variance":
                    saved_ratios = Path(
                        f"data/xy_scaling_dim{ndim}_var_ratios.npy"
                    )
                variance_ratios[loss_function].append(
                    load_variance_ratios(
                        saved_ratios,
                        ndim,
                        dimension_betas,
                        loss_function,
                    )
                )
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
        else:
            if requested_betas:
                ensemble_files = [
                    (
                        beta,
                        Path(
                            f"data/xy_dim{ndim}_b{beta:.2f}_L"
                            f"{lattice_sizes[dim_idx]}_ens.npy"
                        ),
                    )
                    for beta in requested_betas
                ]
            else:
                ensemble_files = ensemble_files_for_dimension(
                    ndim, lattice_sizes[dim_idx]
                )
                if not ensemble_files:
                    raise FileNotFoundError(
                        "no ensemble files found for "
                        f"dimension {ndim}, L={lattice_sizes[dim_idx]}: "
                        f"data/xy_dim{ndim}_b*_L{
                            lattice_sizes[dim_idx]}_ens.npy"
                    )

            dimension_betas = [beta for beta, _ in ensemble_files]
            for loss_function in loss_functions:
                betas_by_loss[loss_function].append(dimension_betas)

            source = (0,) * ndim
            ratios_for_dimension = {
                loss: np.zeros(len(dimension_betas))
                for loss in loss_functions
            }
            for beta_idx, (beta, ensemble_file) in enumerate(ensemble_files):
                fname = str(ensemble_file)
                fields, _ = xy.load_ensemble(
                    fname,
                    decorrelation_factor=args.decorrelation)

                latsize = fields.shape[-ndim:]

                action = XYAction(beta, ndim)
                obs = XYOnePtFn(latsize, source, action)
                obs_str = f"OPF {source}"
                split_mask = make_split_mask(fields)

                for loss_function in loss_functions:
                    saved_training = training_savename(
                        ndim, beta, loss_function
                    )
                    train_function = (
                        train_variance_loss
                        if loss_function == "variance"
                        else train_penalty_loss
                    )
                    variance_ratio = train_function(
                        fields=fields,
                        ndim=ndim,
                        beta=beta,
                        obs=obs,
                        obs_str=obs_str,
                        args=args,
                        source=source,
                        training_savename=saved_training,
                        split_mask=split_mask,
                    )
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
                    ratios_for_dimension[loss_function][beta_idx] = (
                        variance_ratio
                    )

            for loss_function in loss_functions:
                saved_ratios = ratio_savename(ndim, loss_function)
                ratio_data = {
                    "metadata": {
                        "loss_function": loss_function,
                        "ndim": ndim,
                        "lattice_size": int(lattice_sizes[dim_idx]),
                    },
                    "betas": np.asarray(dimension_betas),
                    "variance_ratios": ratios_for_dimension[loss_function],
                }
                np.save(saved_ratios, ratio_data, allow_pickle=True)
                variance_ratios[loss_function].append(
                    ratios_for_dimension[loss_function]
                )
                print(
                    f"saved {ndim}-dim {loss_function} variance ratios "
                    f"to {saved_ratios}"
                )

    if training_figures:
        plt.show()

    plot_scaling(
        dims,
        lattice_sizes,
        betas_by_loss,
        variance_ratios,
        loss_functions,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--betas",
        nargs="+",
        default=[],
        help=(
            "beta values to use for --run, e.g. 1.2 1.3; "
            "omit to discover matching ensemble files"
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
        required=True,
        help="lattice sizes in each dimension, e.g. 64 64 32",
    )
    parser.add_argument("--epochs", help="epochs", type=int, default=200)
    parser.add_argument("--batch-size", help="batch size",
                        type=int, default=64)
    parser.add_argument(
        "--learning-rate", help="learning rate", type=float, default=1e-3
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
        help="rolling validation-objective window used for plateau detection",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=40,
        help="number of non-improving rolling windows before stopping",
    )
    parser.add_argument(
        "--plateau-min-delta",
        type=float,
        default=1e-3,
        help="minimum rolling validation-objective improvement",
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
        choices=LOSS_FUNCTIONS,
        default=None,
        help=(
            "train/plot only one objective; omit to compare variance and "
            "penalty objectives"
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
        "--run",
        help="run training instead of loading saved losses",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()

    if len(args.dims) != len(args.Ls):
        parser.error("--dims and --Ls must contain the same number of values")
    if any(int(dim) not in (1, 2, 3) for dim in args.dims):
        parser.error("--dims values must be 1, 2, or 3")
    if args.alpha_bounds[0] >= args.alpha_bounds[1]:
        parser.error("--alpha-bounds LOWER must be smaller than UPPER")
    if args.alpha_grid_points < 3:
        parser.error("--alpha-grid-points must be at least 3")
    if args.alpha_xatol <= 0:
        parser.error("--alpha-xatol must be positive")
    main(args)
