"""Numerical field-scale and saddle-point optimizers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import torch

from actions import XYAction
from deformations import ThetaDefs
from losses import VarianceLoss
from observables import XYOnePtFn
from training import periodic_box_mask


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
            "observable variance was non-finite over all alpha values"
        )

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


def saddle_point_objective(field: np.ndarray, beta: float) -> float:
    """Return ``-2 S_penalty`` for the saddle-point field."""
    value = 0.0
    for mu in range(field.ndim):
        difference = np.roll(field, -1, axis=mu) - field
        value += 2.0 * beta * np.sum(np.cosh(difference) - 1.0)
    return float(value)


def saddle_point_gradient(
    field: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Differentiate ``saddle_point_objective`` over the full lattice."""
    gradient = np.zeros_like(field)
    for mu in range(field.ndim):
        difference = np.roll(field, -1, axis=mu) - field
        link_gradient = np.sinh(difference)
        gradient += 2.0 * beta * (
            -link_gradient + np.roll(link_gradient, 1, axis=mu)
        )
    return gradient


def solve_saddle_point_field(
    latsize: tuple[int, ...] | torch.Size,
    source: tuple[int, ...],
    beta: float,
    *,
    maxiter: int,
    mask_radius: int,
    gtol: float,
    ftol: float,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Numerically solve for masked ``f_x`` with ``f_x[source] = 1``."""
    shape = tuple(int(length) for length in latsize)
    if beta < 0:
        raise ValueError("SaddlePoint beta must be non-negative")
    if len(shape) != len(source):
        raise ValueError("SaddlePoint lattice and source dimensions differ")
    if any(length < 1 for length in shape):
        raise ValueError("SaddlePoint lattice dimensions must be positive")
    if any(
        not 0 <= coordinate < length
        for coordinate, length in zip(source, shape)
    ):
        raise ValueError("SaddlePoint source lies outside the lattice")
    support = periodic_box_mask(
        shape, source, radius=mask_radius
    ).cpu().numpy()
    free = support.copy()
    free[source] = False

    def unpack(values: np.ndarray) -> np.ndarray:
        field = np.zeros(shape, dtype=np.float64)
        field[source] = 1.0
        field[free] = values
        return field

    initial = np.zeros(int(free.sum()), dtype=np.float64)

    def objective(values: np.ndarray) -> float:
        return saddle_point_objective(unpack(values), beta)

    def gradient(values: np.ndarray) -> np.ndarray:
        return saddle_point_gradient(unpack(values), beta)[free]

    if len(initial):
        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * len(initial),
            options={"maxiter": maxiter, "gtol": gtol, "ftol": ftol},
        )
        if not result.success:
            raise RuntimeError(
                "SaddlePoint optimization failed: " + str(result.message)
            )
        field = unpack(result.x)
        diagnostics: dict[str, float | int | bool | str] = {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "log_objective": float(result.fun),
            "max_abs_free_gradient": float(
                np.max(np.abs(gradient(result.x)), initial=0.0)
            ),
        }
    else:
        field = unpack(initial)
        diagnostics = {
            "success": True,
            "message": "no unconstrained sites",
            "iterations": 0,
            "function_evaluations": 1,
            "log_objective": objective(initial),
            "max_abs_free_gradient": 0.0,
        }

    if field[source] != 1.0 or np.any(field[~support] != 0.0):
        raise RuntimeError("SaddlePoint constraints were not preserved")
    return field, diagnostics


def run_saddle_point(
    fields: torch.Tensor,
    ndim: int,
    beta: float,
    obs: XYOnePtFn,
    obs_str: str,
    args: argparse.Namespace,
    source: tuple[int, ...],
    result_savename: str | Path,
    split_mask: torch.Tensor | None = None,
) -> float:
    """Solve the SaddlePoint field and optimize its full-variance scale."""
    del split_mask
    result_savename = Path(result_savename)
    result_savename.parent.mkdir(parents=True, exist_ok=True)
    latsize = fields.shape[1:]
    saddle_field, diagnostics = solve_saddle_point_field(
        latsize,
        source,
        beta,
        maxiter=args.saddle_maxiter,
        gtol=args.saddle_gtol,
        ftol=args.saddle_ftol,
        mask_radius=args.mask_radius,
    )
    f_x = torch.as_tensor(
        saddle_field, dtype=fields.dtype, device=fields.device
    ).unsqueeze(-1)

    model = ThetaDefs(ndims=ndim)
    model.samples = {"all": fields}
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
    mean, err = variance_loss.get_stats(
        alphas=scaled_field, sampletype="all"
    )

    metadata = {
        "loss_function": "saddle-point",
        "objective_name": "SaddlePoint",
        "ndim": ndim,
        "beta": beta,
        "lattice_size": tuple(int(length) for length in latsize),
        "source": tuple(int(coordinate) for coordinate in source),
        "field_at_source": float(f_x[*source, 0].item()),
        "mask_radius": int(args.mask_radius),
        "optimizer": "L-BFGS-B",
        "objective": "-2*S_penalty_saddle",
    }
    data = {
        "metadata": metadata,
        "loss_function": "saddle-point",
        "objective_name": "SaddlePoint",
        "optimization_diagnostics": diagnostics,
        "f_x": f_x.detach().cpu().numpy(),
        "learned_field": f_x.detach().cpu().numpy(),
        "alpha_scale": float(alpha),
        "scaled_field": scaled_field.detach().cpu().numpy(),
        "variance_ratio": float(variance_ratio),
    }
    np.save(result_savename, data, allow_pickle=True)

    print("-----SaddlePoint " + obs_str + "----------------")
    print(
        f"-2 S_penalty^saddle = {diagnostics['log_objective']:3g}"
        f"\niterations = {diagnostics['iterations']}"
        f"\nalpha = {alpha:3g}"
        f"\nexp = {mean:3g}, std={err:3g}"
        f"\nVar: def/undef = {variance_ratio:3g}"
    )
    return float(variance_ratio)


def solve_lattice_laplace_field(
    latsize: tuple[int, ...] | torch.Size,
    source: tuple[int, ...],
    beta: float,
    *,
    mask_radius: int,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Solve the masked lattice Dirichlet problem with ``f[source] = 1``.

    This is the smooth-field limit of the XY penalty with its homogeneous
    link coefficient.  On free sites the solution obeys ``Delta_lat f = 0``;
    outside the periodic box it is fixed to zero.  The overall positive
    factor ``beta`` is retained in the linear system, although it cancels
    from the normalized profile.
    """
    shape = tuple(int(length) for length in latsize)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("Laplace beta must be positive and finite")
    if (
        isinstance(mask_radius, bool)
        or not isinstance(mask_radius, (int, np.integer))
        or mask_radius < 0
    ):
        raise ValueError("mask_radius must be a non-negative integer")
    if len(shape) != len(source):
        raise ValueError("Laplace lattice and source dimensions differ")
    if not shape or any(length < 1 for length in shape):
        raise ValueError("Laplace lattice dimensions must be positive")
    if any(
        not 0 <= coordinate < length
        for coordinate, length in zip(source, shape)
    ):
        raise ValueError("Laplace source lies outside the lattice")

    support = periodic_box_mask(
        shape, source, radius=mask_radius
    ).cpu().numpy()
    free_sites = np.argwhere(support)
    source_index = tuple(int(coordinate) for coordinate in source)
    free_sites = np.asarray(
        [site for site in free_sites if tuple(site) != source_index],
        dtype=np.int64,
    ).reshape(-1, len(shape))
    free_index = np.full(shape, -1, dtype=np.int64)
    for index, site in enumerate(free_sites):
        free_index[tuple(site)] = index

    field = np.zeros(shape, dtype=np.float64)
    field[source_index] = 1.0
    if len(free_sites):
        rows: list[int] = []
        columns: list[int] = []
        entries: list[float] = []
        rhs = np.zeros(len(free_sites), dtype=np.float64)
        diagonal = 2 * len(shape) * float(beta)

        for row, site_array in enumerate(free_sites):
            site = tuple(int(coordinate) for coordinate in site_array)
            rows.append(row)
            columns.append(row)
            entries.append(diagonal)
            for axis, length in enumerate(shape):
                for shift in (-1, 1):
                    neighbour_list = list(site)
                    neighbour_list[axis] = (
                        neighbour_list[axis] + shift
                    ) % length
                    neighbour = tuple(neighbour_list)
                    if neighbour == source_index:
                        rhs[row] += beta
                    elif support[neighbour]:
                        rows.append(row)
                        columns.append(int(free_index[neighbour]))
                        entries.append(-float(beta))

        laplacian = coo_matrix(
            (entries, (rows, columns)),
            shape=(len(free_sites), len(free_sites)),
        ).tocsc()
        values = np.asarray(spsolve(laplacian, rhs), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Laplace solve returned non-finite values")
        for index, site in enumerate(free_sites):
            field[tuple(site)] = values[index]
        residual = laplacian @ values - rhs
        max_abs_residual = float(np.max(np.abs(residual), initial=0.0))
    else:
        max_abs_residual = 0.0

    if field[source_index] != 1.0 or np.any(field[~support] != 0.0):
        raise RuntimeError("Laplace constraints were not preserved")
    if np.any(field[support] < -1e-10) or np.any(
        field[support] > 1.0 + 1e-10
    ):
        raise RuntimeError("Laplace solution violated the discrete maximum principle")

    energy = 0.0
    for axis in range(len(shape)):
        difference = np.roll(field, -1, axis=axis) - field
        energy += float(beta) * float(np.square(difference).sum())
    diagnostics: dict[str, float | int | bool] = {
        "success": True,
        "beta": float(beta),
        "support_sites": int(support.sum()),
        "free_sites": int(len(free_sites)),
        "max_abs_laplace_residual": max_abs_residual,
        "quadratic_penalty": energy,
    }
    return field, diagnostics


def fit_lattice_laplace_profile(
    fields: torch.Tensor,
    beta: float,
    source: tuple[int, ...],
    *,
    mask_radius: int,
    alpha_bounds: tuple[float, float] = (-5.0, 5.0),
    alpha_grid_points: int = 41,
    alpha_xatol: float = 1e-4,
) -> dict[str, object]:
    """Solve Approximation 3 and optimize its scale on the XY ensemble.

    Returns the normalized Laplace profile ``f_x``, its variance-optimized
    multiplier ``alpha_scale``, the applied field ``scaled_field``, and
    solver/variance diagnostics.  ``fields`` must contain configurations
    from the requested XY geometry and coupling.
    """
    if fields.ndim < 2:
        raise ValueError("fields must contain configurations and a lattice")
    shape = tuple(int(length) for length in fields.shape[1:])
    if len(source) != len(shape):
        raise ValueError("Laplace source and field dimensions differ")

    laplace_field, diagnostics = solve_lattice_laplace_field(
        shape,
        source,
        beta,
        mask_radius=mask_radius,
    )
    model = ThetaDefs(ndims=len(shape))
    model.samples = {"all": fields}
    obs = XYOnePtFn(shape, source, XYAction(beta, len(shape)))
    variance_loss = VarianceLoss(model, obs)
    f_x = torch.as_tensor(
        laplace_field,
        dtype=fields.dtype,
        device=fields.device,
    ).unsqueeze(-1)
    alpha, optimized_variance = optimize_field_scale(
        f_x,
        variance_loss,
        bounds=alpha_bounds,
        grid_points=alpha_grid_points,
        xatol=alpha_xatol,
    )
    scaled_field = alpha * f_x
    variance_ratio = optimized_variance / variance_loss.initial_var
    diagnostics.update(
        alpha_scale=float(alpha),
        variance=float(optimized_variance),
        variance_ratio=float(variance_ratio),
    )
    return {
        "f_x": f_x.detach().cpu().numpy(),
        "alpha_scale": float(alpha),
        "scaled_field": scaled_field.detach().cpu().numpy(),
        "variance_ratio": float(variance_ratio),
        "optimization_diagnostics": diagnostics,
    }
