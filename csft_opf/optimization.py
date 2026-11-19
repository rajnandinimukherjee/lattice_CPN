"""Field-scale, small-fluctuation, and smooth-field optimizers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import torch

from actions import ComplexScalarAction
from losses import VarianceLoss
from observables import ComplexScalarOnePtFn
from training import periodic_box_mask

from .model import ComplexScalarAngularDeformations
from .training import make_model


def optimize_field_scale(
    f_x: torch.Tensor,
    variance_loss: VarianceLoss,
    bounds: tuple[float, float],
    grid_points: int = 41,
    xatol: float = 1e-4,
) -> tuple[float, float]:
    """Minimize the exact scalar OPF variance over ``alpha * f_x``."""
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
    finite = [index for index, value in enumerate(values) if np.isfinite(value)]
    if not finite:
        raise RuntimeError("scalar variance was non-finite over all alpha values")
    best = min(finite, key=lambda index: (values[index], abs(candidates[index] - 1)))
    if 0 < best < len(candidates) - 1:
        result = minimize_scalar(
            objective,
            bounds=(candidates[best - 1], candidates[best + 1]),
            method="bounded",
            options={"xatol": xatol},
        )
        if result.success and np.isfinite(result.fun):
            candidates.append(float(result.x))
            values.append(float(result.fun))
    best = min(
        range(len(candidates)),
        key=lambda index: (values[index], abs(candidates[index] - 1)),
    )
    return candidates[best], values[best]


def mean_link_coefficient(fields: torch.Tensor) -> float:
    """Estimate ``C_phi=<Re(phi_x conj(phi_{x+mu}))>`` over all links."""
    if not fields.is_complex() or fields.ndim < 2:
        raise ValueError("fields must be complex configurations on a lattice")
    coefficients = []
    for axis in range(1, fields.ndim):
        forward = torch.roll(fields, -1, dims=axis)
        coefficients.append((fields * forward.conj()).real.mean())
    coefficient = float(torch.stack(coefficients).mean().item())
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("the measured scalar mean link coefficient must be positive")
    return coefficient


def saddle_point_objective(
    field: np.ndarray,
    link_coefficient: float,
) -> float:
    """Return the Approximation-2 exponent for a normalized profile."""
    value = 0.0
    for axis in range(field.ndim):
        difference = np.roll(field, -1, axis=axis) - field
        value += np.sum(np.cosh(difference) - 1.0)
    # The theory.py normalization has kappa=1/2, hence 4*kappa*C_phi=2*C_phi.
    return float(2.0 * link_coefficient * value)


def saddle_point_gradient(
    field: np.ndarray,
    link_coefficient: float,
) -> np.ndarray:
    """Differentiate the Approximation-2 exponent over the full lattice."""
    gradient = np.zeros_like(field)
    for axis in range(field.ndim):
        difference = np.roll(field, -1, axis=axis) - field
        link_gradient = np.sinh(difference)
        gradient += -link_gradient + np.roll(link_gradient, 1, axis=axis)
    return 2.0 * link_coefficient * gradient


def solve_saddle_point_field(
    latsize: tuple[int, ...] | torch.Size,
    source: tuple[int, ...],
    link_coefficient: float,
    *,
    maxiter: int,
    mask_radius: int,
    gtol: float,
    ftol: float,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Solve the masked Approximation-2 profile with ``f[source]=1``."""
    shape = tuple(int(length) for length in latsize)
    if not np.isfinite(link_coefficient) or link_coefficient <= 0:
        raise ValueError("link_coefficient must be positive and finite")
    if len(shape) != len(source) or any(length < 1 for length in shape):
        raise ValueError("invalid lattice/source geometry")
    support = periodic_box_mask(shape, source, radius=mask_radius).cpu().numpy()
    free = support.copy()
    free[source] = False

    def unpack(values: np.ndarray) -> np.ndarray:
        field = np.zeros(shape, dtype=np.float64)
        field[source] = 1.0
        field[free] = values
        return field

    initial = np.zeros(int(free.sum()), dtype=np.float64)

    def objective(values: np.ndarray) -> float:
        return saddle_point_objective(unpack(values), link_coefficient)

    def gradient(values: np.ndarray) -> np.ndarray:
        return saddle_point_gradient(unpack(values), link_coefficient)[free]

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
            raise RuntimeError("scalar saddle optimization failed: " + str(result.message))
        field = unpack(result.x)
        diagnostics: dict[str, float | int | bool | str] = {
            "success": True,
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "penalty_exponent": float(result.fun),
            "max_abs_free_gradient": float(
                np.max(np.abs(gradient(result.x)), initial=0.0)
            ),
            "link_coefficient": float(link_coefficient),
        }
    else:
        field = unpack(initial)
        diagnostics = {
            "success": True,
            "message": "no unconstrained sites",
            "iterations": 0,
            "function_evaluations": 1,
            "penalty_exponent": objective(initial),
            "max_abs_free_gradient": 0.0,
            "link_coefficient": float(link_coefficient),
        }
    if field[source] != 1.0 or np.any(field[~support] != 0.0):
        raise RuntimeError("scalar saddle constraints were not preserved")
    return field, diagnostics


def _variance_problem(
    fields: torch.Tensor,
    am: float,
    source: tuple[int, ...],
) -> tuple[ComplexScalarAngularDeformations, VarianceLoss]:
    shape = tuple(int(length) for length in fields.shape[1:])
    model = make_model(fields, len(shape))
    action = ComplexScalarAction(am, len(shape))
    obs = ComplexScalarOnePtFn(shape, source, action)
    return model, VarianceLoss(model, obs)


def run_saddle_point(
    fields: torch.Tensor,
    ndim: int,
    am: float,
    obs: ComplexScalarOnePtFn,
    obs_str: str,
    args: argparse.Namespace,
    source: tuple[int, ...],
    result_savename: str | Path,
    split_mask: torch.Tensor | None = None,
) -> float:
    """Solve Approximation 2 and optimize its scale with the exact variance."""
    del split_mask
    latsize = fields.shape[1:]
    coefficient = mean_link_coefficient(fields)
    saddle_field, diagnostics = solve_saddle_point_field(
        latsize,
        source,
        coefficient,
        maxiter=args.saddle_maxiter,
        gtol=args.saddle_gtol,
        ftol=args.saddle_ftol,
        mask_radius=args.mask_radius,
    )
    f_x = torch.as_tensor(
        saddle_field, dtype=fields.real.dtype, device=fields.device
    ).unsqueeze(-1)
    model = make_model(fields, ndim)
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
    result_path = Path(result_savename)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "loss_function": "saddle-point",
        "objective_name": "complex-scalar Approximation 2",
        "ndim": int(ndim),
        "am": float(am),
        "m2": float(am) ** 2,
        "kappa": 0.5,
        "lattice_size": tuple(int(length) for length in latsize),
        "source": source,
        "field_at_source": float(scaled_field[*source, 0].item()),
        "mask_radius": int(args.mask_radius),
        "mean_link_coefficient": coefficient,
    }
    np.save(
        result_path,
        {
            "metadata": metadata,
            "loss_function": "saddle-point",
            "optimization_diagnostics": diagnostics,
            "f_x": f_x.cpu().numpy(),
            "learned_field": f_x.cpu().numpy(),
            "alpha_scale": float(alpha),
            "scaled_field": scaled_field.cpu().numpy(),
            "variance_ratio": float(variance_ratio),
        },
        allow_pickle=True,
    )
    print(
        f"-----Approximation 2 {obs_str}----------------\n"
        f"C_phi = {coefficient:.8g}\nalpha = {alpha:.8g}\n"
        f"Var: def/undef = {variance_ratio:.8g}"
    )
    return float(variance_ratio)


def solve_lattice_laplace_field(
    latsize: tuple[int, ...] | torch.Size,
    source: tuple[int, ...],
    link_coefficient: float,
    *,
    mask_radius: int,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Solve the masked Approximation-3 lattice Dirichlet problem."""
    shape = tuple(int(length) for length in latsize)
    if not np.isfinite(link_coefficient) or link_coefficient <= 0:
        raise ValueError("link_coefficient must be positive and finite")
    support = periodic_box_mask(shape, source, radius=mask_radius).cpu().numpy()
    source_index = tuple(source)
    free_sites = np.asarray(
        [site for site in np.argwhere(support) if tuple(site) != source_index],
        dtype=np.int64,
    ).reshape(-1, len(shape))
    free_index = np.full(shape, -1, dtype=np.int64)
    for index, site in enumerate(free_sites):
        free_index[tuple(site)] = index

    field = np.zeros(shape, dtype=np.float64)
    field[source_index] = 1.0
    max_abs_residual = 0.0
    if len(free_sites):
        rows: list[int] = []
        columns: list[int] = []
        entries: list[float] = []
        rhs = np.zeros(len(free_sites), dtype=np.float64)
        diagonal = 2.0 * len(shape) * link_coefficient
        for row, site_array in enumerate(free_sites):
            site = tuple(int(value) for value in site_array)
            rows.append(row)
            columns.append(row)
            entries.append(diagonal)
            for axis, length in enumerate(shape):
                for shift in (-1, 1):
                    neighbour_list = list(site)
                    neighbour_list[axis] = (neighbour_list[axis] + shift) % length
                    neighbour = tuple(neighbour_list)
                    if neighbour == source_index:
                        rhs[row] += link_coefficient
                    elif support[neighbour]:
                        rows.append(row)
                        columns.append(int(free_index[neighbour]))
                        entries.append(-link_coefficient)
        laplacian = coo_matrix(
            (entries, (rows, columns)),
            shape=(len(free_sites), len(free_sites)),
        ).tocsc()
        values = np.asarray(spsolve(laplacian, rhs), dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError("scalar Laplace solve returned non-finite values")
        for index, site in enumerate(free_sites):
            field[tuple(site)] = values[index]
        max_abs_residual = float(
            np.max(np.abs(laplacian @ values - rhs), initial=0.0)
        )
    energy = 0.0
    for axis in range(len(shape)):
        difference = np.roll(field, -1, axis=axis) - field
        energy += link_coefficient * float(np.square(difference).sum())
    diagnostics: dict[str, float | int | bool] = {
        "success": True,
        "link_coefficient": float(link_coefficient),
        "support_sites": int(support.sum()),
        "free_sites": int(len(free_sites)),
        "max_abs_laplace_residual": max_abs_residual,
        "quadratic_penalty": energy,
    }
    return field, diagnostics


def fit_lattice_laplace_profile(
    fields: torch.Tensor,
    am: float,
    source: tuple[int, ...],
    *,
    mask_radius: int,
    alpha_bounds: tuple[float, float] = (-5.0, 5.0),
    alpha_grid_points: int = 41,
    alpha_xatol: float = 1e-4,
) -> dict[str, object]:
    """Solve Approximation 3 and optimize its scale on the exact ensemble."""
    shape = tuple(int(length) for length in fields.shape[1:])
    coefficient = mean_link_coefficient(fields)
    field, diagnostics = solve_lattice_laplace_field(
        shape, source, coefficient, mask_radius=mask_radius
    )
    _, variance_loss = _variance_problem(fields, am, source)
    f_x = torch.as_tensor(
        field, dtype=fields.real.dtype, device=fields.device
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
        "f_x": f_x.cpu().numpy(),
        "alpha_scale": float(alpha),
        "scaled_field": scaled_field.cpu().numpy(),
        "variance_ratio": float(variance_ratio),
        "optimization_diagnostics": diagnostics,
    }
