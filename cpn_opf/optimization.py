"""Numerical post-processing optimizers for CP(N) OPF fields."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import torch

from losses import VarianceLoss
from training import periodic_box_mask


def _approximation_support(
    latsize: tuple[int, ...] | torch.Size,
    source: tuple[int, ...],
    mask_radius: int,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """Validate and return a CP(N) periodic-box support and free-site mask."""
    shape = tuple(int(length) for length in latsize)
    if not shape or any(length < 1 for length in shape):
        raise ValueError("lattice dimensions must be positive")
    if len(shape) != len(source):
        raise ValueError("lattice and source dimensions differ")
    if any(
        not 0 <= int(coordinate) < length
        for coordinate, length in zip(source, shape)
    ):
        raise ValueError("source lies outside the lattice")
    if (
        isinstance(mask_radius, bool)
        or not isinstance(mask_radius, (int, np.integer))
        or mask_radius < 0
    ):
        raise ValueError("mask_radius must be a non-negative integer")
    support = periodic_box_mask(shape, source, radius=mask_radius).cpu().numpy()
    free = support.copy()
    free[tuple(source)] = False
    return shape, support, free


def _opf_source_direction(channels: int, i: int, j: int) -> np.ndarray:
    """Return the unit source vector fixed by a CP(N) one-point OPF."""
    if channels < 1 or not 0 <= i < channels or not 0 <= j < channels:
        raise ValueError("OPF component indices are outside the CP(N) vector")
    direction = np.zeros(channels, dtype=np.float64)
    if i == j:
        direction[i] = 1.0
    else:
        direction[i] = 1.0 / np.sqrt(2.0)
        direction[j] = -1.0 / np.sqrt(2.0)
    return direction


def _initial_scalar_profile(
    shape: tuple[int, ...],
    source: tuple[int, ...],
    support: np.ndarray,
    source_value: float,
) -> np.ndarray:
    """Build a positive source-centred seed inside a periodic box."""
    coordinates = np.indices(shape, dtype=np.int64)
    center = np.asarray(source, dtype=np.int64).reshape(
        -1, *([1] * len(shape))
    )
    lengths = np.asarray(shape, dtype=np.int64).reshape(
        -1, *([1] * len(shape))
    )
    distance = np.minimum((coordinates - center) % lengths, (center - coordinates) % lengths)
    radius = max(int(distance[:, support].max(initial=0)), 1)
    chebyshev_distance = distance.max(axis=0)
    scalar = source_value * (1.0 - chebyshev_distance / (radius + 1.0))
    scalar[~support] = 0.0
    scalar[tuple(source)] = source_value
    return scalar


def _validate_approximation_inputs(
    latsize: tuple[int, ...] | torch.Size,
    source: tuple[int, ...],
    beta: float,
    cp_n: int,
    i: int,
    j: int,
    mask_radius: int,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Validate common inputs for the Approximation 2 and 3 solvers."""
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    if isinstance(cp_n, bool) or not isinstance(cp_n, (int, np.integer)) or cp_n < 1:
        raise ValueError("cp_n must be a positive integer")
    channels = int(cp_n) + 1
    direction = _opf_source_direction(channels, i, j)
    shape, support, free = _approximation_support(
        latsize, source, mask_radius
    )
    return shape, support, free, direction


def solve_approx2_penalty_field(
    latsize: tuple[int, ...] | torch.Size,
    source: tuple[int, ...],
    beta: float,
    *,
    cp_n: int,
    i: int,
    j: int,
    mask_radius: int = 6,
    maxiter: int = 2000,
    gtol: float = 1e-9,
    ftol: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Minimize the nonlinear small-fluctuation (Approximation 2) penalty.

    The constrained source field has unit norm, with equal and opposite
    components for an off-diagonal OPF. Rotational symmetry makes the
    minimizer collinear with that source vector. Writing its radial coordinate
    as ``q = sinh(u)`` turns each link cost into ``cosh(u_x-u_y)-1``; the
    resulting discrete problem is convex in ``u`` and is solved on the free
    sites of the periodic support box.

    The returned field is in the deformation coordinates ``alpha_x`` (so the
    source norm is one), and the diagnostic objective is the exponent
    ``-2 Re Delta S^(2)`` appearing in the paper.
    """
    shape, support, free, direction = _validate_approximation_inputs(
        latsize, source, beta, cp_n, i, j, mask_radius
    )
    if maxiter < 1 or not np.isfinite(gtol) or gtol <= 0:
        raise ValueError("maxiter and gtol must be positive")
    if not np.isfinite(ftol) or ftol <= 0:
        raise ValueError("ftol must be positive and finite")
    source_index = tuple(int(coordinate) for coordinate in source)
    source_q = 1.0 / np.sqrt(cp_n + 1.0)
    source_u = float(np.arcsinh(source_q))
    free_count = int(free.sum())

    def unpack(values: np.ndarray) -> np.ndarray:
        field = np.zeros(shape, dtype=np.float64)
        field[source_index] = source_u
        field[free] = values
        return field

    def objective_and_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
        field = unpack(values)
        objective = 0.0
        gradient = np.zeros_like(field)
        for axis in range(len(shape)):
            difference = field - np.roll(field, -1, axis=axis)
            link_gradient = np.sinh(difference)
            objective += np.sum(np.cosh(difference) - 1.0)
            gradient += link_gradient - np.roll(link_gradient, 1, axis=axis)
        factor = 4.0 * beta
        return factor * objective, (factor * gradient)[free]

    if free_count:
        initial_profile = _initial_scalar_profile(
            shape, source_index, support, source_u
        )
        initial = initial_profile[free]
        result = minimize(
            objective_and_gradient,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, source_u)] * free_count,
            options={"maxiter": maxiter, "gtol": gtol, "ftol": ftol},
        )
        if not result.success:
            raise RuntimeError(
                "Approximation 2 optimization failed: " + str(result.message)
            )
        u_field = unpack(result.x)
        objective, gradient = objective_and_gradient(result.x)
        diagnostics: dict[str, float | int | bool | str] = {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "penalty_exponent": float(objective),
            "max_abs_free_gradient": float(
                np.max(np.abs(gradient), initial=0.0)
            ),
        }
    else:
        u_field = unpack(np.empty(0, dtype=np.float64))
        objective, _ = objective_and_gradient(np.empty(0, dtype=np.float64))
        diagnostics = {
            "success": True,
            "message": "no unconstrained sites",
            "iterations": 0,
            "function_evaluations": 1,
            "penalty_exponent": float(objective),
            "max_abs_free_gradient": 0.0,
        }

    scalar = np.sqrt(cp_n + 1.0) * np.sinh(u_field)
    field = scalar[..., None] * direction
    field[~support] = 0.0
    if not np.allclose(np.linalg.norm(field[source_index]), 1.0, atol=1e-12):
        raise RuntimeError("Approximation 2 source normalization was not preserved")
    return field, diagnostics


def _approx3_objective_and_gradient(
    scalar: np.ndarray,
    beta: float,
) -> tuple[float, np.ndarray]:
    """Evaluate the collinear Approximation 3 metric penalty and gradient."""
    objective = 0.0
    gradient = np.zeros_like(scalar)
    for axis in range(scalar.ndim):
        forward = np.roll(scalar, -1, axis=axis)
        difference = forward - scalar
        denominator = 1.0 + scalar**2
        objective += np.sum(difference**2 / denominator)
        gradient += (
            -2.0 * difference / denominator
            - 2.0 * scalar * difference**2 / denominator**2
            + np.roll(2.0 * difference / denominator, 1, axis=axis)
        )
    factor = 2.0 * beta
    return factor * objective, factor * gradient


def solve_approx3_penalty_field(
    latsize: tuple[int, ...] | torch.Size,
    source: tuple[int, ...],
    beta: float,
    *,
    cp_n: int,
    i: int,
    j: int,
    mask_radius: int = 6,
    maxiter: int = 4000,
    gtol: float = 1e-8,
    ftol: float = 1e-12,
    initial_field: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Minimize the smooth-deformation metric penalty (Approximation 3).

    As for Approximation 2, the link metric is minimized by keeping the field
    aligned with the normalized OPF source vector. The scalar lattice profile
    is optimized numerically with a fixed unit source and zero field outside
    the periodic support box.
    """
    shape, support, free, direction = _validate_approximation_inputs(
        latsize, source, beta, cp_n, i, j, mask_radius
    )
    if maxiter < 1 or not np.isfinite(gtol) or gtol <= 0:
        raise ValueError("maxiter and gtol must be positive")
    if not np.isfinite(ftol) or ftol <= 0:
        raise ValueError("ftol must be positive and finite")
    source_index = tuple(int(coordinate) for coordinate in source)
    source_q = 1.0 / np.sqrt(cp_n + 1.0)

    def unpack(values: np.ndarray) -> np.ndarray:
        scalar = np.zeros(shape, dtype=np.float64)
        scalar[source_index] = source_q
        scalar[free] = values
        return scalar

    free_count = int(free.sum())
    if free_count:
        starts: list[np.ndarray] = []
        taper = _initial_scalar_profile(
            shape, source_index, support, source_q
        )
        starts.append(taper[free])
        if initial_field is not None:
            supplied = np.asarray(initial_field, dtype=np.float64)
            if supplied.shape != (*shape, cp_n + 1):
                raise ValueError(
                    "initial_field must match the lattice and CP(N) channels"
                )
            # Project a supplied Approximation 2 field onto the fixed source
            # direction, then convert alpha to the saddle coordinate q.
            profile = np.tensordot(supplied, direction, axes=([-1], [0]))
            starts.append((profile / np.sqrt(cp_n + 1.0))[free])

        def objective_and_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
            objective, gradient = _approx3_objective_and_gradient(
                unpack(values), beta
            )
            return objective, gradient[free]

        results = [
            minimize(
                objective_and_gradient,
                initial,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "gtol": gtol, "ftol": ftol},
            )
            for initial in starts
        ]
        successful = [result for result in results if result.success]
        if not successful:
            details = "; ".join(str(result.message) for result in results)
            raise RuntimeError("Approximation 3 optimization failed: " + details)
        result = min(successful, key=lambda candidate: float(candidate.fun))
        scalar = unpack(result.x)
        objective, gradient = _approx3_objective_and_gradient(scalar, beta)
        diagnostics: dict[str, float | int | bool | str] = {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "starts": len(starts),
            "penalty_exponent": float(objective),
            "max_abs_free_gradient": float(
                np.max(np.abs(gradient[free]), initial=0.0)
            ),
        }
    else:
        scalar = np.zeros(shape, dtype=np.float64)
        scalar[source_index] = source_q
        objective, _ = _approx3_objective_and_gradient(scalar, beta)
        diagnostics = {
            "success": True,
            "message": "no unconstrained sites",
            "iterations": 0,
            "function_evaluations": 1,
            "starts": 1,
            "penalty_exponent": float(objective),
            "max_abs_free_gradient": 0.0,
        }

    field = (np.sqrt(cp_n + 1.0) * scalar)[..., None] * direction
    field[~support] = 0.0
    if not np.allclose(np.linalg.norm(field[source_index]), 1.0, atol=1e-12):
        raise RuntimeError("Approximation 3 source normalization was not preserved")
    return field, diagnostics


def optimize_field_scale(
    f_x: torch.Tensor,
    variance_loss: VarianceLoss,
    bounds: tuple[float, float],
    grid_points: int = 41,
    xatol: float = 1e-4,
    variance_evaluator: Callable[[float], float] | None = None,
) -> tuple[float, float]:
    """Minimize OPF variance over the scalar family ``alpha * f_x``.

    ``variance_evaluator`` can supply an equivalent specialized evaluation
    for a particular action.  The default continues to evaluate the supplied
    ``VarianceLoss`` directly.
    """
    lower, upper = (float(value) for value in bounds)
    if not lower < upper:
        raise ValueError("alpha bounds must be strictly increasing")
    if grid_points < 3:
        raise ValueError("alpha_grid_points must be at least 3")

    cache: dict[float, float] = {}

    def objective(alpha: float) -> float:
        alpha = float(alpha)
        if alpha not in cache:
            if variance_evaluator is None:
                with torch.no_grad():
                    value = variance_loss(
                        alphas=alpha * f_x, sampletype="all"
                    ).item()
            else:
                value = float(variance_evaluator(alpha))
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
