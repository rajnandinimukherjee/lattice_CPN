"""Fit masked scalar deformations to the sampled real-OPF variance.

Each static field starts at the saved Approximation 3 profile.  The fit uses
the same exact contour-reweighted real observable as ComplexScalarOnePtFn,
with a fixed training/validation split and a best-validation checkpoint.
Raw free-field ensembles are the ``data/Nd*_L*_m*.npz`` files used by the scalar scaling analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from csft_opf.persistence import (
    ratio_savename,
    training_files_for_dimension,
    training_savename,
    upsert_variance_ratios,
)
from training import periodic_box_mask

LATTICE_SIZES = {1: 64, 2: 32, 3: 16}
SEED = 1407


def variance_for_indices(
    configurations: torch.Tensor,
    field: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Exact sampled variance of Re[phi_0 exp(-F_0-Delta S)] for one field."""
    phi = configurations[indices]
    lattice_axes = tuple(range(1, phi.ndim))
    amplitude = torch.zeros(len(phi), dtype=field.dtype)
    phase = torch.zeros_like(amplitude)
    for axis in range(field.ndim):
        difference = field - torch.roll(field, -1, dims=axis)
        link = phi * torch.roll(phi, -1, dims=axis + 1).conj()
        amplitude = amplitude + (
            link.real * (torch.cosh(difference) - 1.0)
        ).sum(dim=lattice_axes)
        phase = phase + (
            link.imag * torch.sinh(difference)
        ).sum(dim=lattice_axes)
    source = (0,) * field.ndim
    source_phi = phi[(slice(None), *source)]
    observable = torch.exp(-field[source] + amplitude) * (
        source_phi.real * torch.cos(phase)
        + source_phi.imag * torch.sin(phase)
    )
    return observable.var()


def fit_one(dim: int, mass: float, *, steps: int, batch_size: int,
            refit: bool = False) -> None:
    length = LATTICE_SIZES[dim]
    ensemble_path = Path(f"data/Nd{dim}_L{length}_m{mass:.2f}.npz")
    approx_path = training_savename(dim, mass, "approx3")
    saved_path = training_savename(dim, mass, "variance")
    if saved_path.exists() and not refit:
        saved_ratio = float(np.load(saved_path, allow_pickle=True).item()["variance_ratio"])
        upsert_variance_ratios(
            ratio_savename(dim, "variance"), ndim=dim, lattice_size=length,
            loss_function="variance", updates={mass: saved_ratio},
        )
        print(f"reusing {saved_path}", flush=True)
        return
    with np.load(ensemble_path) as ensemble:
        configurations = torch.from_numpy(
            np.asarray(ensemble["cfgs"], dtype=np.complex64).copy()
        )
    approx = np.load(approx_path, allow_pickle=True).item()
    metadata = approx["metadata"]
    initial = torch.as_tensor(
        np.asarray(approx["scaled_field"])[..., 0], dtype=torch.float32
    ).clone()
    radius = int(metadata["mask_radius"])
    mask = periodic_box_mask(initial.shape, (0,) * dim, radius=radius).float()
    if tuple(configurations.shape[1:]) != tuple(initial.shape):
        raise ValueError(f"ensemble and Approximation 3 geometry disagree: {mass}")

    generator = torch.Generator().manual_seed(SEED + 1000 * dim + round(1000 * mass))
    permutation = torch.randperm(len(configurations), generator=generator)
    n_training = int(0.8 * len(configurations))
    training_indices = permutation[:n_training]
    validation_indices = permutation[n_training:]
    residual = torch.nn.Parameter(torch.zeros_like(initial))
    learning_rate = {1: 0.015, 2: 0.005, 3: 0.001}[dim]
    optimizer = torch.optim.Adam([residual], lr=learning_rate)
    best_field = initial.clone()
    with torch.no_grad():
        best_validation = float(variance_for_indices(
            configurations, initial, validation_indices
        ))
    history = [(0, best_validation)]
    for step in range(1, steps + 1):
        batch = training_indices[torch.randint(
            len(training_indices), (batch_size,), generator=generator
        )]
        field = initial + residual * mask
        loss = variance_for_indices(configurations, field, batch)
        loss = loss + 1e-4 * (residual * mask).square().sum()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([residual], 10.0)
        optimizer.step()
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                candidate = initial + residual * mask
                validation = float(variance_for_indices(
                    configurations, candidate, validation_indices
                ))
                history.append((step, validation))
                if np.isfinite(validation) and validation < best_validation:
                    best_validation = validation
                    best_field = candidate.clone()
    all_indices = torch.arange(len(configurations))
    with torch.no_grad():
        fitted_variance = float(variance_for_indices(
            configurations, best_field, all_indices
        ))
        source_values = configurations[(slice(None), *((0,) * dim))].real
        initial_variance = float(source_values.var())
    ratio = fitted_variance / initial_variance
    payload = {
        "metadata": {
            "loss_function": "variance",
            "objective_name": "sampled real-OPF variance",
            "ndim": dim,
            "am": mass,
            "lattice_size": tuple(initial.shape),
            "source": (0,) * dim,
            "field_at_source": float(best_field[(0,) * dim]),
            "mask_radius": radius,
            "warm_started_from": str(approx_path),
            "fit_method": "masked static field, Adam, best validation checkpoint",
            "training_configurations": n_training,
            "validation_configurations": len(validation_indices),
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": SEED,
        },
        "loss_function": "variance",
        "learned_field": best_field.unsqueeze(-1).numpy(),
        "scaled_field": best_field.unsqueeze(-1).numpy(),
        "alpha_scale": 1.0,
        "variance_ratio": ratio,
        "validation_history": np.asarray(history),
    }
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(saved_path, payload, allow_pickle=True)
    upsert_variance_ratios(
        ratio_savename(dim, "variance"), ndim=dim, lattice_size=length,
        loss_function="variance", updates={mass: ratio},
    )
    print(f"d={dim}, am={mass:g}: approx3={approx['variance_ratio']:.6g}, "
          f"direct={ratio:.6g}, F0={payload['metadata']['field_at_source']:.6g}",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=(1, 2, 3))
    parser.add_argument("--mass", type=float)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    if args.dim is None and args.mass is not None:
        parser.error("--mass requires --dim")
    dims = (args.dim,) if args.dim is not None else (1, 2, 3)
    for dim in dims:
        masses = [args.mass] if args.mass is not None else [
            mass for mass, _ in training_files_for_dimension(dim, "approx3")
        ]
        for mass in masses:
            fit_one(dim, mass, steps=args.steps, batch_size=args.batch_size,
                    refit=args.refit)


if __name__ == "__main__":
    main()
