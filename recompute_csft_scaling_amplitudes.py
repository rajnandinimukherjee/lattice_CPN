"""Reoptimize saved complex-scalar Approximation 3 scaling amplitudes.

The Laplace profiles are independent of the alpha bound. Reuse those
profiles, evaluate the exact sampled variance on the original free-field
ensembles, and update the saved results and mass-indexed ratios.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from csft_opf.optimization import _variance_problem, optimize_field_scale
from csft_opf.persistence import ratio_savename, upsert_variance_ratios


BOUNDS = (-40.0, 40.0)


def recompute(path: Path) -> None:
    result = np.load(path, allow_pickle=True).item()
    meta = result["metadata"]
    if tuple(meta.get("alpha_bounds", ())) == BOUNDS:
        print(f"already updated {path}", flush=True)
        return

    dim = int(meta["ndim"])
    mass = float(meta["am"])
    length = int(meta["lattice_size"][0])
    ensemble = Path(f"data/Nd{dim}_L{length}_m{mass:.2f}.npz")
    with np.load(ensemble) as data:
        if not np.isclose(float(data["m"]), mass):
            raise ValueError(f"mass mismatch in {ensemble}")
        fields = torch.from_numpy(
            np.asarray(data["cfgs"], dtype=np.complex64).copy()
        )
    if len(fields) != int(meta["configuration_count"]):
        raise ValueError(f"sample count changed for {path}")

    _, loss = _variance_problem(fields, mass, tuple(meta["source"]))
    profile = torch.as_tensor(result["f_x"], dtype=fields.real.dtype)
    old_alpha = float(result["alpha_scale"])
    old_ratio = float(result["variance_ratio"])
    with torch.no_grad():
        reproduced = float(loss(
            alphas=old_alpha * profile, sampletype="all"
        )) / loss.initial_var
    if not np.isclose(reproduced, old_ratio, rtol=5e-3):
        raise ValueError(
            f"available ensemble does not reproduce {path}: "
            f"{reproduced} versus {old_ratio}"
        )

    alpha, variance = optimize_field_scale(
        profile, loss, BOUNDS, grid_points=41, xatol=1e-4
    )
    ratio = float(variance / loss.initial_var)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"invalid recomputed ratio for {path}")
    if ratio > old_ratio * (1 + 2e-4):
        raise ValueError(f"variance increased for {path}: {old_ratio} -> {ratio}")

    result["alpha_scale"] = float(alpha)
    result["variance_ratio"] = ratio
    result["scaled_field"] = alpha * np.asarray(result["f_x"])
    meta.update(
        alpha_bounds=BOUNDS,
        alpha_grid_points=41,
        alpha_xatol=1e-4,
        field_at_source=float(result["scaled_field"][*(0,) * dim, 0]),
    )
    result["optimization_diagnostics"].update(
        alpha_scale=float(alpha), variance=float(variance), variance_ratio=ratio
    )
    np.save(path, result, allow_pickle=True)
    upsert_variance_ratios(
        ratio_savename(dim, "approx3"), ndim=dim, lattice_size=length,
        loss_function="approx3", updates={mass: ratio},
    )
    print(
        f"{path.name}: alpha {old_alpha:.7g} -> {alpha:.7g}, "
        f"ratio {old_ratio:.7g} -> {ratio:.7g}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    dimensions = (args.dim,) if args.dim else (1, 2, 3)
    for dim in dimensions:
        paths = sorted(Path("training_data").glob(
            f"csft_scaling_dim{dim}_am*_loss-approx3_result.npy"
        ))
        if not paths:
            raise FileNotFoundError(f"no scalar d={dim} results")
        for path in paths:
            recompute(path)


if __name__ == "__main__":
    main()
