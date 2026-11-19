"""Learned objectives for complex-scalar one-point deformations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from losses import PenaltyLoss, VarianceLoss
from networks import UNet
from observables import ComplexScalarOnePtFn
from training import init_weights

from .config import BEST_WINDOW
from .optimization import optimize_field_scale
from .persistence import training_savename as objective_training_savename
from .training import (
    make_model,
    make_split_mask,
    run_training,
    save_training_result,
)


class _ResidualFieldDependentUNet(nn.Module):
    """Predict a field-dependent residual around a fixed scalar profile."""

    def __init__(
        self,
        input_dim: int,
        ndim: int,
        hidden_dim: int | None,
        initial_profile: torch.Tensor,
    ) -> None:
        super().__init__()
        self.network = UNet(input_dim, 3, ndim=ndim, hd=hidden_dim)
        self.register_buffer("initial_profile", initial_profile)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        residual = self.network(grid)
        return torch.cat(
            (self.initial_profile + residual[..., :1], residual[..., 1:]),
            dim=-1,
        )


def _load_fixed_profile(
    ndim: int,
    am: float,
    latsize: tuple[int, ...] | torch.Size,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    filename = objective_training_savename(ndim, am, "variance")
    if not filename.exists():
        return torch.zeros((*latsize, 1), dtype=dtype, device=device), "zero profile"
    data = np.load(filename, allow_pickle=True).item()
    field = data.get("scaled_field", data.get("learned_field"))
    if field is None:
        raise ValueError(f"no fixed deformation field stored in {filename}")
    field = np.asarray(field, dtype=np.float32)
    if field.shape == (*tuple(latsize), 1):
        field = field[..., 0]
    if field.shape != tuple(latsize):
        raise ValueError(f"fixed profile in {filename} has shape {field.shape}")
    return (
        torch.as_tensor(field, dtype=dtype, device=device).unsqueeze(-1),
        str(filename),
    )


def train_variance_loss(
    fields: torch.Tensor,
    ndim: int,
    am: float,
    obs: ComplexScalarOnePtFn,
    obs_str: str,
    args: argparse.Namespace,
    source: tuple[int, ...],
    training_savename: str | Path,
    split_mask: torch.Tensor | None = None,
) -> float:
    """Train a masked field by directly minimizing real-OPF variance."""
    split_mask = make_split_mask(fields, split_mask)
    latsize = fields.shape[1:]
    model = make_model(fields, ndim, split_mask)
    loss_fn = VarianceLoss(model, obs)
    alphanet = UNet(1, 1, ndim=ndim, hd=args.unet_hidden_dim)
    alphanet.apply(init_weights)
    nn.init.zeros_(alphanet.output.weight)
    nn.init.zeros_(alphanet.output.bias)
    trainer = run_training(model, loss_fn, alphanet, latsize, source, args)
    field = trainer.best_alphas(BEST_WINDOW)
    variance_ratio = float(
        (loss_fn(alphas=field, sampletype="all") / loss_fn.initial_var).item()
    )
    mean, error = loss_fn.get_stats(alphas=field, sampletype="all")
    print(
        f"-----deformed {obs_str}----------------\n"
        f"exp = {mean:.8g}, std = {error:.8g}\n"
        f"Var: def/undef = {variance_ratio:.8g}"
    )
    save_training_result(
        training_savename,
        trainer,
        loss_function="variance",
        ndim=ndim,
        am=am,
        latsize=latsize,
        source=source,
        learned_field=field,
        alpha_scale=1.0,
        variance_ratio=variance_ratio,
        extra_data={"training_sample_mask": split_mask.cpu().numpy()},
    )
    return variance_ratio


def train_field_dependent_variance_loss(
    fields: torch.Tensor,
    ndim: int,
    am: float,
    obs: ComplexScalarOnePtFn,
    obs_str: str,
    args: argparse.Namespace,
    source: tuple[int, ...],
    training_savename: str | Path,
    split_mask: torch.Tensor | None = None,
) -> float:
    """Train ``F_x(theta)=a_x+b_x sin(theta_x)+c_x cos(theta_x)``."""
    split_mask = make_split_mask(fields, split_mask)
    latsize = fields.shape[1:]
    model = make_model(fields, ndim, split_mask)
    loss_fn = VarianceLoss(model, obs)
    initialization = args.field_dependent_init
    if initialization == "fixed-profile":
        initial_profile, initialization_source = _load_fixed_profile(
            ndim,
            am,
            latsize,
            dtype=fields.real.dtype,
            device=fields.device,
        )
    else:
        initial_profile = torch.zeros(
            (*latsize, 1), dtype=fields.real.dtype, device=fields.device
        )
        initialization_source = "zero profile"
    alphanet = _ResidualFieldDependentUNet(
        1, ndim, args.unet_hidden_dim, initial_profile
    )
    alphanet.apply(init_weights)
    nn.init.zeros_(alphanet.network.output.weight)
    nn.init.zeros_(alphanet.network.output.bias)
    trainer = run_training(model, loss_fn, alphanet, latsize, source, args)
    coefficients = trainer.best_alphas(BEST_WINDOW)
    variance_ratio = float(
        (
            loss_fn(alphas=coefficients, sampletype="all")
            / loss_fn.initial_var
        ).item()
    )
    save_training_result(
        training_savename,
        trainer,
        loss_function="field-dependent",
        ndim=ndim,
        am=am,
        latsize=latsize,
        source=source,
        learned_field=coefficients,
        alpha_scale=1.0,
        variance_ratio=variance_ratio,
        extra_metadata={
            "deformation_type": "configuration-dependent angular shift",
            "field_parameterization": "F_x=a_x+b_x*sin(theta_x)+c_x*cos(theta_x)",
            "jacobian": "diagonal; exact determinant included",
            "initialization": initialization_source,
        },
        extra_data={"training_sample_mask": split_mask.cpu().numpy()},
    )
    print(
        f"-----field-dependent {obs_str}----------------\n"
        f"Var: def/undef = {variance_ratio:.8g}"
    )
    return variance_ratio


def train_penalty_loss(
    fields: torch.Tensor,
    ndim: int,
    am: float,
    obs: ComplexScalarOnePtFn,
    obs_str: str,
    args: argparse.Namespace,
    source: tuple[int, ...],
    training_savename: str | Path,
    split_mask: torch.Tensor | None = None,
) -> float:
    """Learn a normalized Approximation-1 profile, then optimize its scale."""
    split_mask = make_split_mask(fields, split_mask)
    latsize = fields.shape[1:]
    model = make_model(fields, ndim, split_mask)
    penalty_loss = PenaltyLoss(model, obs.action, obs=obs)
    alphanet = UNet(1, 1, ndim=ndim, hd=args.unet_hidden_dim)
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
        learning_rate=1e-3,
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
    variance_ratio = optimized_variance / variance_loss.initial_var
    print(
        f"-----penalty-trained {obs_str}----------------\n"
        f"alpha = {alpha:.8g}\nVar: def/undef = {variance_ratio:.8g}"
    )
    save_training_result(
        training_savename,
        trainer,
        loss_function="penalty",
        ndim=ndim,
        am=am,
        latsize=latsize,
        source=source,
        learned_field=f_x,
        alpha_scale=alpha,
        variance_ratio=variance_ratio,
    )
    return float(variance_ratio)
