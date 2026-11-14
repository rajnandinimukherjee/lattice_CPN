"""Objective-specific neural-training workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from deformations import ThetaDefs
from losses import PenaltyLoss, VarianceLoss
from networks import UNet
from observables import XYOnePtFn
from training import init_weights

from .config import BEST_WINDOW
from .optimization import optimize_field_scale
from .persistence import training_savename as objective_training_savename
from .training import make_split_mask, run_training, save_training_result


class _ResidualFieldDependentUNet(nn.Module):
    """Predict a field-dependent residual around a fixed XY profile."""

    def __init__(
        self,
        input_dim: int,
        ndim: int,
        hidden_dim: int | None,
        initial_profile: torch.Tensor,
    ) -> None:
        super().__init__()
        self.network = UNet(
            input_dim=input_dim,
            output_dim=3,
            ndim=ndim,
            hd=hidden_dim,
        )
        self.register_buffer("initial_profile", initial_profile)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        residual = self.network(grid)
        return torch.cat(
            (
                self.initial_profile + residual[..., :1],
                residual[..., 1:],
            ),
            dim=-1,
        )


def _load_fixed_profile(
    ndim: int,
    beta: float,
    latsize: tuple[int, ...] | torch.Size,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    """Use a saved fixed-field solution as the field-dependent start point."""
    filename = objective_training_savename(ndim, beta, "variance")
    if not filename.exists():
        return (
            torch.zeros((*latsize, 1), dtype=dtype, device=device),
            "zero profile (no saved fixed-field solution)",
        )

    payload = np.load(filename, allow_pickle=True)
    if payload.shape != () or not isinstance(payload.item(), dict):
        raise ValueError(f"invalid fixed-field training artifact: {filename}")
    saved = payload.item()
    metadata = saved.get("metadata", {})
    saved_size = tuple(metadata.get("lattice_size", ()))
    if saved_size != tuple(int(length) for length in latsize):
        raise ValueError(
            f"fixed-field lattice size in {filename} does not match "
            f"{tuple(latsize)}"
        )
    field = saved.get("scaled_field", saved.get("learned_field"))
    if field is None:
        raise ValueError(f"no fixed deformation field stored in {filename}")
    field = np.asarray(field, dtype=np.float32)
    if field.shape == (*tuple(latsize), 1):
        field = field[..., 0]
    if field.shape != tuple(latsize):
        raise ValueError(
            f"fixed deformation field in {filename} has shape {field.shape}"
        )
    return (
        torch.as_tensor(field, dtype=dtype, device=device).unsqueeze(-1),
        str(filename),
    )


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
    signal_to_noise = mean / err
    print("---undeformed " + obs_str + "----------------")
    print(
        f"exp = {mean:3g}, std={err:3g},"
        f"\nStN = {signal_to_noise:3g}"
    )

    trainer = run_training(
        model,
        loss_fn,
        alphanet,
        latsize,
        source,
        args,
    )
    best_alphas = trainer.best_alphas(BEST_WINDOW)

    mean, err = loss_fn.get_stats(
        alphas=best_alphas, sampletype="all"
    )
    signal_to_noise = mean / err
    variance_ratio = float(
        (
            loss_fn(alphas=best_alphas, sampletype="all")
            / loss_fn.initial_var
        ).item()
    )
    print("-----deformed " + obs_str + "----------------")
    print(
        f"exp = {mean:3g}, std={err:3g},"
        f"\nStN = {signal_to_noise:3g}"
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
        extra_data={
            "training_sample_mask": split_mask.detach().cpu().numpy(),
        },
    )
    return variance_ratio


def train_field_dependent_variance_loss(
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
    """Train a masked, configuration-dependent XY deformation.

    A residual U-Net learns three spatial coefficient fields in
    ``F_x(theta) = a_x + b_x sin(theta_x) + c_x cos(theta_x)``. The zero
    profile is the default initialization; an optional fixed-profile start
    uses a saved field-independent variance result. The observable evaluates
    this map on each configuration and includes its exact diagonal Jacobian
    determinant in every weight.
    """
    split_mask = make_split_mask(fields, split_mask)
    latsize = fields.shape[1:]
    model = ThetaDefs(ndims=ndim)
    model.samples = {
        "all": fields,
        "train": fields[split_mask],
        "test": fields[~split_mask],
    }

    loss_fn = VarianceLoss(model, obs)
    initialization = getattr(args, "field_dependent_init", "zero")
    if initialization == "zero":
        initial_profile = torch.zeros(
            (*latsize, 1), dtype=fields.dtype, device=fields.device
        )
        initialization_source = "zero profile; zero-initialized U-Net output"
    elif initialization == "fixed-profile":
        initial_profile, initialization_source = _load_fixed_profile(
            ndim,
            beta,
            latsize,
            dtype=fields.dtype,
            device=fields.device,
        )
    else:
        raise ValueError(
            "field_dependent_init must be 'zero' or 'fixed-profile'"
        )
    print(f"field-dependent initialization: {initialization_source}")
    alphanet = _ResidualFieldDependentUNet(
        input_dim=obs.grid.shape[-1],
        ndim=ndim,
        hidden_dim=args.unet_hidden_dim,
        initial_profile=initial_profile,
    )
    alphanet.apply(init_weights)
    nn.init.zeros_(alphanet.network.output.weight)
    nn.init.zeros_(alphanet.network.output.bias)

    mean, err = loss_fn.target_exp, loss_fn.exp_err
    print("---undeformed " + obs_str + "----------------")
    print(f"exp = {mean:3g}, std={err:3g}")

    trainer = run_training(
        model,
        loss_fn,
        alphanet,
        latsize,
        source,
        args,
    )
    best_coefficients = trainer.best_alphas(BEST_WINDOW)
    mean, err = loss_fn.get_stats(
        alphas=best_coefficients, sampletype="all"
    )
    variance_ratio = float(
        (
            loss_fn(alphas=best_coefficients, sampletype="all")
            / loss_fn.initial_var
        ).item()
    )
    print("-----field-dependent " + obs_str + "----------------")
    print(
        f"exp = {mean:3g}, std={err:3g}"
        f"\nVar: def/undef = {variance_ratio:3g}"
    )
    source_coefficients = best_coefficients[*source].detach().cpu().numpy()
    save_training_result(
        training_savename,
        trainer,
        loss_function="field-dependent",
        ndim=ndim,
        beta=beta,
        latsize=latsize,
        source=source,
        learned_field=best_coefficients,
        alpha_scale=1.0,
        variance_ratio=variance_ratio,
        extra_metadata={
            "deformation_type": "configuration-dependent",
            "field_parameterization": (
                "F_x=a_x+b_x*sin(theta_x)+c_x*cos(theta_x)"
            ),
            "jacobian": "diagonal; exact determinant included",
            "mask_radius": int(args.mask_radius),
            "initialization": initialization_source,
            "source_coefficients": tuple(
                float(value) for value in source_coefficients
            ),
        },
        extra_data={
            "deformation_coefficients": (
                best_coefficients.detach().cpu().numpy()
            ),
            "training_sample_mask": split_mask.detach().cpu().numpy(),
        },
    )
    return variance_ratio


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
    # The hard f(source)=1 constraint breaks PenaltyLoss's stationary
    # zero-field direction. Keep the remaining unconstrained output small so
    # the extensive exponential stays finite on large lattices.
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
