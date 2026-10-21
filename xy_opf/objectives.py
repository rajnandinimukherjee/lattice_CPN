"""Objective-specific neural-training workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from deformations import ThetaDefs
from losses import PenaltyLoss, VarianceLoss
from networks import UNet
from observables import XYOnePtFn
from training import init_weights

from .config import BEST_WINDOW
from .optimization import optimize_field_scale
from .training import make_split_mask, run_training, save_training_result


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
