"""Shared train/test, optimizer, and serialization helpers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from deformations import ThetaDefs
from networks import UNet
from training import Trainer, periodic_box_mask

from .config import SPLIT


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
            "split_mask must be a boolean vector over configurations"
        )
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
    support_centers: Sequence[tuple[int, ...]] | None = None,
) -> Trainer:
    """Train one UNet with the common optimizer, mask, and stopping rules."""
    if learning_rate is None:
        learning_rate = args.learning_rate
    optimizer = optim.Adam(alphanet.parameters(), lr=learning_rate)
    trainer = Trainer(
        model, loss_fn, optimizer, args.epochs, args.batch_size
    )
    centers = tuple(support_centers) if support_centers is not None else (source,)
    alpha_support = torch.zeros(tuple(latsize), dtype=torch.bool)
    for center in centers:
        alpha_support |= periodic_box_mask(
            latsize, center, radius=args.mask_radius
        )
    alpha_mask = alpha_support.unsqueeze(-1).to(
        model.samples["all"].device
    )
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
        plateau_threshold_mode=args.plateau_threshold_mode,
        early_stopping=args.early_stopping,
        lr_scheduler=args.lr_scheduler,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_cooldown=args.lr_cooldown,
        min_lr=args.min_learning_rate,
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
    extra_metadata: Mapping[str, object] | None = None,
    extra_data: Mapping[str, object] | None = None,
) -> None:
    """Save histories, final fields, and identifying run metadata."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    data = trainer.export_training_data()
    metadata = {
        "loss_function": loss_function,
        "ndim": ndim,
        "beta": beta,
        "lattice_size": tuple(int(length) for length in latsize),
        "source": tuple(int(coordinate) for coordinate in source),
        "field_at_source": float(learned_field[*source, 0].item()),
    }
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    data.update(
        metadata=metadata,
        loss_function=loss_function,
        learned_field=learned_field.detach().cpu().numpy(),
        alpha_scale=float(alpha_scale),
        scaled_field=(alpha_scale * learned_field).detach().cpu().numpy(),
        variance_ratio=float(variance_ratio),
    )
    if extra_data is not None:
        data.update(extra_data)
    if loss_function == "penalty":
        data["f_x"] = learned_field.detach().cpu().numpy()
        data["f_source_constraint"] = 1.0
    np.save(filename, data, allow_pickle=True)
