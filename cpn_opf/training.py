"""Shared CP(N) OPF training and serialization helpers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from networks import UNet
from training import Trainer, periodic_box_mask

from .config import SPLIT
from .model import CPnHomogeneousDeformations


def make_deformation_mask(
    latsize: tuple[int, ...],
    source: tuple[int, ...],
    radius: int | float,
    channels: int,
    *,
    centers: Sequence[tuple[int, ...]] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the periodic OPF support mask for every deformation channel.

    A U-Net produces one deformation value per CP(N) component.  Expanding
    the spatial support over the final axis makes the locality constraint
    explicit: every component is zero outside ``radius`` of ``source``.
    """
    if channels < 1:
        raise ValueError(
            "the deformation field must have at least one channel")
    support = torch.zeros(latsize, dtype=torch.bool)
    for center in (centers if centers is not None else (source,)):
        support |= periodic_box_mask(latsize, center, radius)
    return support.unsqueeze(-1).expand(*support.shape, channels).to(device)


def penalty_source_constraints(
    source: tuple[int, ...],
    channels: int,
    i: int,
    j: int,
) -> dict[tuple[int, ...], float]:
    """Return a unit-norm, OPF-symmetric hard constraint at ``source``.

    For an off-diagonal OPF, the source deformation has equal and opposite
    real amplitudes in its two color channels,
    ``f_i = 1/sqrt(2)`` and ``f_j = -1/sqrt(2)``.  All remaining channels
    vanish at the source, so the full channel-vector norm is exactly one.  A
    diagonal OPF retains the scalar normalization ``f_i = 1``.

    This fixes the penalty loss's stationary zero-field direction without
    arbitrarily selecting one color from an off-diagonal observable.
    """
    if channels < 1:
        raise ValueError(
            "the deformation field must have at least one channel")
    if not 0 <= i < channels or not 0 <= j < channels:
        raise ValueError("OPF color indices must address deformation channels")

    if i == j:
        return {
            (*source, channel): 1.0 if channel == i else 0.0
            for channel in range(channels)
        }

    amplitude = 1.0 / sqrt(2.0)
    return {
        (*source, channel): (
            amplitude if channel == i else -amplitude if channel == j else 0.0
        )
        for channel in range(channels)
    }


def make_split_mask(
    fields: torch.Tensor,
    split_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a random, validated train/test configuration mask."""
    n_cfgs = len(fields)
    if n_cfgs < 2:
        raise ValueError("at least two thinned configurations are required")
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
        raise ValueError(
            "split_mask must include train and test configurations")
    return split_mask


def make_model(
    fields: torch.Tensor,
    links: torch.Tensor | None,
    *,
    n: int,
    action_type: str,
    split_mask: torch.Tensor,
    device: torch.device,
) -> CPnHomogeneousDeformations:
    """Build a deformation model with the selected action's inputs."""
    if action_type == "gauge":
        if links is None:
            raise ValueError("the gauge action requires auxiliary gauge links")
        if len(fields) != len(links):
            raise ValueError(
                "fields and links must have the same configuration count"
            )
    model = CPnHomogeneousDeformations(
        n, action_type=action_type, device=device
    ).to(device)
    model.samples = {
        "all": fields,
        "train": fields[split_mask],
        "test": fields[~split_mask],
    }
    if action_type == "gauge":
        assert links is not None
        model.set_gauge_samples(
            {
                "all": links,
                "train": links[split_mask],
                "test": links[~split_mask],
            }
        )
    return model


def run_training(
    model: CPnHomogeneousDeformations,
    loss_fn: torch.nn.Module,
    alphanet: torch.nn.Module,
    latsize: tuple[int, ...],
    source: tuple[int, ...],
    args: argparse.Namespace,
    fixed_alphas: Mapping[tuple[int, ...], float] | None = None,
    learning_rate: float | None = None,
    support_centers: Sequence[tuple[int, ...]] | None = None,
) -> Trainer:
    """Train one CP(N) deformation with the shared scheduler controls."""
    if learning_rate is None:
        learning_rate = args.learning_rate
    optimizer = optim.Adam(alphanet.parameters(), lr=learning_rate, eps=1e-6)
    trainer = Trainer(model, loss_fn, optimizer, args.epochs, args.batch_size)
    mask = make_deformation_mask(
        latsize,
        source,
        radius=args.mask_radius,
        channels=model.DOF,
        centers=support_centers,
        device=model.samples["all"].device,
    )
    trainer.train(
        alphanet,
        mask=mask,
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
        validation_interval=args.validation_interval,
        validation_samples=args.validation_samples,
    )
    return trainer


def save_training_result(
    filename: str | Path,
    trainer: Trainer,
    *,
    n: int,
    action_type: str,
    ndim: int,
    beta: float,
    latsize: tuple[int, ...],
    source: tuple[int, ...],
    i: int,
    j: int,
    learned_field: torch.Tensor,
    variance_ratio: float,
    loss_function: str = "variance",
    alpha_scale: float = 1.0,
    extra_metadata: Mapping[str, object] | None = None,
    extra_data: Mapping[str, object] | None = None,
) -> None:
    """Save histories, deformation field, and CP(N)-specific metadata."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    data = trainer.export_training_data()
    field = learned_field.detach().cpu().numpy()
    metadata = {
            "loss_function": loss_function,
            "cp_n": int(n),
            "ndim": int(ndim),
            "beta": float(beta),
            "lattice_size": tuple(int(length) for length in latsize),
            "source": tuple(int(coordinate) for coordinate in source),
            "i": int(i),
            "j": int(j),
            "action_type": action_type,
            "field_at_source": float(field[*source, i]),
            "field_norm_at_source": float(np.linalg.norm(field[*source])),
        }
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    data.update(
        metadata=metadata,
        loss_function=loss_function,
        learned_field=field,
        alpha_scale=float(alpha_scale),
        scaled_field=float(alpha_scale) * field,
        variance_ratio=float(variance_ratio),
    )
    if extra_data is not None:
        data.update(extra_data)
    if loss_function == "penalty":
        source_constraints = penalty_source_constraints(
            source, field.shape[-1], i, j
        )
        source_channels = np.asarray(sorted({i, j}), dtype=np.int64)
        data["f_x"] = field
        data["f_source_components"] = source_channels
        data["f_source_component_values"] = np.asarray(
            [source_constraints[(*source, channel)]
             for channel in source_channels],
            dtype=float,
        )
        data["f_source_norm_constraint"] = 1.0
    np.save(filename, data, allow_pickle=True)
