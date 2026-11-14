"""Reproduce the tested seed-1407 near-zero XY variance run separately."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import XY_utils as xy
from actions import XYAction
from observables import XYOnePtFn
from xy_opf.objectives import train_variance_loss


SOURCE_RUN = ROOT / (
    "training_data/xy_scaling_dim2_b0.65_loss-variance_"
    "near-zero_training.npy"
)
ENSEMBLE = ROOT / "data/xy_dim2_b0.65_L32_ens.npy"
OUTPUT = ROOT / (
    "training_data/xy_scaling_dim2_b0.65_loss-variance_"
    "near-zero_seed-1407_training.npy"
)


def main() -> None:
    if not SOURCE_RUN.exists():
        raise FileNotFoundError(SOURCE_RUN)
    if not ENSEMBLE.exists():
        raise FileNotFoundError(ENSEMBLE)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")

    source_data = np.load(SOURCE_RUN, allow_pickle=True).item()
    fields, beta = xy.load_ensemble(str(ENSEMBLE), decorrelation_factor=1.0)
    if not np.isclose(beta, 0.65):
        raise ValueError(f"ensemble stores beta={beta:g}; expected beta=0.65")
    saved_split = np.asarray(source_data["training_sample_mask"], dtype=bool)
    if saved_split.shape != (len(fields),):
        raise ValueError(
            "saved XY split does not match the reloaded ensemble: "
            f"{saved_split.shape} versus {(len(fields),)}"
        )
    split_mask = torch.as_tensor(saved_split, dtype=torch.bool)
    source = (0, 0)
    action = XYAction(beta, 2)
    obs = XYOnePtFn(tuple(int(size) for size in fields.shape[-2:]), source, action)
    args = argparse.Namespace(
        mask_radius=6,
        epochs=6000,
        batch_size=512,
        learning_rate=0.002,
        lr_scheduler="constant",
        lr_factor=0.5,
        lr_patience=5,
        lr_cooldown=1,
        min_learning_rate=1e-5,
        max_training_time=None,
        plateau_window=20,
        plateau_patience=40,
        plateau_min_delta=0.001,
        plateau_threshold_mode="rel",
        early_stopping=True,
        unet_hidden_dim=None,
    )

    # Seed 1407 is one of the prior sweep repeats (full-ensemble ratio
    # 0.12597236), using the same split and 512/0.002/constant configuration.
    torch.manual_seed(1407)
    train_variance_loss(
        fields,
        ndim=2,
        beta=beta,
        obs=obs,
        obs_str=f"OPF {source}",
        args=args,
        source=source,
        training_savename=OUTPUT,
        split_mask=split_mask,
    )

    saved = np.load(OUTPUT, allow_pickle=True).item()
    metadata = saved.setdefault("metadata", {})
    metadata.update(
        initialization="near-zero U-Net with zero-initialized output layer",
        sweep_name="xy-fixed-near-zero-lr-batch-scheduler-grid seed repeat",
        seed=1407,
        selected_from_prior_seed_repeats=True,
        split_sha256=str(source_data["metadata"]["split_sha256"]),
        prior_full_ensemble_variance_ratio=0.12597236037254333,
    )
    np.save(OUTPUT, saved, allow_pickle=True)
    print(f"Saved {OUTPUT}")
    print(f"full-ensemble variance ratio = {saved['variance_ratio']:.8f}")
    print(f"best_epoch = {saved['best_epoch']}, stop = {saved['stop_reason']}")


if __name__ == "__main__":
    main()
