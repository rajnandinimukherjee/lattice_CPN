"""Train isolated near-zero CP(2) histories for diagnostic plots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from actions import CPnGaugeAction
from cpn_opf.loading import load_ensemble
from cpn_opf.objectives import train_penalty_loss, train_variance_loss
from cpn_opf.training import make_model, make_split_mask
from losses import VarianceLoss
from observables import CPnOnePtFn


ENSEMBLE = ROOT / "data/cp2_gauge_dim2_b2.50_L64_ens.npy"
OUTPUTS = {
    "variance": ROOT / (
        "training_data/cp2_scaling_dim2_b2.5_L64_i0_j1_action-gauge_"
        "loss-variance_near-zero_training.npy"
    ),
    "penalty": ROOT / (
        "training_data/cp2_scaling_dim2_b2.5_L64_i0_j1_action-gauge_"
        "loss-penalty_near-zero_training.npy"
    ),
}
MANIFEST = ROOT / (
    "training_data/cp2_scaling_dim2_b2.5_L64_i0_j1_action-gauge_"
    "near-zero_run.json"
)


def training_args() -> argparse.Namespace:
    """Use the saved CP(2) diagnostic run's training and stopping settings."""
    return argparse.Namespace(
        mask_radius=6,
        epochs=6000,
        batch_size=64,
        learning_rate=3e-4,
        lr_scheduler="plateau",
        lr_factor=0.5,
        lr_patience=1,
        lr_cooldown=1,
        min_learning_rate=1e-5,
        max_training_time=None,
        plateau_window=5,
        plateau_patience=10,
        plateau_min_delta=0.005,
        plateau_threshold_mode="rel",
        early_stopping=True,
        validation_interval=10,
        validation_samples=256,
        unet_hidden_dim=None,
        alpha_bounds=(-5.0, 5.0),
        alpha_grid_points=41,
        alpha_xatol=1e-4,
    )


def enrich_artifact(
    filename: Path,
    *,
    fields: torch.Tensor,
    links: torch.Tensor,
    split_mask: torch.Tensor,
    obs: CPnOnePtFn,
    split_seed: int,
    training_seed: int,
    initialization: str,
) -> dict[str, object]:
    """Record initialization and evaluate the penalty field at alpha=1."""
    payload = np.load(filename, allow_pickle=True)
    data = payload.item()
    split_sha256 = hashlib.sha256(
        split_mask.detach().cpu().numpy().tobytes()
    ).hexdigest()
    metadata = data.setdefault("metadata", {})
    metadata.update(
        initialization=initialization,
        training_seed=int(training_seed),
        split_seed=int(split_seed),
        split_sha256=split_sha256,
        seeded_from_approx2=False,
    )

    alpha_one_ratio = float(data["variance_ratio"])
    if data["loss_function"] == "penalty":
        model = make_model(
            fields,
            links,
            n=2,
            action_type="gauge",
            split_mask=split_mask,
            device=fields.device,
        )
        variance_loss = VarianceLoss(model, obs)
        learned_field = torch.as_tensor(
            data["learned_field"], dtype=fields.dtype, device=fields.device
        )
        with torch.no_grad():
            alpha_one_ratio = float(
                (
                    variance_loss(alphas=learned_field, sampletype="all")
                    / variance_loss.initial_var
                ).detach().cpu().item()
            )
            alpha_star_ratio = float(
                (
                    variance_loss(
                        alphas=float(data["alpha_scale"]) * learned_field,
                        sampletype="all",
                    )
                    / variance_loss.initial_var
                ).detach().cpu().item()
            )
        data["alpha_one_variance_ratio"] = alpha_one_ratio
        data["alpha_star_full_ensemble_variance_ratio"] = alpha_star_ratio

    np.save(filename, data, allow_pickle=True)
    return {
        "artifact": str(filename.relative_to(ROOT)),
        "initialization": initialization,
        "training_seed": int(training_seed),
        "split_seed": int(split_seed),
        "split_sha256": split_sha256,
        "stop_reason": data["stop_reason"],
        "best_epoch": int(data["best_epoch"]),
        "variance_ratio_at_alpha_star": float(data["variance_ratio"]),
        "alpha_scale": float(data["alpha_scale"]),
        "alpha_one_variance_ratio": float(data.get("alpha_one_variance_ratio", data["variance_ratio"])),
        "alpha_star_full_ensemble_variance_ratio": float(
            data.get("alpha_star_full_ensemble_variance_ratio", data["variance_ratio"])
        ),
        "training_config": data["training_config"],
    }


def main() -> None:
    if not ENSEMBLE.exists():
        raise FileNotFoundError(ENSEMBLE)
    existing = [path for path in [*OUTPUTS.values(), MANIFEST] if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite isolated run output(s): "
            + ", ".join(str(path) for path in existing)
        )

    device = torch.device("cpu")
    fields, links, beta = load_ensemble(
        ENSEMBLE,
        n=2,
        ndim=2,
        lattice_size=64,
        beta=2.5,
        action_type="gauge",
        i=0,
        j=1,
        decorrelation_factor=1.0,
    )
    fields = fields.to(device)
    links = links.to(device)
    source = (0, 0)
    latsize = tuple(int(length) for length in fields.shape[1:-1])
    action = CPnGaugeAction(beta, 2, latsize, device=device)
    obs = CPnOnePtFn(
        2,
        latsize,
        i=0,
        j=1,
        source=source,
        action=action,
        device=device,
    )

    split_seed = 52822
    torch.manual_seed(split_seed)
    split_mask = make_split_mask(fields)
    args = training_args()
    objective_records = {}

    for loss_function, training_seed, initialization, train_function in (
        (
            "variance",
            52823,
            "near-zero U-Net initialization; no Approximation 2 seed",
            train_variance_loss,
        ),
        (
            "penalty",
            52824,
            "near-zero U-Net residual with only the unit-norm source constraint; no Approximation 2 seed",
            train_penalty_loss,
        ),
    ):
        torch.manual_seed(training_seed)
        print(f"\n--- CP(2) {loss_function}: near-zero initialization ---")
        train_function(
            fields,
            links,
            n=2,
            action_type="gauge",
            ndim=2,
            beta=beta,
            obs=obs,
            args=args,
            source=source,
            i=0,
            j=1,
            training_savename=OUTPUTS[loss_function],
            device=device,
            split_mask=split_mask,
        )
        objective_records[loss_function] = enrich_artifact(
            OUTPUTS[loss_function],
            fields=fields,
            links=links,
            split_mask=split_mask,
            obs=obs,
            split_seed=split_seed,
            training_seed=training_seed,
            initialization=initialization,
        )

    manifest = {
        "ensemble": str(ENSEMBLE.relative_to(ROOT)),
        "cp_n": 2,
        "action": "gauge",
        "ndim": 2,
        "beta": beta,
        "lattice_size": list(latsize),
        "i": 0,
        "j": 1,
        "source": list(source),
        "mask_radius": args.mask_radius,
        "decorrelation_factor": 1.0,
        "seeded_from_approx2": False,
        "shared_split_seed": split_seed,
        "shared_split_sha256": hashlib.sha256(
            split_mask.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "objectives": objective_records,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {MANIFEST}")


if __name__ == "__main__":
    main()
