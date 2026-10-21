"""File naming, discovery, and persistence for XY OPF study artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .config import (
    BEST_WINDOW,
    DATA_DIRECTORY,
    TRAINING_DATA_DIRECTORY,
)


def training_savename(ndim: int, beta: float, loss_function: str) -> Path:
    """Return an objective-specific training or numerical-result filename."""
    suffix = "result" if loss_function == "saddle-point" else "training"
    return TRAINING_DATA_DIRECTORY / (
        f"xy_scaling_dim{ndim}_b{beta:.2f}_"
        f"loss-{loss_function}_{suffix}.npy"
    )


def ratio_savename(ndim: int, loss_function: str) -> Path:
    """Return an objective-specific variance-ratio filename."""
    return TRAINING_DATA_DIRECTORY / (
        f"xy_scaling_dim{ndim}_loss-{loss_function}_var_ratios.npy"
    )


def beta_key(beta: float) -> float:
    """Return the beta value represented by the two-decimal filenames."""
    return float(f"{float(beta):.2f}")


def read_variance_ratio_data(
    filename: str | Path,
    ndim: int,
    loss_function: str,
) -> tuple[dict, dict[float, float]]:
    """Read a variance-ratio file into metadata and a beta-keyed mapping."""
    filename = Path(filename)
    loaded = np.load(filename, allow_pickle=True)
    if loaded.shape != () or not isinstance(loaded.item(), dict):
        raise ValueError(f"invalid variance-ratio data in {filename}")
    data = loaded.item()
    metadata = dict(data.get("metadata", {}))
    saved_betas = np.asarray(data["betas"], dtype=float)
    saved_ratios = np.asarray(data["variance_ratios"], dtype=float)

    if saved_betas.ndim != 1 or saved_ratios.ndim != 1:
        raise ValueError(f"invalid variance-ratio arrays in {filename}")
    if len(saved_betas) != len(saved_ratios):
        raise ValueError(
            f"cannot associate ratios in {filename} with beta values"
        )

    ratios_by_beta: dict[float, float] = {}
    for beta, ratio in zip(saved_betas, saved_ratios):
        key = beta_key(beta)
        if key in ratios_by_beta:
            raise ValueError(f"duplicate beta={key:g} in {filename}")
        ratios_by_beta[key] = float(ratio)
    return metadata, ratios_by_beta


def write_variance_ratio_data(
    filename: str | Path,
    *,
    metadata: dict,
    ratios_by_beta: Mapping[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Write sorted beta/ratio arrays and return the arrays that were saved."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        (beta_key(beta), float(ratio))
        for beta, ratio in ratios_by_beta.items()
    )
    betas = np.asarray([beta for beta, _ in ordered], dtype=float)
    ratios = np.asarray([ratio for _, ratio in ordered], dtype=float)
    np.save(
        filename,
        {
            "metadata": metadata,
            "betas": betas,
            "variance_ratios": ratios,
        },
        allow_pickle=True,
    )
    return betas, ratios


def upsert_variance_ratios(
    filename: str | Path,
    *,
    ndim: int,
    lattice_size: int,
    loss_function: str,
    updates: Mapping[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Add or replace selected betas without dropping stored beta results."""
    filename = Path(filename)
    metadata: dict = {}
    ratios_by_beta: dict[float, float] = {}
    if filename.exists():
        metadata, ratios_by_beta = read_variance_ratio_data(
            filename, ndim, loss_function
        )

    update_keys = {beta_key(beta) for beta in updates}
    for beta, artifact in training_files_for_dimension(
        ndim, loss_function
    ):
        key = beta_key(beta)
        if key in ratios_by_beta or key in update_keys:
            continue
        artifact_data = np.load(artifact, allow_pickle=True)
        if artifact_data.shape != () or not isinstance(
            artifact_data.item(), dict
        ):
            continue
        artifact_data = artifact_data.item()
        if "variance_ratio" in artifact_data:
            ratios_by_beta[key] = float(artifact_data["variance_ratio"])

    ratios_by_beta.update(
        (beta_key(beta), float(ratio))
        for beta, ratio in updates.items()
    )
    metadata.update(
        loss_function=loss_function,
        ndim=int(ndim),
        lattice_size=int(lattice_size),
    )
    return write_variance_ratio_data(
        filename,
        metadata=metadata,
        ratios_by_beta=ratios_by_beta,
    )


def remove_variance_ratios(
    filename: str | Path,
    *,
    ndim: int,
    loss_function: str,
    betas: list[float],
) -> list[float]:
    """Remove selected beta entries from a saved variance-ratio file."""
    filename = Path(filename)
    if not filename.exists():
        return []
    metadata, ratios_by_beta = read_variance_ratio_data(
        filename, ndim, loss_function
    )
    removed = []
    for beta in betas:
        key = beta_key(beta)
        if key in ratios_by_beta:
            del ratios_by_beta[key]
            removed.append(key)
    if removed:
        write_variance_ratio_data(
            filename,
            metadata=metadata,
            ratios_by_beta=ratios_by_beta,
        )
    return removed


def load_variance_ratios(
    filename: str | Path,
    ndim: int,
    betas: list[float] | np.ndarray,
    loss_function: str = "variance",
) -> np.ndarray:
    """Load saved variance ratios for selected betas."""
    _, ratios_by_beta = read_variance_ratio_data(
        filename, ndim, loss_function
    )
    try:
        return np.asarray(
            [ratios_by_beta[beta_key(beta)] for beta in betas],
            dtype=float,
        )
    except KeyError as error:
        raise ValueError(
            f"no saved variance ratio for beta={float(error.args[0]):g}"
        ) from None


def training_files_for_dimension(
    ndim: int,
    loss_function: str,
) -> list[tuple[float, Path]]:
    """Return objective-specific saved artifacts for one dimension."""
    files = []
    suffix = "result" if loss_function == "saddle-point" else "training"
    pattern = f"xy_scaling_dim{ndim}_b*_loss-{loss_function}_{suffix}.npy"
    for path in TRAINING_DATA_DIRECTORY.glob(pattern):
        beta_string = path.stem.split("_b", 1)[1].split("_loss-", 1)[0]
        files.append((float(beta_string), path))

    return sorted(files)


def load_deformation_at_source(
    filename: str | Path,
    ndim: int,
) -> float:
    """Load the applied deformation field at its source site."""
    data = np.load(filename, allow_pickle=True)
    if data.shape != () or not isinstance(data.item(), dict):
        raise ValueError(f"invalid training artifact in {filename}")
    data = data.item()
    metadata = data.get("metadata", {})
    source = tuple(metadata.get("source", (0,) * ndim))

    if "scaled_field" in data:
        field = np.asarray(data["scaled_field"], dtype=float)
    elif "learned_field" in data:
        field = float(data.get("alpha_scale", 1.0)) * np.asarray(
            data["learned_field"], dtype=float
        )
    elif "alphas" in data:
        fields = np.asarray(data["alphas"], dtype=float)
        if not len(fields):
            raise ValueError(f"no recorded deformation fields in {filename}")

        def best_index(values: object) -> int | None:
            history = np.asarray(values, dtype=float)
            scores = np.full(len(history), np.inf)
            for end in range(BEST_WINDOW - 1, len(history)):
                block = history[end - BEST_WINDOW + 1:end + 1]
                if np.isfinite(block).all():
                    scores[end] = block.mean()
            if not np.isfinite(scores).any():
                scores = np.where(np.isfinite(history), history, np.inf)
            if not np.isfinite(scores).any():
                return None
            return int(np.argmin(scores))

        index = best_index(data.get("test_var", []))
        if index is None:
            index = best_index(data.get("train_var", []))
        if index is None:
            raise ValueError(f"no finite training scores in {filename}")
        field = fields[index]
    else:
        raise ValueError(f"no deformation field stored in {filename}")

    source_index = (*source, 0) if field.ndim == ndim + 1 else source
    return float(field[source_index])


def load_deformations_at_source(
    ndim: int,
    loss_function: str,
    betas: list[float] | np.ndarray,
) -> np.ndarray:
    """Load source-site deformations aligned with selected beta values."""
    artifacts = {
        beta_key(beta): path
        for beta, path in training_files_for_dimension(
            ndim, loss_function
        )
    }
    values = []
    for beta in betas:
        key = beta_key(beta)
        try:
            artifact = artifacts[key]
        except KeyError:
            raise FileNotFoundError(
                "no training artifact containing the deformation field for "
                f"dimension {ndim}, loss={loss_function}, beta={key:g}"
            ) from None
        values.append(load_deformation_at_source(artifact, ndim))
    return np.asarray(values, dtype=float)


def delete_beta_training_data(
    ndim: int,
    loss_function: str,
    betas: list[float],
) -> tuple[list[Path], list[float]]:
    """Delete selected per-beta artifacts and their ratio-index entries."""
    ratio_file = ratio_savename(ndim, loss_function)
    suffix = "result" if loss_function == "saddle-point" else "training"
    deleted_files = []
    for beta in betas:
        key = beta_key(beta)
        artifact = training_savename(ndim, key, loss_function)
        if artifact.exists():
            artifact.unlink()
            deleted_files.append(artifact)

    removed_betas = remove_variance_ratios(
        ratio_file,
        ndim=ndim,
        loss_function=loss_function,
        betas=betas,
    )
    return deleted_files, removed_betas


def ensemble_files_for_dimension(
    ndim: int, lattice_size: str
) -> list[tuple[float, Path]]:
    """Return ensemble files and their beta values for one lattice size."""
    files = []
    for path in DATA_DIRECTORY.glob(
        f"xy_dim{ndim}_b*_L{lattice_size}_ens.npy"
    ):
        beta_string = path.stem.split("_b", 1)[1].rsplit("_L", 1)[0]
        files.append((float(beta_string), path))
    return sorted(files)
