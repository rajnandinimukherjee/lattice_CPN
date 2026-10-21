"""Naming, discovery, and result persistence for CP(N) OPF studies."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .config import BEST_WINDOW, DATA_DIRECTORY, TRAINING_DATA_DIRECTORY


def beta_key(beta: float) -> float:
    """Return a stable float key for a beta value stored in a filename."""
    value = float(beta)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("beta must be positive and finite")
    return value


def beta_label(beta: float) -> str:
    """Format beta without trailing zeros, matching ``b{beta:2g}`` names."""
    # ``:2g`` is the requested ensemble convention.  Strip its width-padding
    # so a one-digit beta does not create a filename containing a space.
    return f"{beta_key(beta):2g}".strip()


def action_label(action_type: str) -> str:
    """Return a validated action name suitable for artifact filenames."""
    action = action_type.lower()
    if action not in {"gauge", "quartic"}:
        raise ValueError("action_type must be 'gauge' or 'quartic'")
    return action


def artifact_suffix(loss_function: str) -> str:
    """Return the per-beta artifact suffix for a CP(N) objective."""
    return "result" if loss_function == "approx2" else "training"


def ensemble_savename(
    n: int,
    ndim: int,
    beta: float,
    lattice_size: int,
    action_type: str = "gauge",
) -> Path:
    """Return the canonical CP(N) ensemble filename.

    For example, CP(2) in two dimensions at beta 4 on a 64^2 lattice is
    ``data/cp2_gauge_dim2_b4_L64_ens.npy``.
    """
    return DATA_DIRECTORY / (
        f"cp{int(n)}_{action_label(action_type)}_dim{int(ndim)}_"
        f"b{beta_key(beta):.2f}_"
        f"L{int(lattice_size)}_ens.npy"
    )


def _artifact_stem(
    n: int,
    ndim: int,
    beta: float,
    lattice_size: int,
    i: int,
    j: int,
    action_type: str,
) -> str:
    return (
        f"cp{int(n)}_scaling_dim{int(ndim)}_b{beta_label(beta)}_"
        f"L{int(lattice_size)}_i{int(i)}_j{int(j)}_action-{action_label(action_type)}"
    )


def training_savename(
    n: int,
    ndim: int,
    beta: float,
    lattice_size: int,
    i: int,
    j: int,
    loss_function: str = "variance",
    action_type: str = "gauge",
) -> Path:
    """Return the per-beta training or numerical-result path."""
    return TRAINING_DATA_DIRECTORY / (
        _artifact_stem(n, ndim, beta, lattice_size, i, j, action_type)
        + f"_loss-{loss_function}_{artifact_suffix(loss_function)}.npy"
    )


def ratio_savename(
    n: int,
    ndim: int,
    lattice_size: int,
    i: int,
    j: int,
    loss_function: str = "variance",
    action_type: str = "gauge",
) -> Path:
    """Return the beta-indexed variance-ratio path for one scaling series."""
    return TRAINING_DATA_DIRECTORY / (
        f"cp{int(n)}_scaling_dim{int(ndim)}_L{int(lattice_size)}_"
        f"i{int(i)}_j{int(j)}_action-{action_label(action_type)}_"
        f"loss-{loss_function}_var_ratios.npy"
    )


def read_variance_ratio_data(
    filename: str | Path,
) -> tuple[dict, dict[float, float]]:
    """Read a beta-indexed variance-ratio file into metadata and a mapping."""
    filename = Path(filename)
    data = np.load(filename, allow_pickle=True)
    if data.shape != () or not isinstance(data.item(), dict):
        raise ValueError(f"invalid variance-ratio data in {filename}")
    payload = data.item()
    betas = np.asarray(payload.get("betas"), dtype=float)
    ratios = np.asarray(payload.get("variance_ratios"), dtype=float)
    if betas.ndim != 1 or ratios.ndim != 1 or len(betas) != len(ratios):
        raise ValueError(f"invalid beta/ratio arrays in {filename}")
    values: dict[float, float] = {}
    for beta, ratio in zip(betas, ratios):
        key = beta_key(beta)
        if key in values:
            raise ValueError(f"duplicate beta={key:g} in {filename}")
        values[key] = float(ratio)
    return dict(payload.get("metadata", {})), values


def write_variance_ratio_data(
    filename: str | Path,
    *,
    metadata: Mapping[str, object],
    ratios_by_beta: Mapping[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Persist sorted beta/ratio values and return the arrays written."""
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
            "metadata": dict(metadata),
            "betas": betas,
            "variance_ratios": ratios,
        },
        allow_pickle=True,
    )
    return betas, ratios


def training_files_for_dimension(
    n: int,
    ndim: int,
    lattice_size: int,
    i: int,
    j: int,
    loss_function: str = "variance",
    action_type: str = "gauge",
) -> list[tuple[float, Path]]:
    """Return saved training artifacts for one CP(N) scaling series."""
    pattern = (
        f"cp{int(n)}_scaling_dim{int(ndim)}_b*_L{int(lattice_size)}_"
        f"i{int(i)}_j{int(j)}_action-{action_label(action_type)}_"
        f"loss-{loss_function}_{artifact_suffix(loss_function)}.npy"
    )
    files: list[tuple[float, Path]] = []
    for path in TRAINING_DATA_DIRECTORY.glob(pattern):
        beta_text = path.stem.split("_b", 1)[1].split("_L", 1)[0]
        try:
            files.append((beta_key(float(beta_text)), path))
        except ValueError:
            continue
    return sorted(files)


def upsert_variance_ratios(
    filename: str | Path,
    *,
    n: int,
    ndim: int,
    lattice_size: int,
    i: int,
    j: int,
    loss_function: str,
    updates: Mapping[float, float],
    action_type: str = "gauge",
) -> tuple[np.ndarray, np.ndarray]:
    """Add or replace requested betas without dropping existing results."""
    filename = Path(filename)
    metadata: dict = {}
    ratios_by_beta: dict[float, float] = {}
    if filename.exists():
        metadata, ratios_by_beta = read_variance_ratio_data(filename)

    update_keys = {beta_key(beta) for beta in updates}
    for beta, artifact in training_files_for_dimension(
        n, ndim, lattice_size, i, j, loss_function, action_type
    ):
        key = beta_key(beta)
        if key in ratios_by_beta or key in update_keys:
            continue
        payload = np.load(artifact, allow_pickle=True)
        if payload.shape == () and isinstance(payload.item(), dict):
            saved = payload.item()
            if "variance_ratio" in saved:
                ratios_by_beta[key] = float(saved["variance_ratio"])

    ratios_by_beta.update(
        (beta_key(beta), float(ratio)) for beta, ratio in updates.items()
    )
    metadata.update(
        cp_n=int(n),
        ndim=int(ndim),
        lattice_size=int(lattice_size),
        i=int(i),
        j=int(j),
        action_type=action_label(action_type),
        loss_function=loss_function,
    )
    return write_variance_ratio_data(
        filename,
        metadata=metadata,
        ratios_by_beta=ratios_by_beta,
    )


def load_variance_ratios(
    filename: str | Path,
    betas: list[float] | np.ndarray,
) -> np.ndarray:
    """Load variance ratios aligned with the requested beta values."""
    _, ratios_by_beta = read_variance_ratio_data(filename)
    try:
        return np.asarray(
            [ratios_by_beta[beta_key(beta)] for beta in betas], dtype=float
        )
    except KeyError as error:
        raise ValueError(
            f"no saved variance ratio for beta={float(error.args[0]):g}"
        ) from None


def load_deformation_at_source(
    filename: str | Path,
    *,
    component: int,
) -> float:
    """Return one saved deformation component at the observable source."""
    field, source = load_deformation_field(filename)
    index = (*source, int(component))
    return float(field[index])


def load_deformation_field(
    filename: str | Path,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Load a saved, fully scaled deformation field and its source location.

    Older artifacts without ``scaled_field`` are supported by selecting their
    best recorded field, just as the scaling summary does.
    """
    filename = Path(filename)
    payload = np.load(filename, allow_pickle=True)
    if payload.shape != () or not isinstance(payload.item(), dict):
        raise ValueError(f"invalid training artifact in {filename}")
    data = payload.item()
    metadata = data.get("metadata", {})
    source = tuple(metadata.get("source", ()))
    if not source:
        raise ValueError(f"no source metadata in {filename}")

    if "scaled_field" in data:
        field = np.asarray(data["scaled_field"], dtype=float)
    elif "learned_field" in data:
        field = np.asarray(data["learned_field"], dtype=float)
    else:
        fields = np.asarray(data.get("alphas", []), dtype=float)
        if not len(fields):
            raise ValueError(f"no deformation field stored in {filename}")
        history = np.asarray(data.get("test_var", []), dtype=float)
        if len(history) != len(fields) or not np.isfinite(history).any():
            history = np.asarray(data.get("train_var", []), dtype=float)
        if len(history) != len(fields) or not np.isfinite(history).any():
            raise ValueError(f"no finite training score in {filename}")
        scores = np.full(len(history), np.inf)
        for end in range(BEST_WINDOW - 1, len(history)):
            block = history[end - BEST_WINDOW + 1 : end + 1]
            if np.isfinite(block).all():
                scores[end] = block.mean()
        if not np.isfinite(scores).any():
            scores = np.where(np.isfinite(history), history, np.inf)
        field = fields[int(np.argmin(scores))]

    return field, source


def load_deformations_at_source(
    n: int,
    ndim: int,
    lattice_size: int,
    i: int,
    j: int,
    betas: list[float] | np.ndarray,
    loss_function: str = "variance",
    action_type: str = "gauge",
) -> np.ndarray:
    """Load source-component deformations aligned with selected betas."""
    artifacts = {
        beta_key(beta): path
        for beta, path in training_files_for_dimension(
            n, ndim, lattice_size, i, j, loss_function, action_type
        )
    }
    values = []
    for beta in betas:
        key = beta_key(beta)
        try:
            artifact = artifacts[key]
        except KeyError:
            raise FileNotFoundError(
                "no CP(N) training artifact for "
                f"N={n}, d={ndim}, L={lattice_size}, i={i}, j={j}, "
                f"beta={key:g}"
            ) from None
        values.append(load_deformation_at_source(artifact, component=i))
    return np.asarray(values, dtype=float)


def remove_variance_ratios(
    filename: str | Path,
    *,
    betas: list[float],
) -> list[float]:
    """Remove selected beta entries from a ratio index."""
    filename = Path(filename)
    if not filename.exists():
        return []
    metadata, values = read_variance_ratio_data(filename)
    removed: list[float] = []
    for beta in betas:
        key = beta_key(beta)
        if key in values:
            del values[key]
            removed.append(key)
    if removed:
        write_variance_ratio_data(
            filename, metadata=metadata, ratios_by_beta=values
        )
    return removed


def delete_beta_training_data(
    n: int,
    ndim: int,
    lattice_size: int,
    i: int,
    j: int,
    betas: list[float],
    loss_function: str = "variance",
    action_type: str = "gauge",
) -> tuple[list[Path], list[float]]:
    """Delete selected CP(N) artifacts and their ratio-index values."""
    deleted: list[Path] = []
    for beta in betas:
        path = training_savename(
            n, ndim, beta, lattice_size, i, j, loss_function, action_type
        )
        if path.exists():
            path.unlink()
            deleted.append(path)
    removed = remove_variance_ratios(
        ratio_savename(
            n, ndim, lattice_size, i, j, loss_function, action_type
        ),
        betas=betas,
    )
    return deleted, removed


def ensemble_files_for_dimension(
    n: int,
    ndim: int,
    lattice_size: int,
    action_type: str = "gauge",
) -> list[tuple[float, Path]]:
    """Discover CP(N) ensemble files for one dimension and lattice size."""
    pattern = (
        f"cp{int(n)}_{action_label(action_type)}_dim{int(ndim)}_"
        f"b*_L{int(lattice_size)}_ens.npy"
    )
    files: list[tuple[float, Path]] = []
    for path in DATA_DIRECTORY.glob(pattern):
        beta_text = path.stem.split("_b", 1)[1].rsplit("_L", 1)[0]
        try:
            files.append((beta_key(float(beta_text)), path))
        except ValueError:
            continue
    return sorted(files)
