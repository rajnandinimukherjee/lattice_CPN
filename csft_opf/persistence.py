"""File discovery and persistence for complex-scalar OPF artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .config import BEST_WINDOW, DATA_DIRECTORY, TRAINING_DATA_DIRECTORY


def am_key(am: float) -> float:
    """Return the mass value represented by three-decimal filenames."""
    return float(f"{float(am):.3f}")


def training_savename(ndim: int, am: float, loss_function: str) -> Path:
    suffix = "result" if loss_function in {"saddle-point", "approx3"} else "training"
    return TRAINING_DATA_DIRECTORY / (
        f"csft_scaling_dim{ndim}_am{am:.3f}_"
        f"loss-{loss_function}_{suffix}.npy"
    )


def ratio_savename(ndim: int, loss_function: str) -> Path:
    return TRAINING_DATA_DIRECTORY / (
        f"csft_scaling_dim{ndim}_loss-{loss_function}_var_ratios.npy"
    )


def read_variance_ratio_data(
    filename: str | Path,
    ndim: int,
    loss_function: str,
) -> tuple[dict, dict[float, float]]:
    loaded = np.load(filename, allow_pickle=True)
    if loaded.shape != () or not isinstance(loaded.item(), dict):
        raise ValueError(f"invalid variance-ratio data in {filename}")
    data = loaded.item()
    masses = np.asarray(data["ams"], dtype=float)
    ratios = np.asarray(data["variance_ratios"], dtype=float)
    if masses.ndim != 1 or ratios.ndim != 1 or len(masses) != len(ratios):
        raise ValueError(f"invalid mass/ratio arrays in {filename}")
    mapping: dict[float, float] = {}
    for am, ratio in zip(masses, ratios):
        key = am_key(am)
        if key in mapping:
            raise ValueError(f"duplicate am={key:g} in {filename}")
        mapping[key] = float(ratio)
    return dict(data.get("metadata", {})), mapping


def write_variance_ratio_data(
    filename: str | Path,
    *,
    metadata: dict,
    ratios_by_am: Mapping[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        (am_key(am), float(ratio)) for am, ratio in ratios_by_am.items()
    )
    masses = np.asarray([am for am, _ in ordered], dtype=float)
    ratios = np.asarray([ratio for _, ratio in ordered], dtype=float)
    np.save(
        filename,
        {"metadata": metadata, "ams": masses, "variance_ratios": ratios},
        allow_pickle=True,
    )
    return masses, ratios


def upsert_variance_ratios(
    filename: str | Path,
    *,
    ndim: int,
    lattice_size: int,
    loss_function: str,
    updates: Mapping[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    filename = Path(filename)
    metadata: dict = {}
    ratios: dict[float, float] = {}
    if filename.exists():
        metadata, ratios = read_variance_ratio_data(filename, ndim, loss_function)
    update_keys = {am_key(am) for am in updates}
    for am, artifact in training_files_for_dimension(ndim, loss_function):
        key = am_key(am)
        if key in ratios or key in update_keys:
            continue
        data = np.load(artifact, allow_pickle=True)
        if data.shape == () and isinstance(data.item(), dict):
            value = data.item().get("variance_ratio")
            if value is not None:
                ratios[key] = float(value)
    ratios.update((am_key(am), float(value)) for am, value in updates.items())
    metadata.update(
        loss_function=loss_function,
        ndim=int(ndim),
        lattice_size=int(lattice_size),
        scaling_variable="am",
    )
    return write_variance_ratio_data(
        filename, metadata=metadata, ratios_by_am=ratios
    )


def load_variance_ratios(
    filename: str | Path,
    ndim: int,
    masses: list[float] | np.ndarray,
    loss_function: str = "variance",
) -> np.ndarray:
    _, ratios = read_variance_ratio_data(filename, ndim, loss_function)
    try:
        return np.asarray([ratios[am_key(am)] for am in masses], dtype=float)
    except KeyError as error:
        raise ValueError(f"no saved variance ratio for am={error.args[0]:g}") from None


def training_files_for_dimension(
    ndim: int,
    loss_function: str,
) -> list[tuple[float, Path]]:
    suffix = "result" if loss_function in {"saddle-point", "approx3"} else "training"
    files = []
    pattern = f"csft_scaling_dim{ndim}_am*_loss-{loss_function}_{suffix}.npy"
    for path in TRAINING_DATA_DIRECTORY.glob(pattern):
        mass = path.stem.split("_am", 1)[1].split("_loss-", 1)[0]
        files.append((float(mass), path))
    return sorted(files)


def load_deformation_at_source(filename: str | Path, ndim: int) -> float:
    data = np.load(filename, allow_pickle=True)
    if data.shape != () or not isinstance(data.item(), dict):
        raise ValueError(f"invalid training artifact in {filename}")
    payload = data.item()
    source = tuple(payload.get("metadata", {}).get("source", (0,) * ndim))
    if "scaled_field" in payload:
        field = np.asarray(payload["scaled_field"], dtype=float)
    elif "learned_field" in payload:
        field = float(payload.get("alpha_scale", 1.0)) * np.asarray(
            payload["learned_field"], dtype=float
        )
    elif "alphas" in payload:
        fields = np.asarray(payload["alphas"], dtype=float)
        history = np.asarray(payload.get("test_var", []), dtype=float)
        if not len(history) or not len(fields):
            raise ValueError(f"no usable deformation history in {filename}")
        scores = np.full(len(history), np.inf)
        for end in range(BEST_WINDOW - 1, len(history)):
            block = history[end - BEST_WINDOW + 1:end + 1]
            if np.isfinite(block).all():
                scores[end] = block.mean()
        if not np.isfinite(scores).any():
            scores = np.where(np.isfinite(history), history, np.inf)
        field = fields[int(np.argmin(scores))]
    else:
        raise ValueError(f"no deformation field stored in {filename}")
    index = (*source, 0) if field.ndim == ndim + 1 else source
    return float(field[index])


def load_deformations_at_source(
    ndim: int,
    loss_function: str,
    masses: list[float] | np.ndarray,
) -> np.ndarray:
    artifacts = {
        am_key(am): path
        for am, path in training_files_for_dimension(ndim, loss_function)
    }
    values = []
    for am in masses:
        key = am_key(am)
        if key not in artifacts:
            raise FileNotFoundError(
                f"no deformation artifact for dimension {ndim}, "
                f"loss={loss_function}, am={key:g}"
            )
        values.append(load_deformation_at_source(artifacts[key], ndim))
    return np.asarray(values, dtype=float)


def remove_variance_ratios(
    filename: str | Path,
    *,
    ndim: int,
    loss_function: str,
    masses: list[float],
) -> list[float]:
    filename = Path(filename)
    if not filename.exists():
        return []
    metadata, ratios = read_variance_ratio_data(filename, ndim, loss_function)
    removed = []
    for am in masses:
        key = am_key(am)
        if key in ratios:
            del ratios[key]
            removed.append(key)
    if removed:
        write_variance_ratio_data(
            filename, metadata=metadata, ratios_by_am=ratios
        )
    return removed


def delete_am_training_data(
    ndim: int,
    loss_function: str,
    masses: list[float],
) -> tuple[list[Path], list[float]]:
    deleted = []
    for am in masses:
        artifact = training_savename(ndim, am_key(am), loss_function)
        if artifact.exists():
            artifact.unlink()
            deleted.append(artifact)
    removed = remove_variance_ratios(
        ratio_savename(ndim, loss_function),
        ndim=ndim,
        loss_function=loss_function,
        masses=masses,
    )
    return deleted, removed


def ensemble_files_for_dimension(
    ndim: int,
    lattice_size: str,
) -> list[tuple[float, Path]]:
    files = []
    for path in DATA_DIRECTORY.glob(
        f"csft_dim{ndim}_am*_L{lattice_size}_ens.npy"
    ):
        mass = path.stem.split("_am", 1)[1].rsplit("_L", 1)[0]
        files.append((float(mass), path))
    return sorted(files)
