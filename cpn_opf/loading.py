"""CP(N) ensemble loading and autocorrelation-aware thinning."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
import warnings

import numpy as np
import torch

from CPN_utils import CNtoR2N
from XY_utils import autocorrelation, integrated_autocorrelation_time
from actions import CPnGaugeAction, CPnQuarticAction


_VECTOR_KEYS = ("z", "vectors", "fields", "x")
_LINK_KEYS = (
    "lam",
    "lambda",
    "theta",
    "thetas",
    "gauge",
    "gauge_field",
    "gauge_fields",
    "gauge_links",
    "links",
    "link_angles",
    "U",
    "u",
)
_ENERGY_KEYS = ("energies", "energy")


def _payload_from_file(filename: str | PathLike[str]) -> Mapping[str, object]:
    """Load a dictionary-like ``.npy`` or ``.npz`` CP(N) ensemble payload."""
    loaded = np.load(filename, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        return {key: loaded[key] for key in loaded.files}
    if loaded.shape == () and isinstance(loaded.item(), Mapping):
        return loaded.item()
    if loaded.dtype.names is not None:
        return {key: loaded[key] for key in loaded.dtype.names}
    if loaded.dtype == object and loaded.ndim == 1 and len(loaded) == 2:
        return {"z": loaded[0], "theta": loaded[1]}
    raise ValueError(
        "a CP(N) ensemble must be a saved mapping with a vector array "
        "(and link-angle array for gauge analyses)"
    )


def _lookup(
    payload: Mapping[str, object],
    names: tuple[str, ...],
    label: str,
    *,
    required: bool = True,
) -> object | None:
    for name in names:
        if name in payload:
            return payload[name]
    configs = payload.get("configs")
    if isinstance(configs, Mapping):
        for name in names:
            if name in configs:
                return configs[name]
    if required:
        raise KeyError(
            f"the CP(N) ensemble is missing {label}; expected one of "
            + ", ".join(names)
        )
    return None


def _with_chain_axis(
    values: object,
    *,
    ndim: int,
    lattice_size: int,
    final_size: int,
    name: str,
    prefer_components_first: bool = False,
) -> tuple[torch.Tensor, bool]:
    """Normalise a canonical or channel-first CP(N) array layout.

    Returns the canonical ``(chain, time, *lattice, components)`` tensor and
    whether the input stored components before the lattice axes.  The latter
    is the split-real layout used by local copied gauge ensembles.
    """
    tensor = torch.as_tensor(values)
    lattice = (int(lattice_size),) * int(ndim)
    single_shape_ndim = ndim + 2
    chained_shape_ndim = ndim + 3
    components_first = False
    if tensor.ndim == single_shape_ndim:
        canonical = (
            tuple(tensor.shape[1:-1]) == lattice
            and tensor.shape[-1] == final_size
        )
        channel_first = (
            tensor.shape[1] == final_size and tuple(tensor.shape[2:]) == lattice
        )
        if channel_first and (prefer_components_first or not canonical):
            tensor = tensor.movedim(1, -1).unsqueeze(0)
            components_first = True
        elif canonical:
            tensor = tensor.unsqueeze(0)
        else:
            raise ValueError(
                f"{name} must have shape (samples, *lattice, components) or "
                f"(samples, components, *lattice), got {tuple(tensor.shape)}"
            )
    elif tensor.ndim == chained_shape_ndim:
        canonical = (
            tuple(tensor.shape[2:-1]) == lattice
            and tensor.shape[-1] == final_size
        )
        channel_first = (
            tensor.shape[2] == final_size and tuple(tensor.shape[3:]) == lattice
        )
        if channel_first and (prefer_components_first or not canonical):
            tensor = tensor.movedim(2, -1)
            components_first = True
        elif canonical:
            pass
        else:
            raise ValueError(
                f"{name} must have shape (chains, samples, *lattice, components) "
                f"or (chains, samples, components, *lattice), got {tuple(tensor.shape)}"
            )
    else:
        raise ValueError(
            f"{name} must have shape (samples, *lattice, components) or "
            f"(chains, samples, *lattice, components), got {tuple(tensor.shape)}"
        )
    if tuple(tensor.shape[2:-1]) != lattice:
        raise ValueError(
            f"{name} lattice shape {tuple(tensor.shape[2:-1])} does not "
            f"match L={lattice_size} in {ndim} dimensions"
        )
    if tensor.shape[-1] != final_size:
        raise ValueError(
            f"{name} has {tensor.shape[-1]} components; expected {final_size}"
        )
    if tensor.shape[0] < 1 or tensor.shape[1] < 2:
        raise ValueError(f"{name} must contain at least one chain and two samples")
    return tensor, components_first


def _energy_series(
    supplied: object | None,
    fields: torch.Tensor,
    links: torch.Tensor | None,
    *,
    n: int,
    lattice: tuple[int, ...],
    beta: float,
    action_type: str,
) -> torch.Tensor:
    """Use compatible stored energies or evaluate the selected action."""
    if supplied is not None and action_type == "gauge":
        energy = torch.as_tensor(supplied, dtype=torch.float64)
        is_generator_batch_mean = (
            energy.ndim == 1
            and fields.shape[0] > 1
            and tuple(energy.shape) == (fields.shape[1],)
        )
        if energy.ndim == 1:
            energy = energy.unsqueeze(0)
        if energy.shape == fields.shape[:2]:
            return energy
        if not is_generator_batch_mean:
            warnings.warn(
                "ignoring an energies array whose chain/time shape does not match "
                "the CP(N) vectors",
                stacklevel=2,
            )
    action = (
        CPnGaugeAction(beta, n, lattice)
        if action_type == "gauge"
        else CPnQuarticAction(beta, n, lattice)
    )
    # The generator's 64^2 ensemble contains 8,000 configurations.  Evaluate
    # its diagnostic energy in modest configuration batches so autocorrelation
    # estimation does not require several extra full-size lattice work arrays.
    volume = int(np.prod(lattice))
    chunk_size = max(1, 2**18 // volume)
    flat_fields = fields.flatten(0, 1)
    chunks = []
    for start in range(0, len(flat_fields), chunk_size):
        stop = min(start + chunk_size, len(flat_fields))
        action_fields: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        if action_type == "gauge":
            if links is None:
                raise ValueError("the gauge action requires auxiliary gauge links")
            flat_links = links.flatten(0, 1)
            action_fields = (flat_fields[start:stop], flat_links[start:stop])
        else:
            action_fields = flat_fields[start:stop]
        chunks.append(action.energy(action_fields).real)
    return torch.cat(chunks).reshape(fields.shape[:2]).to(dtype=torch.float64)


def _autocorrelation_stride(
    fields: torch.Tensor,
    links: torch.Tensor | None,
    energies: object | None,
    *,
    n: int,
    beta: float,
    action_type: str,
    i: int,
    j: int,
    decorrelation_factor: float,
) -> tuple[int, float, str, float]:
    """Choose a shared thinning stride from action-invariant diagnostics."""
    if not np.isfinite(decorrelation_factor) or decorrelation_factor <= 0:
        raise ValueError("decorrelation_factor must be positive and finite")
    lattice = tuple(int(length) for length in fields.shape[2:-1])
    energy = _energy_series(
        energies,
        fields,
        links,
        n=n,
        lattice=lattice,
        beta=beta,
        action_type=action_type,
    )
    spatial_dims = tuple(range(2, fields.ndim - 1))
    z_i = fields[..., 2 * i] + 1j * fields[..., 2 * i + 1]
    z_j = fields[..., 2 * j] + 1j * fields[..., 2 * j + 1]
    projector = z_i * z_j.conj()
    diagnostics: dict[str, torch.Tensor] = {
        "energy": energy,
        "Re(P_ij)": projector.real.mean(dim=spatial_dims),
        "Im(P_ij)": projector.imag.mean(dim=spatial_dims),
    }
    # Component densities remain gauge invariant and can expose slow modes
    # that an energy-only diagnostic misses.
    for component in range(n + 1):
        diagnostics[f"P_{component}{component}"] = (
            (
                fields[..., 2 * component].square()
                + fields[..., 2 * component + 1].square()
            ).mean(dim=spatial_dims)
        )

    times: dict[str, float] = {}
    for name, values in diagnostics.items():
        _, acf = autocorrelation(values)
        try:
            times[name] = integrated_autocorrelation_time(acf)
        except ValueError as error:
            warnings.warn(
                f"skipping constant {name} autocorrelation diagnostic: {error}",
                stacklevel=2,
            )
    if not times:
        raise ValueError("cannot estimate CP(N) autocorrelation from constants")

    limiting_name, tau_int = max(times.items(), key=lambda item: item[1])
    statistical_inefficiency = 2 * tau_int
    stride = max(
        1, int(np.ceil(decorrelation_factor * statistical_inefficiency))
    )
    effective_sample_size = (
        fields.shape[0] * fields.shape[1] / statistical_inefficiency
    )
    if fields.shape[1] < 50 * statistical_inefficiency:
        warnings.warn(
            "the CP(N) chain is shorter than 50 autocorrelation times; "
            "the thinning estimate may be unreliable",
            stacklevel=2,
        )
    return stride, tau_int, limiting_name, effective_sample_size


def load_ensemble(
    filename: str | PathLike[str],
    *,
    n: int,
    ndim: int,
    lattice_size: int,
    beta: float,
    action_type: str = "gauge",
    i: int = 0,
    j: int = 0,
    decorrelation_factor: float = 1.0,
    diagnostics: dict[str, object] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, float | None]:
    """Load and thin a CP(N) ensemble for the selected action.

    Accepted data contain complex ``z`` vectors, interleaved real vectors, or
    the split real channel layout used by copied gauge ensembles. Arrays may
    have one chain axis or be a single stream. Returned vectors always use
    interleaved real coordinates suitable for ``HomogeneousDeformations``.
    Gauge analyses additionally require link angles, which can be stored last
    or before the lattice axes. Quartic analyses need only vector fields and
    return ``None`` for links, even if the input happens to contain them.
    """
    if action_type not in {"gauge", "quartic"}:
        raise ValueError("action_type must be 'gauge' or 'quartic'")
    if not 0 <= i <= n or not 0 <= j <= n:
        raise ValueError(f"i and j must lie between 0 and N={n}")
    payload = _payload_from_file(filename)
    raw_z = _lookup(payload, _VECTOR_KEYS, "CP(N) vectors")
    raw_links = _lookup(
        payload,
        _LINK_KEYS,
        "auxiliary gauge link angles",
        required=action_type == "gauge",
    )
    raw_energy = _lookup(
        payload, _ENERGY_KEYS, "energies", required=False
    )

    z = torch.as_tensor(raw_z)
    vector_components = n + 1 if z.is_complex() else 2 * (n + 1)
    # ``x`` is the channel-first real convention used by the local CP(N)
    # generators. Treat it as such even for a vector-only quartic ensemble.
    split_real_layout = "x" in payload
    z, split_real_channels = _with_chain_axis(
        z,
        ndim=ndim,
        lattice_size=lattice_size,
        final_size=vector_components,
        name="CP(N) vectors",
        prefer_components_first=split_real_layout,
    )
    if z.is_complex():
        if split_real_channels:
            raise ValueError("CP(N) vectors cannot be complex with split real channels")
        fields = CNtoR2N(z).to(dtype=torch.float32)
    elif z.is_floating_point():
        if split_real_channels:
            n_components = n + 1
            fields = torch.empty_like(z, dtype=torch.float32)
            fields[..., 0::2] = z[..., :n_components]
            fields[..., 1::2] = z[..., n_components:]
        else:
            fields = z.to(dtype=torch.float32)
    else:
        raise ValueError("CP(N) vectors must be complex or floating-point")

    links: torch.Tensor | None = None
    if action_type == "gauge":
        # ``raw_links`` is guaranteed above for a gauge analysis.
        assert raw_links is not None
        links, _ = _with_chain_axis(
            raw_links,
            ndim=ndim,
            lattice_size=lattice_size,
            final_size=ndim,
            name="gauge link angles",
            prefer_components_first=split_real_channels,
        )
        # Some generators save U(1) links directly rather than their angles.
        # The gauge action accepts angles, so retain phase information only.
        if links.is_complex():
            links = torch.angle(links)
        elif not links.is_floating_point():
            raise ValueError(
                "gauge links must be real angles or complex U(1) phases"
            )
        links = links.to(dtype=torch.float32)
        if fields.shape[:2] != links.shape[:2]:
            raise ValueError(
                "CP(N) vectors and gauge links have different chain/time shapes"
            )

    stride, tau_int, limiting_name, ess = _autocorrelation_stride(
        fields,
        links,
        raw_energy,
        n=n,
        beta=beta,
        action_type=action_type,
        i=i,
        j=j,
        decorrelation_factor=decorrelation_factor,
    )
    if diagnostics is not None:
        diagnostics.update(
            stride=int(stride),
            tau_int=float(tau_int),
            limited_by=limiting_name,
            effective_sample_size=float(ess),
        )
    fields = fields[:, ::stride].flatten(0, 1)
    if links is not None:
        links = links[:, ::stride].flatten(0, 1)
    saved_beta = payload.get("beta")
    loaded_beta = float(saved_beta) if saved_beta is not None else None

    print(
        "\n"
        f"beta={beta:g}, fields.shape={tuple(fields.shape)}, "
        f"tau_int={tau_int:.3f}, g={2 * tau_int:.3f}, ESS~{ess:.0f}, "
        f"stride={stride}, limited_by={limiting_name}"
    )
    return fields, links, loaded_beta
