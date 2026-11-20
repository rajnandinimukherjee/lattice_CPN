"""Loading helpers for free complex-scalar ensembles."""

from __future__ import annotations

from os import PathLike
from typing import Any

import numpy as np
import torch


def load_ensemble(
    filename: str | PathLike[str],
    *,
    expected_am: float | None = None,
    dtype: torch.dtype = torch.complex64,
) -> tuple[torch.Tensor, float | None]:
    """Load configurations produced by ``theory.ScalarAction.sample``.

    The native on-disk representation is a complex NumPy array with shape
    ``(configurations, *lattice_shape)``.  A dictionary containing ``phi`` (or
    ``fields``) and optional ``am`` metadata is accepted as well.
    """
    loaded = np.load(filename, allow_pickle=True)
    stored_am: float | None = None
    if isinstance(loaded, np.lib.npyio.NpzFile):
        # The native format produced by the data-generation pipeline: an
        # ``.npz`` archive with a ``cfgs`` array and a scalar ``m`` (bare
        # mass) field, alongside provenance metadata.
        with loaded as archive:
            field_key = next(
                (key for key in ("cfgs", "phi", "fields") if key in archive.files),
                None,
            )
            if field_key is None:
                raise ValueError(f"no 'cfgs'/'phi' field found in {filename}")
            fields = np.asarray(archive[field_key])
            mass_key = next(
                (key for key in ("m", "am") if key in archive.files), None
            )
            if mass_key is not None:
                stored_am = float(archive[mass_key])
    elif loaded.shape == () and isinstance(loaded.item(), dict):
        payload: dict[str, Any] = loaded.item()
        field_data = payload.get("phi", payload.get("fields"))
        if field_data is None:
            raise ValueError(f"no 'phi' field found in {filename}")
        fields = np.asarray(field_data)
        if "am" in payload:
            stored_am = float(payload["am"])
    else:
        fields = np.asarray(loaded)

    if fields.ndim < 2:
        raise ValueError(
            "scalar ensembles must have shape (configurations, *lattice_shape)"
        )
    if fields.shape[0] < 2:
        raise ValueError("scalar ensembles require at least two configurations")
    if not np.iscomplexobj(fields):
        raise ValueError("scalar configurations must be complex-valued")
    if not np.isfinite(fields.real).all() or not np.isfinite(fields.imag).all():
        raise ValueError("scalar configurations must contain only finite values")
    if expected_am is not None:
        expected_am = float(expected_am)
        if expected_am <= 0 or not np.isfinite(expected_am):
            raise ValueError("expected_am must be positive and finite")
        if stored_am is not None and not np.isclose(stored_am, expected_am):
            raise ValueError(
                f"{filename} stores am={stored_am:g}, expected {expected_am:g}"
            )

    tensor = torch.as_tensor(fields).to(dtype=dtype)
    print(
        "\n"
        f"am={stored_am if stored_am is not None else expected_am}, "
        f"fields.shape={tuple(tensor.shape)}, exact free-field samples"
    )
    return tensor, stored_am
