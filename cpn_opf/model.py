"""CP(N) deformation model helpers for auxiliary-gauge ensembles."""

from __future__ import annotations

import torch

from CPN_utils import ComplexPair
from deformations import HomogeneousDeformations


class CPnHomogeneousDeformations(HomogeneousDeformations):
    """Homogeneous CP(N) deformation with the selected action interface.

    Gauge links stay paired with their sampled vectors for the auxiliary-gauge
    action. The quartic action acts only on the deformed vectors.
    """

    def __init__(
        self,
        n: int,
        *,
        action_type: str,
        device: str | torch.device = "cpu",
    ) -> None:
        if action_type not in {"gauge", "quartic"}:
            raise ValueError("action_type must be 'gauge' or 'quartic'")
        super().__init__(n, device=device)
        self.action_type = action_type

    def set_gauge_samples(
        self,
        links: dict[str, torch.Tensor],
    ) -> None:
        """Attach gauge-link batches indexed like ``self.samples``."""
        if self.action_type != "gauge":
            raise RuntimeError("quartic CP(N) deformations do not use gauge links")
        self.link_samples = links

    def action_input(
        self,
        fields: torch.Tensor | ComplexPair,
        *,
        sampletype: str,
        batch_idx: torch.Tensor | slice = slice(None),
    ) -> torch.Tensor | ComplexPair | tuple[torch.Tensor | ComplexPair, torch.Tensor]:
        """Return the selected action's input for a vector deformation."""
        if self.action_type == "quartic":
            return fields
        try:
            links = self.link_samples[sampletype]
        except (AttributeError, KeyError) as error:
            raise RuntimeError(
                "gauge links are missing for the requested CP(N) sample set"
            ) from error
        if sampletype == "train":
            links = links[batch_idx]
        if len(fields) != len(links):
            raise ValueError(
                "CP(N) vectors and gauge links have different batch sizes"
            )
        if fields.shape[-1] != 2 * self.DOF:
            raise ValueError("CP(N) fields have an incompatible component count")
        # Preserve interleaved real coordinates.  A deformed field is a
        # ComplexPair, so even its outer complex value remains two real
        # tensors rather than an unsupported MPS complex dtype.
        return fields, links
