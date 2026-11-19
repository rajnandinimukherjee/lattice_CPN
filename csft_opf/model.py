"""Angular-deformation model for complex-scalar ensembles."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from deformations import ThetaDefs


class ComplexScalarAngularDeformations(ThetaDefs):
    """Store angular samples and their undeformed radial partners."""

    radial_samples: Mapping[str, torch.Tensor]

    def action_input(
        self,
        theta: torch.Tensor,
        *,
        sampletype: str,
        batch_idx: torch.Tensor | slice,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        radial = self.radial_samples[sampletype]
        if sampletype == "train":
            radial = radial[batch_idx]
        return radial, theta
