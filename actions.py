from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


class CPnAction:
    """Quartic or auxiliary-gauge CP(n) lattice action."""

    def __init__(
        self,
        beta: float,
        N: int,
        latsize: Sequence[int],
        style: str = "quartic",
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize an action on a lattice of the requested size."""
        if style not in {"quartic", "gauge"}:
            raise ValueError("style must be 'quartic' or 'gauge'")

        self.beta = beta
        self.N = N
        self.latsize = tuple(latsize)
        self.device = torch.device(device)
        self.style = style

        blocks = [torch.tensor([[0, -1], [1, 0]])] * (N + 1)
        self.omega = torch.block_diag(*blocks).to(self.device)

        if style == "quartic":
            self.action = self.quartic_action
            self.energy = self.lattice_quartic_energy
            self.local_energy = self.local_quartic_energy
        else:
            self.action = self.gauge_action
            self.energy = self.lattice_gauge_energy
            self.local_energy = self.local_gauge_energy

    def __call__(self, fields: Any, **kwargs: Any) -> torch.Tensor:
        """Evaluate beta times the configured lattice energy."""
        if self.style == "quartic":
            if fields.shape[-1] not in {self.N + 1, 2 * (self.N + 1)}:
                raise ValueError("fields have an incompatible final dimension")
        return self.action(fields, **kwargs)

    def quartic_action(
        self,
        fields: torch.Tensor,
        local: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Evaluate the quartic action globally or at selected sites."""
        energy = (
            self.local_quartic_energy(fields, **kwargs)
            if local
            else self.lattice_quartic_energy(fields)
        )
        return self.beta * energy

    def gauge_action(
        self,
        fields: tuple[torch.Tensor, torch.Tensor],
        local: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Evaluate the auxiliary-gauge action globally or locally."""
        energy = (
            self.local_gauge_energy(fields, **kwargs)
            if local
            else self.lattice_gauge_energy(fields)
        )
        return self.beta * energy

    def local_quartic_energy(
        self,
        fields: torch.Tensor,
        sites: torch.Tensor,
    ) -> torch.Tensor:
        """Return the quartic energy touching each selected lattice site."""
        batch_shape = fields.shape[: -len(self.latsize) - 1]
        energy = torch.zeros(
            batch_shape + (sites.shape[0],),
            dtype=fields.real.dtype if fields.shape[-1] == self.N + 1 else fields.dtype,
            device=fields.device,
        )
        site_index = (Ellipsis, *sites.T.unbind(), slice(None))
        z = fields[site_index]

        for mu in range(len(self.latsize)):
            for shift in (+1, -1):
                neighbours = neighbor_indices(sites, mu, self.latsize, shift)
                neighbour_index = (
                    Ellipsis,
                    *neighbours.T.unbind(),
                    slice(None),
                )
                z_neighbour = fields[neighbour_index]
                if fields.shape[-1] == self.N + 1:
                    link = (z.conj() * z_neighbour).sum(-1)
                    energy = energy + 1 - link.abs().square()
                else:
                    omega = self.omega.to(dtype=z.dtype, device=z.device)
                    omega_z_neighbour = z_neighbour @ omega.T
                    term1 = ((z * z_neighbour).sum(-1)).square()
                    term2 = ((z * omega_z_neighbour).sum(-1)).square()
                    energy = energy + 1 - term1 - term2
        return energy

    def lattice_quartic_energy(self, fields: torch.Tensor) -> torch.Tensor:
        """Return the quartic energy of each configuration."""
        lattice_dims = tuple(
            range(fields.ndim - 1 - len(self.latsize), fields.ndim - 1)
        )
        output_dtype = (
            fields.real.dtype if fields.shape[-1] == self.N + 1 else fields.dtype
        )
        energy = torch.zeros(
            fields.shape[:-1], dtype=output_dtype, device=fields.device
        )

        for mu in lattice_dims:
            for shift in (+1, -1):
                neighbour = torch.roll(fields, shift, dims=mu)
                if fields.shape[-1] == self.N + 1:
                    link = (fields.conj() * neighbour).sum(-1)
                    energy = energy + 1 - link.abs().square()
                else:
                    omega = self.omega.to(dtype=fields.dtype, device=fields.device)
                    omega_neighbour = neighbour @ omega.T
                    term1 = ((fields * neighbour).sum(-1)).square()
                    term2 = ((fields * omega_neighbour).sum(-1)).square()
                    energy = energy + 1 - term1 - term2
        return energy.sum(dim=lattice_dims)

    def local_gauge_energy(
        self,
        fields: tuple[torch.Tensor, torch.Tensor],
        sites: torch.Tensor,
    ) -> torch.Tensor:
        """Return the auxiliary-gauge energy at selected sites."""
        z_field, theta_field = fields
        batch_shape = z_field.shape[: -len(self.latsize) - 1]
        energy = torch.zeros(
            batch_shape + (sites.shape[0],),
            dtype=z_field.dtype,
            device=z_field.device,
        )
        site_index = (Ellipsis, *sites.T.unbind(), slice(None))
        z = z_field[site_index]

        for mu in range(len(self.latsize)):
            for shift in (+1, -1):
                neighbours = neighbor_indices(sites, mu, self.latsize, shift)
                neighbour_index = (
                    Ellipsis,
                    *neighbours.T.unbind(),
                    slice(None),
                )
                z_neighbour = z_field[neighbour_index]
                theta_index = (Ellipsis, *sites.T.unbind(), mu)
                theta = theta_field[theta_index]
                link = (z.conj() * z_neighbour).sum(-1)
                energy = energy + 1 - (link * torch.exp(1j * shift * theta)).real
        return energy

    def lattice_gauge_energy(
        self,
        fields: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Return the auxiliary-gauge energy of each configuration."""
        z, theta = fields
        lattice_dims = tuple(range(z.ndim - 1 - len(self.latsize), z.ndim - 1))
        energy = torch.zeros(z.shape[:-1], dtype=z.real.dtype, device=z.device)
        for direction, mu in enumerate(lattice_dims):
            neighbour = torch.roll(z, -1, dims=mu)
            link = (z.conj() * neighbour).sum(-1)
            energy = energy + 1 - (link * torch.exp(1j * theta[..., direction])).real
        return energy.sum(dim=lattice_dims)


def neighbor_indices(
    sites: torch.Tensor,
    mu: int,
    latsize: Sequence[int],
    shift: int,
) -> torch.Tensor:
    """Shift site indices periodically by ``shift`` in direction ``mu``."""
    neighbours = sites.clone()
    neighbours[:, mu] = (neighbours[:, mu] + shift) % latsize[mu]
    return neighbours


class XYAction:
    """Nearest-neighbour action for the XY model."""

    def __init__(self, beta: float, ndims: int = 2) -> None:
        """Initialize an XY action over the final ``ndims`` axes."""
        if ndims < 1:
            raise ValueError("ndims must be positive")
        self.beta = beta
        self.latdims = tuple(range(-ndims, 0))

    def __call__(self, fields: torch.Tensor) -> torch.Tensor:
        """Evaluate beta times the XY energy."""
        return self.beta * self.energy(fields)

    def energy(self, fields: torch.Tensor) -> torch.Tensor:
        """Return the nearest-neighbour energy of each configuration."""
        batch_shape = fields.shape[: -len(self.latdims)]
        energy = torch.zeros(batch_shape, dtype=fields.dtype, device=fields.device)
        for mu in self.latdims:
            field_forward = torch.roll(fields, -1, dims=mu)
            energy = energy - torch.cos(fields - field_forward).sum(dim=self.latdims)
        return energy
