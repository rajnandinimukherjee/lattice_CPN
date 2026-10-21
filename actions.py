from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from CPN_utils import ComplexPair, complex_inner


def _field_zeros(
    field: torch.Tensor | ComplexPair,
    shape: tuple[int, ...],
) -> torch.Tensor | ComplexPair:
    """Create a zero scalar field matching real-pair field arithmetic."""
    zeros = torch.zeros(shape, dtype=field.dtype, device=field.device)
    if isinstance(field, ComplexPair):
        return ComplexPair(zeros, torch.zeros_like(zeros))
    return zeros


def _field_roll(
    field: torch.Tensor | ComplexPair,
    shift: int,
    dim: int,
) -> torch.Tensor | ComplexPair:
    """Roll either a tensor or both components of a complex pair."""
    if isinstance(field, ComplexPair):
        return field.roll(shift, dim)
    return torch.roll(field, shift, dims=dim)


def _field_matmul(
    field: torch.Tensor | ComplexPair,
    matrix: torch.Tensor,
) -> torch.Tensor | ComplexPair:
    """Apply a real matrix while retaining a real-pair field representation."""
    if isinstance(field, ComplexPair):
        return field.matmul(matrix)
    return field @ matrix


class _CPnAction:
    """Shared CP(N) action state and real-coordinate complex structure."""

    def __init__(
        self,
        beta: float,
        N: int,
        latsize: Sequence[int],
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize an action on a lattice of the requested size."""
        self.beta = beta
        self.N = N
        self.latsize = tuple(latsize)
        self.device = torch.device(device)

        blocks = [torch.tensor([[0, -1], [1, 0]])] * (N + 1)
        self.omega = torch.block_diag(*blocks).to(self.device)


class CPnQuarticAction(_CPnAction):
    """Quartic CP(N) lattice action in complex or real coordinates."""

    def __call__(
        self,
        fields: Any,
        *,
        local: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | ComplexPair:
        """Evaluate beta times the quartic energy."""
        if fields.shape[-1] not in {self.N + 1, 2 * (self.N + 1)}:
            raise ValueError("fields have an incompatible final dimension")
        energy = (
            self.local_energy(fields, **kwargs)
            if local
            else self.energy(fields)
        )
        return self.beta * energy

    def local_energy(
        self,
        fields: torch.Tensor | ComplexPair,
        sites: torch.Tensor,
    ) -> torch.Tensor | ComplexPair:
        """Return the quartic energy touching each selected lattice site."""
        batch_shape = fields.shape[: -len(self.latsize) - 1]
        if isinstance(fields, ComplexPair):
            energy = _field_zeros(fields, batch_shape + (sites.shape[0],))
        else:
            output_dtype = (
                fields.real.dtype
                if fields.shape[-1] == self.N + 1
                else fields.dtype
            )
            energy = torch.zeros(
                batch_shape + (sites.shape[0],),
                dtype=output_dtype,
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
                if isinstance(fields, ComplexPair):
                    omega = self.omega.to(dtype=z.dtype, device=z.device)
                    omega_z_neighbour = _field_matmul(z_neighbour, omega.T)
                    term1 = complex_inner(z, z_neighbour)
                    term2 = complex_inner(z, omega_z_neighbour)
                    energy = energy + 1 - term1 * term1 - term2 * term2
                elif fields.shape[-1] == self.N + 1:
                    link = (z.conj() * z_neighbour).sum(-1)
                    energy = energy + 1 - link.abs().square()
                else:
                    omega = self.omega.to(dtype=z.dtype, device=z.device)
                    omega_z_neighbour = z_neighbour @ omega.T
                    term1 = ((z * z_neighbour).sum(-1)).square()
                    term2 = ((z * omega_z_neighbour).sum(-1)).square()
                    energy = energy + 1 - term1 - term2
        return energy

    def energy(self, fields: torch.Tensor | ComplexPair) -> torch.Tensor | ComplexPair:
        """Return the quartic energy of each configuration."""
        lattice_dims = tuple(
            range(fields.ndim - 1 - len(self.latsize), fields.ndim - 1)
        )
        if isinstance(fields, ComplexPair):
            energy = _field_zeros(fields, fields.shape[:-1])
        else:
            output_dtype = (
                fields.real.dtype
                if fields.shape[-1] == self.N + 1
                else fields.dtype
            )
            energy = torch.zeros(
                fields.shape[:-1], dtype=output_dtype, device=fields.device
            )

        for mu in lattice_dims:
            for shift in (+1, -1):
                neighbour = _field_roll(fields, shift, mu)
                if isinstance(fields, ComplexPair):
                    omega = self.omega.to(
                        dtype=fields.dtype, device=fields.device)
                    omega_neighbour = _field_matmul(neighbour, omega.T)
                    term1 = complex_inner(fields, neighbour)
                    term2 = complex_inner(fields, omega_neighbour)
                    energy = energy + 1 - term1 * term1 - term2 * term2
                elif fields.shape[-1] == self.N + 1:
                    link = (fields.conj() * neighbour).sum(-1)
                    energy = energy + 1 - link.abs().square()
                else:
                    omega = self.omega.to(
                        dtype=fields.dtype, device=fields.device)
                    omega_neighbour = neighbour @ omega.T
                    term1 = ((fields * neighbour).sum(-1)).square()
                    term2 = ((fields * omega_neighbour).sum(-1)).square()
                    energy = energy + 1 - term1 - term2
        return energy.sum(dim=lattice_dims)


class CPnGaugeAction(_CPnAction):
    """Auxiliary-gauge CP(N) action in interleaved real coordinates.

    The vectors use ``(Re z_0, Im z_0, ...)`` and ``lam[..., mu]`` is the
    forward auxiliary-link angle. This real representation gives the
    holomorphic continuation required by contour deformations.
    """

    def __call__(
        self,
        fields: tuple[torch.Tensor | ComplexPair, torch.Tensor],
        *,
        local: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | ComplexPair:
        """Evaluate beta times the auxiliary-gauge energy."""
        energy = (
            self.local_energy(fields, **kwargs)
            if local
            else self.energy(fields)
        )
        return self.beta * energy

    def local_energy(
        self,
        fields: tuple[torch.Tensor | ComplexPair, torch.Tensor],
        sites: torch.Tensor,
    ) -> torch.Tensor | ComplexPair:
        """Return all gauge-link terms incident on selected sites."""
        x_field, lam_field = fields
        self._validate_gauge_fields(x_field, lam_field)
        batch_shape = x_field.shape[: -len(self.latsize) - 1]
        energy = _field_zeros(x_field, batch_shape + (sites.shape[0],))
        site_index = (Ellipsis, *sites.T.unbind(), slice(None))
        x = x_field[site_index]
        omega = self.omega.to(dtype=x_field.dtype, device=x_field.device)

        for mu in range(len(self.latsize)):
            forward_sites = neighbor_indices(sites, mu, self.latsize, +1)
            forward_index = (Ellipsis, *forward_sites.T.unbind(), slice(None))
            x_forward = x_field[forward_index]
            lam_forward = lam_field[(Ellipsis, *sites.T.unbind(), mu)]
            a_forward = complex_inner(x_forward, x)
            b_forward = complex_inner(x_forward, _field_matmul(x, omega.T))
            energy = energy - 2 * (
                a_forward * torch.cos(lam_forward)
                + b_forward * torch.sin(lam_forward)
            )

            # The backward bond is based at ``backward_sites`` and therefore
            # uses that site's forward link angle, not the angle at ``sites``.
            backward_sites = neighbor_indices(sites, mu, self.latsize, -1)
            backward_index = (
                Ellipsis, *backward_sites.T.unbind(), slice(None))
            x_backward = x_field[backward_index]
            lam_backward = lam_field[(
                Ellipsis, *backward_sites.T.unbind(), mu)]
            a_backward = complex_inner(x_backward, x)
            b_backward = complex_inner(x, _field_matmul(x_backward, omega.T))
            energy = energy - 2 * (
                a_backward * torch.cos(lam_backward)
                + b_backward * torch.sin(lam_backward)
            )
        return energy

    def energy(
        self,
        fields: tuple[torch.Tensor | ComplexPair, torch.Tensor],
    ) -> torch.Tensor | ComplexPair:
        """Return the auxiliary-gauge energy of each configuration."""
        x, lam = fields
        self._validate_gauge_fields(x, lam)
        lattice_dims = tuple(range(x.ndim - 1 - len(self.latsize), x.ndim - 1))
        energy = _field_zeros(x, x.shape[:-1])
        omega = self.omega.to(dtype=x.dtype, device=x.device)
        for direction, mu in enumerate(lattice_dims):
            x_forward = _field_roll(x, -1, mu)
            a = complex_inner(x_forward, x)
            b = complex_inner(x_forward, _field_matmul(x, omega.T))
            energy = energy - 2 * (
                a * torch.cos(lam[..., direction])
                + b * torch.sin(lam[..., direction])
            )
        return energy.sum(dim=lattice_dims)

    def _validate_gauge_fields(
        self,
        x: torch.Tensor | ComplexPair,
        lam: torch.Tensor,
    ) -> None:
        """Validate canonical real-vector and link-angle gauge inputs."""
        expected_components = 2 * (self.N + 1)
        if x.shape[-1] != expected_components:
            raise ValueError(
                "gauge vectors must use interleaved real components "
                f"with final size {expected_components}"
            )
        lattice_shape = tuple(x.shape[-1 - len(self.latsize): -1])
        expected_lam_shape = x.shape[: -len(self.latsize) - 1] + lattice_shape + (
            len(self.latsize),
        )
        if tuple(lam.shape) != tuple(expected_lam_shape):
            raise ValueError(
                "gauge links must have shape (*batch, *lattice, ndim); got "
                f"{tuple(lam.shape)}, expected {tuple(expected_lam_shape)}"
            )


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
        energy = torch.zeros(
            batch_shape, dtype=fields.dtype, device=fields.device)
        for mu in self.latdims:
            field_forward = torch.roll(fields, -1, dims=mu)
            energy = energy - \
                torch.cos(fields - field_forward).sum(dim=self.latdims)
        return energy
