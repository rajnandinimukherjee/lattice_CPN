from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any

import matplotlib.pyplot as plt
import torch


def cpn_one_pt_fn(
    Z: torch.Tensor,
    i: int,
    j: int,
    coord: Sequence[int],
) -> torch.Tensor:
    """Evaluate ``z_i z_j^*`` at one lattice coordinate."""
    Z_pt = Z[..., *coord, :].movedim(-1, 0)
    Z_i = Z_pt[2 * i] + 1j * Z_pt[2 * i + 1]
    Z_j_bar = Z_pt[2 * j] - 1j * Z_pt[2 * j + 1]
    return Z_i * Z_j_bar


class CPnOnePtFn:
    """Connected one-point observable for a CP(N) field."""

    def __init__(
        self,
        N: int,
        latsize: Sequence[int],
        i: int = 0,
        j: int = 0,
        source: Sequence[int] = (0, 0),
        action: Any = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize a one-point observable and network input grid."""
        assert i <= N and j <= N, f"i, j need to be <={N}"
        self.i = i
        self.j = j
        self.latsize = tuple(latsize)
        self.source = tuple(source)
        self.action = action
        self.N_lat = prod(self.latsize)
        self.device = device

        self.grid = torch.zeros(*self.latsize, 2 * (N + 1), dtype=torch.float32).to(
            device
        )
        self.grid[*self.source, i] += 1
        self.grid[*self.source, j] += 1

    def __repr__(self) -> str:
        """Return a compact mathematical label."""
        Op = r"z_" + str(self.i) + r"z^\dagger_" + str(self.j)
        return r"$" + Op + str(self.source) + r"$"

    def observable(self, Z: torch.Tensor) -> torch.Tensor:
        """Evaluate and center the bare one-point observable."""
        obs = cpn_one_pt_fn(Z, self.i, self.j, self.source)
        return obs - obs.mean()

    def __call__(
        self,
        model: Any,
        deform: bool = False,
        alphas: torch.Tensor | None = None,
        batch_idx: torch.Tensor | slice = slice(None),
        sampletype: str = "all",
        return_fields: bool = False,
        return_jac: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Evaluate the observable, optionally including contour weights."""

        X = model.samples[sampletype]
        if sampletype == "train":
            X = X[batch_idx]

        if deform:
            Z = model.complexify(X, alphas)
            if return_fields:
                return Z

            Sdiff = torch.exp(-self.action(Z) + self.action(X))
            detJ = model.detJac(X, alphas)
            if return_jac:
                return detJ

            detJ = torch.prod(detJ.view(-1, self.N_lat), dim=-1)
            return self.observable(Z) * detJ * Sdiff
        else:
            return self.observable(X)


class CPnTwoPtFn:
    """Connected two-point observable for a CP(N) field."""

    def __init__(
        self,
        N: int,
        latsize: Sequence[int],
        i: int = 0,
        j: int = 0,
        source: Sequence[int] = (0, 0),
        delta: Sequence[int] = (0, 1),
        trans_inv: bool = False,
        action: Any = None,
        myAlphaScale: float = 0.3,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize source, sink, and deformation seed fields."""
        assert i <= N and j <= N, f"i, j need to be <={N}"
        self.i = i
        self.j = j
        self.latsize = tuple(latsize)
        self.delta = tuple(delta)
        self.source = tuple(source)
        self.sink = tuple(
            (coordinate + displacement) % length
            for coordinate, displacement, length in zip(
                self.source, self.delta, self.latsize
            )
        )
        self.action = action
        self.N_lat = prod(self.latsize)
        self.device = device

        self.grid = torch.zeros(*self.latsize, 2 * (N + 1), dtype=torch.float32).to(
            device
        )
        self.grid[*self.source, i] += 1
        self.grid[*self.source, j] += 1
        self.grid[*self.sink, (N + 1) + i] += 1
        self.grid[*self.sink, (N + 1) + j] += 1
        self.invSqFunc = invSquareFunc(self.latsize, N)
        self.invSqAlphas = myAlphaScale * self.invSqFunc(
            self.source, self.sink, i, j
        ).to(device)

    def __repr__(self) -> str:
        """Return a compact mathematical label."""
        Op = r"z_" + str(self.i) + r"z^\dagger_" + str(self.j)
        return (
            r"$"
            + Op
            + str(self.source)
            + r"["
            + Op
            + r"]^\dagger"
            + str(self.sink)
            + r"$"
        )

    def observable(self, Z: torch.Tensor) -> torch.Tensor:
        """Evaluate and center the bare two-point observable."""
        opf1 = cpn_one_pt_fn(Z, self.i, self.j, self.source)
        opf2_dag = cpn_one_pt_fn(Z, self.j, self.i, self.sink)
        obs = opf1 * opf2_dag
        return obs - obs.mean()

    def __call__(
        self,
        model: Any,
        deform: bool = False,
        alphas: torch.Tensor | None = None,
        batch_idx: torch.Tensor | slice = slice(None),
        sampletype: str = "all",
        return_fields: bool = False,
        return_jac: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Evaluate the observable, optionally including contour weights."""

        X = model.samples[sampletype]
        if sampletype == "train":
            X = X[batch_idx]

        if deform:

            Z = model.complexify(X, alphas)
            if return_fields:
                return Z

            Sdiff = torch.exp(-self.action(Z) + self.action(X))
            detJ = model.detJac(X, alphas)
            if return_jac:
                return detJ

            detJ = torch.prod(detJ.view(-1, self.N_lat), dim=-1)
            return self.observable(Z) * detJ * Sdiff
        else:
            return self.observable(X)


class XYOnePtFn:
    """Complex one-point observable for an XY field."""

    def __init__(
        self,
        latsize: Sequence[int],
        source: Sequence[int],
        action: Any,
    ) -> None:
        """Initialize an XY one-point observable and input grid."""
        self.latsize = tuple(latsize)
        self.source = tuple(source)
        self.action = action

        self.grid = torch.zeros(*latsize, 1, dtype=torch.float32)
        self.grid[*source, 0] += 1
        self.latdims = tuple(range(len(latsize)))

    def __repr__(self) -> str:
        """Return a label containing the observable and ensemble details."""
        obs = r"Re$\left(e^{i\theta" + str(self.source) + r"}\right)$"
        beta = r"$\beta=" + f"{self.action.beta:2g}" + r"$"
        lattice = "x".join([f"{L}" for L in self.latsize])

        return ", ".join([obs, beta, lattice])

    def observable(self, fields: torch.Tensor) -> torch.Tensor:
        """Evaluate the bare complex exponential."""
        theta1 = fields[:, *self.source]
        obs = torch.exp(1j * theta1)
        return obs - obs.mean()

    def __call__(
        self,
        model: Any,
        deform: bool = False,
        alphas: torch.Tensor | None = None,
        batch_idx: torch.Tensor | slice = slice(None),
        get_Sdiff: bool = False,
        sampletype: str = "all",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Evaluate the observable with an optional additive deformation."""
        fields = model.samples[sampletype]
        if sampletype == "train":
            fields = fields[batch_idx]

        if deform:
            S_undef = self.action(fields)

            fields = fields + 1j * alphas[..., 0]
            S_def = self.action(fields)
            Sdiff = S_def - S_undef

            if get_Sdiff:
                return Sdiff
            return self.observable(fields) * torch.exp(-Sdiff)
        else:
            return self.observable(fields)


class XYTwoPtFn:
    """Complex two-point observable for an XY field."""

    def __init__(
        self,
        latsize: Sequence[int],
        source: Sequence[int],
        sink: Sequence[int],
        action: Any,
    ) -> None:
        """Initialize an XY two-point observable and input grid."""
        self.latsize = tuple(latsize)
        self.source = tuple(source)
        self.sink = tuple(
            coordinate % length for coordinate, length in zip(sink, self.latsize)
        )
        self.action = action

        self.grid = torch.zeros(*latsize, 2, dtype=torch.float32)
        self.grid[*source, 0] += 1
        self.grid[*sink, 1] += 1

    def __repr__(self) -> str:
        """Return a label containing the observable and ensemble details."""
        obs = (
            r"Re$\left(e^{i\theta"
            + str(self.source)
            + r"-i\theta"
            + str(self.sink)
            + r"}\right)$"
        )
        beta = r"$\beta=" + f"{self.action.beta:2g}" + r"$"
        lattice = f"{self.latsize[0]}x{self.latsize[1]}"

        return ", ".join([obs, beta, lattice])

    def observable(self, fields: torch.Tensor) -> torch.Tensor:
        """Evaluate the bare phase-difference exponential."""
        theta1 = fields[:, *self.source]
        theta2 = fields[:, *self.sink]
        return torch.exp(1j * (theta1 - theta2))

    def __call__(
        self,
        model: Any,
        deform: bool = False,
        alphas: torch.Tensor | None = None,
        batch_idx: torch.Tensor | slice = slice(None),
        get_Sdiff: bool = False,
        sampletype: str = "all",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Evaluate the observable with an optional additive deformation."""
        fields = model.samples[sampletype]
        if sampletype == "train":
            fields = fields[batch_idx]

        if deform:
            S_undef = self.action(fields)

            fields = fields + 1j * alphas[..., 0]
            S_def = self.action(fields)
            Sdiff = S_def - S_undef

            if get_Sdiff:
                return Sdiff
            return self.observable(fields) * torch.exp(-Sdiff)
        else:
            return self.observable(fields)


class invSquareFunc:
    """Periodic inverse-square seed field for CP(N) deformations."""

    def __init__(self, latsize: Sequence[int], N: int) -> None:
        """Create coordinate grids for the requested lattice."""
        self.latsize = tuple(latsize)
        self.N = N
        self.grid = torch.meshgrid(
            [torch.arange(L) for L in self.latsize], indexing="ij"
        )

    def wrap(self, coord: torch.Tensor) -> torch.Tensor:
        """Wrap coordinate differences to the nearest periodic image."""
        L = torch.tensor(self.latsize, dtype=coord.dtype)
        halfL = L // 2
        return (coord + halfL) % L - halfL

    def plot(self, source: Sequence[int], sink: Sequence[int]) -> None:
        """Plot one component of the seed field."""
        field = self(source, sink, 0, 1)[:, :, 0]
        fig, ax = plt.subplots(1, 1)
        ax.imshow(field, cmap="PuOr")
        ax.set_aspect(1.0)
        plt.show()

    def __call__(
        self,
        source: Sequence[int],
        sink: Sequence[int],
        i: int,
        j: int,
    ) -> torch.Tensor:
        """Construct the signed inverse-square field for two components."""
        ldims = len(self.latsize)
        src_diffs = torch.stack(
            [self.grid[d] - source[d] for d in range(ldims)], dim=ldims
        )
        snk_diffs = torch.stack(
            [self.grid[d] - sink[d] for d in range(ldims)], dim=ldims
        )

        dsrc_sq = (self.wrap(src_diffs) ** 2).sum(dim=ldims)
        dsnk_sq = (self.wrap(snk_diffs) ** 2).sum(dim=ldims)
        field = 1.0 / (dsrc_sq + 1.0) - 1.0 / (dsnk_sq + 1.0)
        A = torch.zeros(*self.latsize, self.N + 1, dtype=torch.float32)
        A[..., i], A[..., j] = field, -field
        return A
