import pdb
from math import prod

import torch

from CPN_utils import CNtoR2N
import matplotlib.pyplot as plt


class OnePtFn:

    def __init__(self, N, latsize, i=0, j=0, coord=(8, 8),
                 action=None):
        assert (i <= N and j <= N), f"i, j need to be <={N}"
        self.i = i
        self.j = j
        self.latsize = latsize
        self.c1 = coord
        self.action = action
        self.N_lat = prod(self.latsize)

        self.grid = torch.zeros(*latsize, 2*(N+1), dtype=torch.double)
        self.grid[*self.c1, i] += 1
        self.grid[*self.c1, j] += 1

    def __repr__(self):
        Op = r"z_"+str(self.i)+r"z^\dagger_"+str(self.j)
        return r"$"+Op+str(self.c1)+r"$"

    def Q(self, z, zconj):
        obs = z[..., *self.c1, self.i]*zconj[..., *self.c1, self.j]
        return obs

    def observable(self, z, zconj):
        Q = self.Q(z, zconj)
        return Q-Q.mean()

    def __call__(self, model,
                 deform=False, alphas=None, batch_idx=slice(None),
                 sampletype="all", return_fields=False, return_jac=False,
                 **kwargs):

        x = model.samples[sampletype]
        if sampletype == "train":
            x = x[batch_idx]

        if deform:

            X = CNtoR2N(x)
            Z = model.complexify(X, alphas)

            z = Z[..., ::2] + 1j * Z[..., 1::2]
            zconj = Z[..., ::2] - 1j * Z[..., 1::2]

            if return_fields:
                return z, zconj

            Sdiff = torch.exp(-self.action(z, conj=zconj, batched=True) +
                              self.action(x, batched=True))
            detJ = model.detJac(X, alphas)

            if return_jac:
                return detJ

            detJ = torch.prod(detJ.view(-1, self.N_lat), dim=-1)

            return self.observable(z, zconj)*detJ*Sdiff
        else:
            return self.observable(x, x.conj())


class TwoPtFn:

    def __init__(self, N, latsize, i=0, j=0, coord=(8, 8),
                 delta=(0, 1), trans_inv=False, action=None,
                 myAlphaScale=0.3):
        assert (i <= N and j <= N), f"i, j need to be <={N}"
        self.i = i
        self.j = j
        self.latsize = latsize
        self.delta = delta
        self.c1 = coord
        self.c2 = tuple(map(sum, zip(coord, self.delta)))
        self.action = action
        self.N_lat = prod(self.latsize)

        self.grid = torch.zeros(*latsize, 2*(N+1), dtype=torch.double)
        self.grid[*self.c1, i] += 1
        self.grid[*self.c1, j] += 1
        self.grid[*self.c2, (N+1)+i] += 1
        self.grid[*self.c2, (N+1)+j] += 1
        self.invSqFunc = invSquareFunc(latsize, N)
        self.invSqAlphas = myAlphaScale*self.invSqFunc(self.c1, self.c2, i, j)

    def __repr__(self):
        Op = r"z_"+str(self.i)+r"z^\dagger_"+str(self.j)
        return r"$"+Op+str(self.c1)+r"["+Op+r"]^\dagger"+str(self.c2)+r"$"

    def O(self, z, zconj):
        obs = zconj[..., *self.c2, self.i]*z[..., *self.c2, self.j]
        return obs

    def Q(self, z, zconj):
        obs = z[..., *self.c1, self.i]*zconj[..., *self.c1, self.j]
        return obs

    def observable(self, z, zconj):
        O = self.O(z, zconj)
        Q = self.Q(z, zconj)
        return O*Q  # -O.mean()*Q.mean()

    def __call__(self, model,
                 deform=False, alphas=None, batch_idx=slice(None),
                 sampletype="all", return_fields=False,
                 return_jac=False, **kwargs):

        x = model.samples[sampletype]
        if sampletype == "train":
            x = x[batch_idx]

        if deform:

            X = CNtoR2N(x)
            Z = model.complexify(X, alphas)

            z = Z[..., ::2] + 1j * Z[..., 1::2]
            zconj = Z[..., ::2] - 1j * Z[..., 1::2]
            if return_fields:
                return z, zconj

            Sdiff = torch.exp(-self.action(z, conj=zconj, batched=True) +
                              self.action(x, batched=True))
            detJ = model.detJac(X, alphas)
            if return_jac:
                return detJ

            detJ = torch.prod(detJ.view(-1, self.N_lat), dim=-1)

            return self.observable(z, zconj)*detJ*Sdiff
        else:
            return self.observable(x, x.conj())


class invSquareFunc:

    def __init__(self, latsize, N):
        self.latsize = latsize
        self.N = N
        self.grid = torch.meshgrid(
            [torch.arange(L) for L in self.latsize], indexing='ij')

    def wrap(self, coord):
        L = torch.tensor(self.latsize, dtype=coord.dtype)
        halfL = L//2
        return (coord+halfL) % L - halfL

    def plot(self, c1, c2):
        field = self(c1, c2, 0, 1)[:, :, 0]
        fig, ax = plt.subplots(1, 1)
        ax.imshow(field, cmap="PuOr")
        ax.set_aspect(1.0)
        plt.show()

    def __call__(self, c1, c2, i, j):
        ldims = len(self.latsize)
        src_diffs = torch.stack([self.grid[d] - c1[d]
                                 for d in range(ldims)], dim=ldims)
        snk_diffs = torch.stack([self.grid[d] - c2[d]
                                 for d in range(ldims)], dim=ldims)

        dsrc2 = (self.wrap(src_diffs) ** 2).sum(dim=ldims)
        dsnk2 = (self.wrap(snk_diffs) ** 2).sum(dim=ldims)
        field = 1.0 / (dsrc2 + 1.0) - 1.0 / (dsnk2 + 1.0)
        A = torch.zeros(*self.latsize, self.N+1, dtype=torch.double)
        A[..., i], A[..., j] = field, -field
        return A
