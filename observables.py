import pdb
from math import prod

import torch

import matplotlib.pyplot as plt


def one_pt_fn(Z, i, j, coord):
    Z_pt = Z[..., *coord, :].movedim(-1, 0)
    Z_i = Z_pt[2*i]+1j*Z_pt[2*i+1]
    Z_j_bar = Z_pt[2*j]-1j*Z_pt[2*j+1]
    return Z_i*Z_j_bar


class OnePtFn:

    def __init__(self, N, latsize, i=0, j=0, coord=(8, 8),
                 action=None, device='cpu'):
        assert (i <= N and j <= N), f"i, j need to be <={N}"
        self.i = i
        self.j = j
        self.latsize = latsize
        self.c1 = coord
        self.action = action
        self.N_lat = prod(self.latsize)
        self.device=device

        self.grid = torch.zeros(*latsize, 2*(N+1), dtype=torch.float32).to(device)
        self.grid[*self.c1, i] += 1
        self.grid[*self.c1, j] += 1

    def __repr__(self):
        Op = r"z_"+str(self.i)+r"z^\dagger_"+str(self.j)
        return r"$"+Op+str(self.c1)+r"$"

    def observable(self, Z):
        obs = one_pt_fn(Z, self.i, self.j, self.c1)
        return obs-obs.mean()

    def __call__(self, model,
                 deform=False, alphas=None, batch_idx=slice(None),
                 sampletype="all", return_fields=False, return_jac=False,
                 **kwargs):

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
            return self.observable(Z)*detJ*Sdiff
        else:
            return self.observable(X)


class TwoPtFn:

    def __init__(self, N, latsize, i=0, j=0, coord=(8, 8),
                 delta=(0, 1), trans_inv=False, action=None,
                 myAlphaScale=0.3, device='cpu'):
        assert (i <= N and j <= N), f"i, j need to be <={N}"
        self.i = i
        self.j = j
        self.latsize = latsize
        self.delta = delta
        self.c1 = coord
        self.c2 = tuple(map(sum, zip(coord, self.delta)))
        self.action = action
        self.N_lat = prod(self.latsize)
        self.device = device

        self.grid = torch.zeros(*latsize, 2*(N+1), dtype=torch.float32).to(device)
        self.grid[*self.c1, i] += 1
        self.grid[*self.c1, j] += 1
        self.grid[*self.c2, (N+1)+i] += 1
        self.grid[*self.c2, (N+1)+j] += 1
        self.invSqFunc = invSquareFunc(latsize, N).to(device)
        self.invSqAlphas = myAlphaScale*self.invSqFunc(self.c1, self.c2, i, j)

    def __repr__(self):
        Op = r"z_"+str(self.i)+r"z^\dagger_"+str(self.j)
        return r"$"+Op+str(self.c1)+r"["+Op+r"]^\dagger"+str(self.c2)+r"$"

    def observable(self, Z):
        opf1 = one_pt_fn(Z, self.i, self.j, self.c1)
        opf2_dag = one_pt_fn(Z, self.j, self.i, self.c2)
        obs = opf1*opf2_dag
        return obs - obs.mean()

    def __call__(self, model,
                 deform=False, alphas=None, batch_idx=slice(None),
                 sampletype="all", return_fields=False,
                 return_jac=False, **kwargs):

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
            return self.observable(Z)*detJ*Sdiff
        else:
            return self.observable(X)


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
        A = torch.zeros(*self.latsize, self.N+1, dtype=torch.float32)
        A[..., i], A[..., j] = field, -field
        return A
