import pdb

import torch
import torch.nn as nn
from torch.func import jacrev, vmap

from CPN_utils import Lambda


class FlowedDeformations(nn.Module):
    name = "FlowDef"

    def __init__(self, N):
        super(FlowedDeformations, self).__init__()
        self.N = N
        self.DOF = 2*(N+1)

    def __repr__(self):
        return self.name

    def forward(self, X, f):
        proj = torch.sum(f*X, dim=-1, keepdim=True)
        F_X = f - proj*X
        divF = -(self.DOF+1)*proj.squeeze(-1)
        return F_X, divF.sum(dim=(-1, -2))


class HomogenousDeformations(nn.Module):
    name = "HomogDef"

    def __init__(self, N, device='cpu'):
        super(HomogenousDeformations, self).__init__()
        self.N = N
        self.DOF = N+1
        self.eye = torch.eye(2*self.DOF).to(device)

        blocks = [torch.tensor([[0, -1], [1, 0]])] * (N+1)
        self.omega = torch.block_diag(*blocks).to(torch.float32).to(device)

    def __repr__(self):
        return self.name

    def deformx(self, X, alphas):
        assert alphas.shape[-1] == self.DOF, "Alphas of incorrect size"

        A = torch.diag_embed(
            torch.repeat_interleave(alphas, repeats=2, dim=-1)).unsqueeze(0)
        Y = torch.einsum('...ij,...j->...i', A@self.omega, X)
        return Y

    def complexify(self, X, alphas):
        Y = self.deformx(X, alphas)
        LX = X * Lambda(Y).unsqueeze(-1)
        Z = LX + 1j * Y
        return Z

    def detJac(self, X, alphas):
        batch_idx = X.shape[:-1]
        detJ = torch.ones(*batch_idx)

        Y = self.deformx(X, alphas)
        L = Lambda(Y)
        L2dim = L.unsqueeze(-1).unsqueeze(-1)
        A = torch.diag_embed(
            torch.repeat_interleave(alphas, repeats=2, dim=-1))
        A2 = torch.einsum('...ij, ...jk->...ik', A, A)

        J = L2dim * self.eye
        J += torch.einsum('...ij,...j,...k->...ik', A2, X, X)/L2dim
        J = J.to(torch.complex128)
        J -= 1j * (A@self.omega)
        detJ = torch.linalg.det(J)/(L**2)

        return detJ
