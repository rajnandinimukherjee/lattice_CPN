import torch
import pdb


class LatticeAction():

    def __init__(self, beta, N, latsize, 
                 style="quartic", device="cpu"):
        self.beta, self.N = beta, N
        blocks = [torch.tensor([[0, -1], [1, 0]])] * (N+1)
        self.omega = torch.block_diag(*blocks).to(device)
        self.latsize = latsize
        self.device = device

        assert style in ["quartic", "gauge"]
        if style == "quartic":
            self.action = self.quartic_action
            self.energy = self.lattice_quartic_energy
            self.local_energy = self.local_quartic_energy

    def __call__(self, fields, **kwargs):
        assert fields.shape[-1] == 2*(self.N+1), "fields must be in real rep"
        return self.action(fields, **kwargs)

    def quartic_action(self, fields, local=False, **kwargs):
        if local:
            return self.beta*self.local_quartic_energy(fields, **kwargs)
        else:
            return self.beta*self.lattice_quartic_energy(fields, **kwargs)

    def gauge_action(self, fields, local=False, **kwargs):
        pass

    def local_quartic_energy(self, fields, sites):
        batch_shape = fields.shape[:-3]
        E_local = torch.zeros(
            *batch_shape, sites.shape[0], dtype=torch.cdouble).to(self.device)
        Z = fields[..., sites[:, 0], sites[:, 1], :]  # (N_sites, N+1)
        omega = self.omega.to(Z.dtype)

        dirs = range(len(self.latsize))
        for mu in dirs:
            for shift in (+1, -1):
                nbr_sites = neighbor_indices(sites, mu, self.latsize, shift).to(self.device)
                Z_nbr = fields[..., nbr_sites[:, 0], nbr_sites[:, 1], :]
                OmegaZ_nbr = Z_nbr@omega.T
                term1 = ((Z*Z_nbr).sum(-1))**2
                term2 = ((Z*OmegaZ_nbr).sum(-1))**2
                E_local += 1 - (term1 + term2)

        return E_local

    def lattice_quartic_energy(self, Z):
        # Z are the fields
        E = torch.zeros(Z.shape[:-1]).to(self.device)
        N_ax = len(Z.shape)-1
        dirs = range(N_ax-len(self.latsize), N_ax)
        omega = self.omega.to(Z.dtype)
        for mu in dirs:
            for shift in (+1, -1):
                Z_nbr = torch.roll(Z, shift, dims=mu)
                OmegaZ_nbr = Z_nbr@omega.T
                term1 = ((Z*Z_nbr).sum(-1))**2
                term2 = ((Z*OmegaZ_nbr).sum(-1))**2
                E = E + 1 - (term1+term2)
        return torch.sum(E, dim=tuple(dirs))

    def local_gauge_energy(self, fields, sites):
        Z, thetas = fields
        batch_shape = Z.shape[:-3]
        E_local = torch.zeros(
            *batch_shape, sites.shape[0], dtype=torch.cdouble)
        Z = Z[..., sites[:, 0], sites[:, 1], :]  # (N_sites, N+1)
        thetas = thetas[..., sites[:, 0], sites[:, 1]]  # (N_sites, N+1)
        omega = self.omega.to(Z.dtype)

        dirs = range(len(self.latsize))
        for mu in dirs:
            for shift in (+1, -1):
                nbr_sites = neighbor_indices(sites, mu, self.latsize, shift)
                Z_nbr = fields[..., nbr_sites[:, 0], nbr_sites[:, 1], :]
                OmegaZ_nbr = Z_nbr@omega.T
                term1 = ((Z*Z_nbr).sum(-1))*torch.cos(shift*thetas)
                term2 = ((Z*OmegaZ_nbr).sum(-1))*torch.sin(shift*thetas)
                E_local += 1 - (term1 + term2)
        return E_local

    def lattice_gauge_energy(self, fields):
        Z, thetas = fields
        E = torch.zeros(Z.shape[:-1])
        N_ax = len(Z.shape)-1
        dirs = range(N_ax-len(self.latsize), N_ax)
        omega = self.omega.to(Z.dtype)
        for mu in dirs:
            for shift in (+1, -1):
                Z_nbr = torch.roll(Z, shift, dims=mu)
                OmegaZ_nbr = Z_nbr@omega.T
                term1 = ((Z*Z_nbr).sum(-1))*torch.cos(shift*thetas)
                term2 = ((Z*OmegaZ_nbr).sum(-1))*torch.sin(shift*thetas)
                E = E + 1 - (term1+term2)
        return torch.sum(E, dim=tuple(dirs))


def neighbor_indices(sites, mu, latsize, shift):
    """shift=+1 for forward, -1 for backward"""
    nbr = sites.clone()
    nbr[:, mu] = (nbr[:, mu] + shift) % latsize[mu]
    return nbr
