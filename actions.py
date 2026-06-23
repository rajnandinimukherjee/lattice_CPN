import torch
import pdb


class LatticeAction():

    def __init__(self, beta):
        self.beta = beta

    def __call__(self, fields, conj=None, local=False, sites=None, batched=False):
        if local and conj is None:
            return self.beta*LocalEnergy(fields, sites)
        else:
            return self.beta*LatticeEnergy(fields, conj=conj, batched=batched)


def LatticeDdimAction(fields, beta, conj=None, local=False, sites=None, batched=False):

    if local and conj is None:
        return beta*LocalEnergy(fields, sites)
    else:
        return beta*LatticeEnergy(fields, conj=conj, batched=batched)


def LocalEnergy(fields, sites):
    latsize = fields.shape[:-1]
    S_local = torch.zeros(
        sites.shape[0], dtype=torch.cdouble, device=fields.device)
    phi = fields[tuple(sites.T)]  # (N_sites, N+1)
    phi_dag = phi.conj()

    dirs = range(len(latsize))
    for mu in dirs:
        for shift in (+1, -1):
            nbr_sites = neighbor_indices(sites, mu, latsize, shift)
            phi_nbr = fields[tuple(nbr_sites.T)]
            t1dagt2 = (phi_dag * phi_nbr).sum(-1)
            S_local += 1 - t1dagt2 * t1dagt2.conj()

    return S_local


def LatticeEnergy(fields, conj=None, batched=False):
    latsize = fields.shape[:-1]

    S = torch.zeros(latsize, dtype=torch.cdouble)
    if batched:
        latsize = latsize[1:]

    t1 = fields
    t1dag = fields.conj() if conj is None else conj

    dirs = range(1, 1+len(latsize)) if batched else range(len(latsize))

    for mu in dirs:
        t2 = torch.roll(t1, -1, dims=mu)
        t2dag = torch.roll(t1dag, -1, dims=mu)

        t1dagt2 = (t1dag*t2).sum(-1)
        t2dagt1 = (t2dag*t1).sum(-1)

        try:
            S = S + 1 - t1dagt2*t2dagt1
        except RuntimeError:
            pdb.set_trace()
    return torch.sum(S, dim=tuple(dirs))


def neighbor_indices(sites, mu, latsize, shift):
    """shift=+1 for forward, -1 for backward"""
    nbr = sites.clone()
    nbr[:, mu] = (nbr[:, mu] + shift) % latsize[mu]
    return nbr
