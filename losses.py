import torch
import torch.nn as nn
from CPN_utils import CNtoR2N
import pdb
from training_utils import bootstrap


class VarianceLoss(nn.Module):

    def __init__(self, model, samples, N, obs):
        super(VarianceLoss, self).__init__()
        self.model = model
        self.samples = samples
        self.N = N
        self.obs = obs

        self.init = self.get_obs(deform=False, sampletype="all")
        self.target_exp = self.init.mean().item()
        self.initial_var = self.init.var().item()
        self.exp_err = bootstrap(self.init).var().item()**0.5

    def get_obs(self, alphas=None, batch_idx=None,
                deform=True, sampletype="train"):
        return self.obs(model=self.model,
                        deform=deform,
                        alphas=alphas,
                        batch_idx=batch_idx,
                        sampletype=sampletype).real

    def forward(self, alphas=None, **kwargs):
        return self.var(alphas=alphas, **kwargs)

    def var(self, **kwargs):
        return self.get_obs(**kwargs).var()

    def exp(self, **kwargs):
        return self.get_obs(**kwargs).mean()

    def StN(self, **kwargs):
        obs = self.get_obs(**kwargs)
        return obs.mean()/obs.var()


class PINNLoss(nn.Module):

    def __init__(self, model, samples, N, obs):
        super(PINNLoss, self).__init__()
        self.model = model
        self.samples = samples
        self.N = N
        self.obs = obs
        self.action = self.obs.action

    def grad_action(self, X):
        def action(X):
            x = X[..., ::2] + 1j * X[..., 1::2]
            xconj = X[..., ::2] - 1j * X[..., 1::2]
            return self.action(x, conj=xconj).real
        return torch.func.vmap(torch.func.jacrev(action))(X)

    def grad_obs(self, X):

        def obs_re(X):
            x = X[..., ::2] + 1j * X[..., 1::2]
            xconj = X[..., ::2] - 1j * X[..., 1::2]
            return self.obs.O(x, xconj).real

        def obs_im(X):
            x = X[..., ::2] + 1j * X[..., 1::2]
            xconj = X[..., ::2] - 1j * X[..., 1::2]
            return self.obs.O(x, xconj).imag
        return torch.func.vmap(torch.func.jacrev(obs_re))(X) +\
            1j * torch.func.vmap(torch.func.jacrev(obs_im))(X)

    def get_F(self, X, alphas):
        F_re, divF_re = self.model(X, alphas[..., :self.model.DOF])
        F_im, divF_im = self.model(X, alphas[..., self.model.DOF:])
        F = F_re + 1j*F_im
        divF = divF_re + 1j*divF_im
        return F, divF

    def PINN(self, alphas, sampletype="all", batch_idx=slice(None),
             FS_only=False):
        x = self.model.samples[sampletype]
        if sampletype == "train":
            x = x[batch_idx]

        X = CNtoR2N(x)
        F, divF = self.get_F(X, alphas)

        inds = tuple(range(1, len(F.shape)))
        FgradS = (F*self.grad_action(X)).sum(inds)
        FS = divF-FgradS
        if FS_only:
            return FS

        Q = self.obs.Q(x, x.conj())
        # Qexp = Q.mean()
        Qexp = 1 if self.obs.i == self.obs.j else 0

        return FS - Q + Qexp

    def get_obs(self, alphas, sampletype="all", batch_idx=slice(None)):
        x = self.model.samples[sampletype]
        if sampletype == "train":
            x = x[batch_idx]
        X = CNtoR2N(x)

        O = self.obs.O(x, x.conj())
        gradO = self.grad_obs(X)
        F, _ = self.get_F(X, alphas)

        inds = tuple(range(1, len(F.shape)))
        FgradO = (F*gradO).sum(inds)
        PINN = self.PINN(alphas, sampletype=sampletype,
                         batch_idx=batch_idx)
        return - (FgradO + PINN*O)

    def forward(self, **kwargs):
        # return self.var(**kwargs)
        PINN = self.PINN(**kwargs)
        return (PINN*PINN.conj()).real.mean()

    def exp(self, **kwargs):
        return self.get_obs(**kwargs).mean()

    def var(self, **kwargs):
        return self.get_obs(**kwargs).var()
