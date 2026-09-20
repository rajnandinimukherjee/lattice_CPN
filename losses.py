from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from CPN_utils import CNtoR2N
from training import bootstrap


class VarianceLoss(nn.Module):
    """Variance objective for a model-deformed observable."""

    objective_name = "variance"
    selection_metric = "var"

    def __init__(self, model: nn.Module, obs: Any) -> None:
        """Record the undeformed observable statistics."""
        super().__init__()
        self.model = model
        self.obs = obs

        self.init = self.get_obs(deform=False, sampletype="all")
        self.target_exp = self.init.mean().item()
        self.initial_var = self.init.var().item()
        self.exp_err = bootstrap(self.init).var().item() ** 0.5

    def get_obs(
        self,
        alphas: torch.Tensor | None = None,
        batch_idx: torch.Tensor | slice = slice(None),
        deform: bool = True,
        sampletype: str = "train",
    ) -> torch.Tensor:
        """Evaluate the real part of the configured observable."""
        return self.obs(
            model=self.model,
            deform=deform,
            alphas=alphas,
            batch_idx=batch_idx,
            sampletype=sampletype,
        ).real

    def forward(
        self,
        alphas: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return the observable variance used for optimization."""
        return self.var(alphas=alphas, **kwargs)

    def var(self, **kwargs: Any) -> torch.Tensor:
        """Return the observable variance."""
        return self.get_obs(**kwargs).var()

    def exp(self, **kwargs: Any) -> torch.Tensor:
        """Return the observable expectation value."""
        return self.get_obs(**kwargs).mean()

    def StN(self, **kwargs: Any) -> torch.Tensor:
        """Return the expectation divided by its standard error."""
        obs = self.get_obs(**kwargs)
        return obs.mean() / (obs.var() / len(obs)).sqrt()

    def mod_squared(self, **kwargs: Any) -> float:
        """Return the mean squared modulus of the observable."""
        deform = bool(kwargs.pop("deform", True))
        obs = self.obs(model=self.model, deform=deform, **kwargs)
        return torch.abs(obs).square().mean().item()

    def get_stats(self, **kwargs: Any) -> tuple[float, float]:
        """Return the bootstrapped mean and its standard error."""
        bootstrapped = bootstrap(self.get_obs(**kwargs))
        return bootstrapped.mean().item(), bootstrapped.var().item() ** 0.5


class PenaltyLoss(nn.Module):
    """Penalty objective for a model-deformed observable."""

    objective_name = "penalty"
    selection_metric = "exp"

    def __init__(self, model: nn.Module, action: Any, obs: Any = None) -> None:
        """Record the undeformed penalty statistics."""
        super().__init__()
        self.model = model
        self.action = action
        # ``Trainer`` obtains the fixed network input grid from the observable.
        # Keep this optional so PenaltyLoss remains usable on its own.
        self.obs = obs

        self.init = self.get_obs(deform=False, sampletype="all")
        self.target_exp = self.init.mean().item()
        self.initial_var = self.init.var().item()
        self.exp_err = bootstrap(self.init).var().item() ** 0.5

    def penalty_action(
        self,
        alphas: torch.Tensor | None = None,
        batch_idx: torch.Tensor | slice = slice(None),
        deform: bool = True,
        sampletype: str = "train",
    ) -> torch.Tensor:
        """Evaluate the real part of the action difference."""
        X = self.model.samples[sampletype]
        if sampletype == "train":
            X = X[batch_idx]

        S_undef = self.action(X)
        if deform:
            X = self.model.complexify(X, alphas[..., 0])
        S_def = self.action(X)
        Re_S_diff = (S_def - S_undef).real
        return torch.exp(-2*Re_S_diff)

    def get_obs(self, **kwargs: Any) -> torch.Tensor:
        """Return per-configuration penalty values for trainer diagnostics."""
        return self.penalty_action(**kwargs)

    def forward(
        self,
        alphas: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return the mean penalty used for optimization."""
        return self.exp(alphas=alphas, **kwargs)

    def exp(self, **kwargs: Any) -> torch.Tensor:
        """Return the observable expectation value."""
        return self.penalty_action(**kwargs).mean()

    def get_stats(self, **kwargs: Any) -> tuple[float, float]:
        """Return the bootstrapped mean and its standard error."""
        bootstrapped = bootstrap(self.penalty_action(**kwargs))
        return bootstrapped.mean().item(), bootstrapped.var().item() ** 0.5


class PINNLoss(nn.Module):
    """Physics-informed objective based on a learned tangent vector field."""

    def __init__(
        self,
        model: nn.Module,
        samples: torch.Tensor,
        N: int,
        obs: Any,
    ) -> None:
        """Initialize the model, samples, and observable."""
        super().__init__()
        self.model = model
        self.samples = samples
        self.N = N
        self.obs = obs
        self.action = self.obs.action

    def grad_action(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiate the action with respect to real field components."""

        def action(real_x: torch.Tensor) -> torch.Tensor:
            """Evaluate the real action from interleaved coordinates."""
            complex_x = real_x[..., ::2] + 1j * real_x[..., 1::2]
            complex_conj = real_x[..., ::2] - 1j * real_x[..., 1::2]
            return self.action(complex_x, conj=complex_conj).real

        return torch.func.vmap(torch.func.jacrev(action))(x)

    def grad_obs(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiate the complex observable over real field components."""

        def obs_re(real_x: torch.Tensor) -> torch.Tensor:
            """Evaluate the real part of the bare observable."""
            complex_x = real_x[..., ::2] + 1j * real_x[..., 1::2]
            complex_conj = real_x[..., ::2] - 1j * real_x[..., 1::2]
            return self.obs.O(complex_x, complex_conj).real

        def obs_im(real_x: torch.Tensor) -> torch.Tensor:
            """Evaluate the imaginary part of the bare observable."""
            complex_x = real_x[..., ::2] + 1j * real_x[..., 1::2]
            complex_conj = real_x[..., ::2] - 1j * real_x[..., 1::2]
            return self.obs.O(complex_x, complex_conj).imag

        jacobian = torch.func.jacrev
        return torch.func.vmap(jacobian(obs_re))(x) + 1j * torch.func.vmap(
            jacobian(obs_im)
        )(x)

    def get_F(
        self,
        x: torch.Tensor,
        alphas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the complex vector field and divergence from model outputs."""
        field_re, divergence_re = self.model(x, alphas[..., : self.model.DOF])
        field_im, divergence_im = self.model(x, alphas[..., self.model.DOF:])
        return field_re + 1j * field_im, divergence_re + 1j * divergence_im

    def PINN(
        self,
        alphas: torch.Tensor,
        sampletype: str = "all",
        batch_idx: torch.Tensor | slice = slice(None),
        FS_only: bool = False,
    ) -> torch.Tensor:
        """Evaluate the physics-informed residual for selected samples."""
        samples = self.model.samples[sampletype]
        if sampletype == "train":
            samples = samples[batch_idx]

        x = CNtoR2N(samples)
        field, divergence = self.get_F(x, alphas)
        sum_dims = tuple(range(1, field.ndim))
        field_grad_action = (field * self.grad_action(x)).sum(sum_dims)
        residual = divergence - field_grad_action
        if FS_only:
            return residual

        q = self.obs.Q(samples, samples.conj())
        q_expectation = 1 if self.obs.i == self.obs.j else 0
        return residual - q + q_expectation

    def get_obs(
        self,
        alphas: torch.Tensor,
        sampletype: str = "all",
        batch_idx: torch.Tensor | slice = slice(None),
    ) -> torch.Tensor:
        """Evaluate the PINN-improved observable."""
        samples = self.model.samples[sampletype]
        if sampletype == "train":
            samples = samples[batch_idx]
        x = CNtoR2N(samples)

        observable = self.obs.O(samples, samples.conj())
        grad_observable = self.grad_obs(x)
        field, _ = self.get_F(x, alphas)
        sum_dims = tuple(range(1, field.ndim))
        field_grad_observable = (field * grad_observable).sum(sum_dims)
        residual = self.PINN(
            alphas, sampletype=sampletype, batch_idx=batch_idx)
        return -(field_grad_observable + residual * observable)

    def forward(self, **kwargs: Any) -> torch.Tensor:
        """Return the mean squared PINN residual."""
        residual = self.PINN(**kwargs)
        return (residual * residual.conj()).real.mean()

    def exp(self, **kwargs: Any) -> torch.Tensor:
        """Return the improved observable expectation."""
        return self.get_obs(**kwargs).mean()

    def var(self, **kwargs: Any) -> torch.Tensor:
        """Return the improved observable variance."""
        return self.get_obs(**kwargs).var()
