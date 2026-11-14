"""Objective-specific CP(N) OPF training workflows."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from CPN_utils import ComplexPair, complex_inner
from losses import PenaltyLoss, VarianceLoss
from networks import UNet
from observables import CPnOnePtFn
from training import WEIGHT_INIT_STD, init_weights

from .config import BEST_WINDOW
from .model import CPnHomogeneousDeformations
from .optimization import (
    optimize_field_scale,
    solve_approx2_penalty_field,
    solve_approx3_penalty_field,
)
from .training import (
    make_model,
    make_split_mask,
    penalty_source_constraints,
    run_training,
    save_training_result,
)


class OffsetFieldNetwork(nn.Module):
    """Add a fixed reference field to a trainable network residual."""

    def __init__(self, network: nn.Module, reference_field: torch.Tensor) -> None:
        super().__init__()
        self.network = network
        self.register_buffer("reference_field", reference_field.detach().clone())

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.reference_field + self.network(grid)


def init_variance_network(alphanet: UNet) -> None:
    """Initialize a CP(N) variance network off the stationary zero contour.

    A diagonal CP(N) OPF has vanishing variance gradient at an exactly zero
    homogeneous deformation.  In particular, zeroing the final U-Net layer
    makes Adam unable to leave that point.  Keep the standard small-weight
    initialization and seed the output bias as well, so the initial masked
    deformation is of order ``WEIGHT_INIT_STD`` rather than identically zero.
    """
    alphanet.apply(init_weights)
    with torch.no_grad():
        nn.init.normal_(
            alphanet.output.weight,
            mean=0.0,
            std=WEIGHT_INIT_STD,
        )
        nn.init.normal_(
            alphanet.output.bias,
            mean=0.0,
            std=WEIGHT_INIT_STD,
        )


def _local_gauge_variance_evaluator(
    fields: torch.Tensor,
    links: torch.Tensor,
    profile: torch.Tensor,
    model: CPnHomogeneousDeformations,
    *,
    beta: float,
    i: int,
    j: int,
    source: tuple[int, ...],
) -> Callable[[float], float]:
    """Build an exact variance evaluator using only profile-touching links.

    A localized profile is exactly zero outside its periodic support. Every
    gauge-action term with both endpoints outside that support cancels between
    the deformed and undeformed actions. Evaluating the remaining links
    directly avoids constructing full-lattice action arrays at every trial
    scale while preserving the complete loaded ensemble.
    """
    if fields.ndim < 3 or fields.shape[0] != links.shape[0]:
        raise ValueError("gauge fields and links have incompatible shapes")
    shape = tuple(int(length) for length in fields.shape[1:-1])
    if tuple(links.shape[1:-1]) != shape or links.shape[-1] != len(shape):
        raise ValueError("gauge links do not match the field lattice")
    if len(source) != len(shape):
        raise ValueError("source and lattice dimensions differ")

    nconfig = int(fields.shape[0])
    volume = int(np.prod(shape))
    n_components = int(fields.shape[-1])
    n_channels = n_components // 2
    if n_components != 2 * n_channels or profile.shape != shape + (n_channels,):
        raise ValueError("Approximation 2 profile has an incompatible shape")

    profile_array = profile.detach().cpu().numpy().reshape(volume, n_channels)
    support_flat = np.flatnonzero(np.any(profile_array != 0.0, axis=-1))
    if support_flat.size == 0:
        raise ValueError("Approximation 2 profile has empty support")
    support_position = np.full(volume, -1, dtype=np.int64)
    support_position[support_flat] = np.arange(support_flat.size)
    source_flat = int(np.ravel_multi_index(source, shape))
    source_position = int(support_position[source_flat])
    if source_position < 0:
        raise ValueError("Approximation 2 profile vanishes at its source")

    support_coordinates = np.column_stack(
        np.unravel_index(support_flat, shape)
    )
    active_links = []
    for axis in range(len(shape)):
        preceding = support_coordinates.copy()
        preceding[:, axis] = (preceding[:, axis] - 1) % shape[axis]
        origins = np.unique(
            np.concatenate((support_coordinates, preceding), axis=0), axis=0
        )
        forward = origins.copy()
        forward[:, axis] = (forward[:, axis] + 1) % shape[axis]
        origins_flat = np.ravel_multi_index(origins.T, shape)
        forward_flat = np.ravel_multi_index(forward.T, shape)
        active_links.append(
            (
                torch.as_tensor(origins_flat.copy(), dtype=torch.long,
                                device=fields.device),
                torch.as_tensor(forward_flat.copy(), dtype=torch.long,
                                device=fields.device),
                support_position[origins_flat] >= 0,
                support_position[forward_flat] >= 0,
                torch.as_tensor(
                    support_position[origins_flat].clip(min=0),
                    dtype=torch.long,
                    device=fields.device,
                ),
                torch.as_tensor(
                    support_position[forward_flat].clip(min=0),
                    dtype=torch.long,
                    device=fields.device,
                ),
            )
        )

    flat_fields = fields.reshape(nconfig, volume, n_components)
    flat_links = links.reshape(nconfig, volume, len(shape))
    support_index = torch.as_tensor(
        support_flat.copy(), dtype=torch.long, device=fields.device
    )
    support_profile = torch.as_tensor(
        profile_array[support_flat], dtype=fields.dtype, device=fields.device
    )
    source_profile = torch.as_tensor(
        profile_array[source_flat], dtype=fields.dtype, device=fields.device
    )
    # Bound temporary arrays independently of lattice dimension.
    chunk_size = max(1, min(256, (2**18) // max(volume, 1)))
    omega_t = model.omega.T.to(dtype=fields.dtype, device=fields.device)

    def source_opf(values: torch.Tensor, alpha: float) -> ComplexPair:
        deformed = model.deform(values, float(alpha) * source_profile)
        real = deformed.real
        imag = deformed.imag
        z_i = ComplexPair(
            real[..., 2 * i] - imag[..., 2 * i + 1],
            imag[..., 2 * i] + real[..., 2 * i + 1],
        )
        z_j_bar = ComplexPair(
            real[..., 2 * j] + imag[..., 2 * j + 1],
            imag[..., 2 * j] - real[..., 2 * j + 1],
        )
        return z_i * z_j_bar

    def evaluate(alpha: float) -> float:
        with torch.no_grad():
            q_chunks = [
                source_opf(
                    flat_fields[start: start + chunk_size, source_flat, :],
                    alpha,
                )
                for start in range(0, nconfig, chunk_size)
            ]
            q_real = torch.cat([value.real for value in q_chunks])
            q_imag = torch.cat([value.imag for value in q_chunks])
            q_mean_real = q_real.mean()
            q_mean_imag = q_imag.mean()

            observable_chunks = []
            for start in range(0, nconfig, chunk_size):
                stop = min(start + chunk_size, nconfig)
                x_support = flat_fields[start:stop].index_select(
                    1, support_index
                )
                alpha_support = float(alpha) * support_profile
                deformed_support = model.deform(x_support, alpha_support)
                jacobian = model.detJac(
                    x_support, alpha_support
                ).prod(dim=-1)

                delta_real = torch.zeros(
                    stop - start, dtype=fields.dtype, device=fields.device
                )
                delta_imag = torch.zeros_like(delta_real)
                for axis, link_data in enumerate(active_links):
                    (
                        origins,
                        forward,
                        origin_is_supported,
                        forward_is_supported,
                        origin_support_index,
                        forward_support_index,
                    ) = link_data
                    x_origin = flat_fields[start:stop].index_select(1, origins)
                    x_forward = flat_fields[start:stop].index_select(1, forward)

                    z_origin_real = x_origin.clone()
                    z_origin_imag = torch.zeros_like(x_origin)
                    if np.any(origin_is_supported):
                        selection = torch.as_tensor(
                            origin_is_supported, dtype=torch.bool,
                            device=fields.device,
                        )
                        selected = origin_support_index[selection]
                        z_origin_real[:, selection, :] = deformed_support.real.index_select(
                            1, selected
                        )
                        z_origin_imag[:, selection, :] = deformed_support.imag.index_select(
                            1, selected
                        )

                    z_forward_real = x_forward.clone()
                    z_forward_imag = torch.zeros_like(x_forward)
                    if np.any(forward_is_supported):
                        selection = torch.as_tensor(
                            forward_is_supported, dtype=torch.bool,
                            device=fields.device,
                        )
                        selected = forward_support_index[selection]
                        z_forward_real[:, selection, :] = deformed_support.real.index_select(
                            1, selected
                        )
                        z_forward_imag[:, selection, :] = deformed_support.imag.index_select(
                            1, selected
                        )

                    z_origin = ComplexPair(z_origin_real, z_origin_imag)
                    z_forward = ComplexPair(z_forward_real, z_forward_imag)
                    link_angles = flat_links[start:stop].index_select(
                        1, origins
                    )[..., axis]
                    cos_link = torch.cos(link_angles)
                    sin_link = torch.sin(link_angles)

                    a_deformed = complex_inner(z_forward, z_origin)
                    b_deformed = complex_inner(
                        z_forward, z_origin.matmul(omega_t)
                    )
                    term_deformed = -2.0 * (
                        a_deformed * cos_link + b_deformed * sin_link
                    )

                    a_undeformed = (x_forward * x_origin).sum(dim=-1)
                    b_undeformed = (
                        x_forward * (x_origin @ omega_t)
                    ).sum(dim=-1)
                    term_undeformed = -2.0 * (
                        a_undeformed * cos_link + b_undeformed * sin_link
                    )
                    delta_real += float(beta) * (
                        term_deformed.real - term_undeformed
                    ).sum(dim=-1)
                    delta_imag += float(beta) * term_deformed.imag.sum(dim=-1)

                action_weight_magnitude = torch.exp(-delta_real)
                action_weight = ComplexPair(
                    action_weight_magnitude * torch.cos(delta_imag),
                    -action_weight_magnitude * torch.sin(delta_imag),
                )
                centered_opf = ComplexPair(
                    q_real[start:stop] - q_mean_real,
                    q_imag[start:stop] - q_mean_imag,
                )
                deformed_opf = centered_opf * jacobian * action_weight
                observable_chunks.append(deformed_opf.real)

            return float(torch.cat(observable_chunks).var().item())

    return evaluate


def train_variance_loss(
    fields: torch.Tensor,
    links: torch.Tensor | None,
    *,
    n: int,
    action_type: str,
    ndim: int,
    beta: float,
    obs: CPnOnePtFn,
    args: argparse.Namespace,
    source: tuple[int, ...],
    i: int,
    j: int,
    training_savename: str | Path,
    device: torch.device,
    split_mask: torch.Tensor | None = None,
    initial_field: np.ndarray | torch.Tensor | None = None,
) -> float:
    """Train a homogeneous CP(N) deformation against OPF variance."""
    split_mask = make_split_mask(fields, split_mask)
    latsize = tuple(int(length) for length in fields.shape[1:-1])
    model = make_model(
        fields,
        links,
        n=n,
        action_type=action_type,
        split_mask=split_mask,
        device=device,
    )
    loss_fn = VarianceLoss(model, obs)
    alphanet = UNet(
        input_dim=obs.grid.shape[-1],
        output_dim=model.DOF,
        ndim=ndim,
        hd=args.unet_hidden_dim,
    ).to(device)
    init_variance_network(alphanet)
    reference_field = None
    reference_ratio = None
    if initial_field is not None:
        reference_field = torch.as_tensor(
            initial_field, dtype=fields.dtype, device=device
        )
        expected_shape = latsize + (model.DOF,)
        if tuple(reference_field.shape) != expected_shape:
            raise ValueError(
                "initial variance field has shape "
                f"{tuple(reference_field.shape)}; expected {expected_shape}"
            )
        if not torch.isfinite(reference_field).all():
            raise ValueError("initial variance field contains non-finite values")
        alphanet = OffsetFieldNetwork(alphanet, reference_field)

    mean, err = loss_fn.target_exp, loss_fn.exp_err
    print(f"---undeformed CP(N) OPF ({i},{j})----------------")
    print(f"exp = {mean:3g}, std={err:3g}, StN = {mean / err:3g}")

    trainer = run_training(model, loss_fn, alphanet, latsize, source, args)
    trained_field = trainer.best_alphas(BEST_WINDOW)
    if loss_fn.initial_var <= 0:
        raise ValueError("undeformed CP(N) OPF variance must be positive")
    variance_ratio = float(
        (loss_fn(alphas=trained_field, sampletype="all") / loss_fn.initial_var)
        .detach()
        .cpu()
        .item()
    )
    trained_candidate_ratio = variance_ratio
    best_alphas = trained_field
    alpha_scale = 1.0
    selected_field = "trained"
    if reference_field is not None:
        reference_ratio = float(
            (loss_fn(alphas=reference_field, sampletype="all")
             / loss_fn.initial_var)
            .detach()
            .cpu()
            .item()
        )
        variance_evaluator = None
        if action_type == "gauge" and links is not None:
            variance_evaluator = _local_gauge_variance_evaluator(
                fields,
                links,
                trained_field,
                model,
                beta=beta,
                i=i,
                j=j,
                source=source,
            )
        alpha_scale, optimized_variance = optimize_field_scale(
            trained_field,
            loss_fn,
            bounds=tuple(args.alpha_bounds),
            grid_points=args.alpha_grid_points,
            xatol=args.alpha_xatol,
            variance_evaluator=variance_evaluator,
        )
        trained_candidate_ratio = float(
            optimized_variance / loss_fn.initial_var
        )
        if trained_candidate_ratio < reference_ratio:
            variance_ratio = trained_candidate_ratio
        else:
            # Retain a verified variance-reducing reference when residual
            # training and scale refinement do not improve on Approximation 2.
            best_alphas = reference_field
            alpha_scale = 1.0
            variance_ratio = reference_ratio
            selected_field = "approx2_reference"

    extra_metadata = None
    if reference_ratio is not None:
        extra_metadata = {
            "variance_initialization": "scaled Approximation 2 plus U-Net residual",
            "approx2_reference_ratio": reference_ratio,
            "trained_candidate_ratio": trained_candidate_ratio,
            "selected_field": selected_field,
        }
    mean, err = loss_fn.get_stats(
        alphas=alpha_scale * best_alphas, sampletype="all"
    )
    print(f"-----deformed CP(N) OPF ({i},{j})-----------------")
    print(
        f"exp = {mean:3g}, std={err:3g}, StN = {mean / err:3g}\n"
        f"Var: def/undef = {variance_ratio:3g}"
        + (
            f" (trained candidate {trained_candidate_ratio:3g}; "
            f"Approx. 2 reference {reference_ratio:3g})"
            if reference_ratio is not None else ""
        )
    )
    save_training_result(
        training_savename,
        trainer,
        n=n,
        action_type=action_type,
        ndim=ndim,
        beta=beta,
        latsize=latsize,
        source=source,
        i=i,
        j=j,
        learned_field=best_alphas,
        alpha_scale=alpha_scale,
        variance_ratio=variance_ratio,
        extra_metadata=extra_metadata,
        loss_function="variance",
    )
    return variance_ratio


def train_penalty_loss(
    fields: torch.Tensor,
    links: torch.Tensor | None,
    *,
    n: int,
    action_type: str,
    ndim: int,
    beta: float,
    obs: CPnOnePtFn,
    args: argparse.Namespace,
    source: tuple[int, ...],
    i: int,
    j: int,
    training_savename: str | Path,
    device: torch.device,
    split_mask: torch.Tensor | None = None,
    initial_field: np.ndarray | torch.Tensor | None = None,
) -> float:
    """Learn a normalized CP(N) penalty field, then optimize its scale.

    The deformation has one component per CP(N) vector component.  We fix
    the otherwise stationary zero-field direction with a unit-norm source
    vector: off-diagonal OPFs use equal and opposite amplitudes in their
    ``i`` and ``j`` channels, while diagonal OPFs retain ``f_i(source)=1``.
    A single real scalar subsequently scales the complete learned vector
    field to minimize OPF variance.
    """
    split_mask = make_split_mask(fields, split_mask)
    latsize = tuple(int(length) for length in fields.shape[1:-1])
    model = make_model(
        fields,
        links,
        n=n,
        action_type=action_type,
        split_mask=split_mask,
        device=device,
    )
    penalty_loss = PenaltyLoss(model, obs.action, obs=obs)
    alphanet = UNet(
        input_dim=obs.grid.shape[-1],
        output_dim=model.DOF,
        ndim=ndim,
        hd=args.unet_hidden_dim,
    ).to(device)
    alphanet.apply(init_weights)
    # Leave the unconstrained field close to zero: the penalty exponent is
    # extensive, while the source constraint supplies a non-trivial field.
    with torch.no_grad():
        alphanet.output.weight.mul_(1e-2)
        nn.init.zeros_(alphanet.output.bias)
    extra_metadata = None
    if initial_field is not None:
        reference_field = torch.as_tensor(
            initial_field, dtype=fields.dtype, device=device
        )
        expected_shape = latsize + (model.DOF,)
        if tuple(reference_field.shape) != expected_shape:
            raise ValueError(
                "initial penalty field has shape "
                f"{tuple(reference_field.shape)}; expected {expected_shape}"
            )
        if not torch.isfinite(reference_field).all():
            raise ValueError("initial penalty field contains non-finite values")
        constraints = penalty_source_constraints(
            source, model.DOF, i, j
        )
        for index, value in constraints.items():
            if not np.isclose(
                float(reference_field[index].item()), value, atol=1e-5
            ):
                raise ValueError(
                    "the Approximation 2 profile does not satisfy the "
                    "penalty source constraints"
                )
        alphanet = OffsetFieldNetwork(alphanet, reference_field)
        extra_metadata = {
            "profile_initialization": (
                "unscaled Approximation 2 profile plus U-Net residual"
            )
        }

    trainer = run_training(
        model,
        penalty_loss,
        alphanet,
        latsize,
        source,
        args,
        fixed_alphas=penalty_source_constraints(
            source, model.DOF, i, j
        ),
        learning_rate=args.learning_rate,
    )
    f_x = trainer.best_alphas(BEST_WINDOW)

    variance_loss = VarianceLoss(model, obs)
    variance_evaluator = None
    if action_type == "gauge" and links is not None:
        variance_evaluator = _local_gauge_variance_evaluator(
            fields,
            links,
            f_x,
            model,
            beta=beta,
            i=i,
            j=j,
            source=source,
        )
    alpha, optimized_variance = optimize_field_scale(
        f_x,
        variance_loss,
        bounds=tuple(args.alpha_bounds),
        grid_points=args.alpha_grid_points,
        xatol=args.alpha_xatol,
        variance_evaluator=variance_evaluator,
    )
    scaled_field = alpha * f_x
    if variance_loss.initial_var <= 0:
        raise ValueError("undeformed CP(N) OPF variance must be positive")
    variance_ratio = optimized_variance / variance_loss.initial_var
    penalty_value = penalty_loss(alphas=f_x, sampletype="all").item()
    mean, err = variance_loss.get_stats(
        alphas=scaled_field, sampletype="all"
    )
    print(f"-----penalty-trained CP(N) OPF ({i},{j})---------")
    print(
        f"<exp(-2 Re S_diff)> = {penalty_value:3g}"
        f"\nalpha = {alpha:3g}"
        f"\nexp = {mean:3g}, std={err:3g}, StN = {mean / err:3g}"
        f"\nVar: def/undef = {variance_ratio:3g}"
    )
    save_training_result(
        training_savename,
        trainer,
        n=n,
        action_type=action_type,
        ndim=ndim,
        beta=beta,
        latsize=latsize,
        source=source,
        i=i,
        j=j,
        learned_field=f_x,
        alpha_scale=alpha,
        variance_ratio=variance_ratio,
        extra_metadata=extra_metadata,
        loss_function="penalty",
    )
    return float(variance_ratio)


def solve_approx2_loss(
    fields: torch.Tensor,
    links: torch.Tensor | None,
    *,
    n: int,
    action_type: str,
    ndim: int,
    beta: float,
    obs: CPnOnePtFn,
    args: argparse.Namespace,
    source: tuple[int, ...],
    i: int,
    j: int,
    training_savename: str | Path,
    device: torch.device,
    split_mask: torch.Tensor | None = None,
) -> float:
    """Solve Approximation 2 and optimize its scale on the supplied ensemble."""
    del split_mask, ndim
    if action_type != "gauge":
        raise ValueError("CP(N) Approximation 2 is defined for the gauge action")
    if links is None:
        raise ValueError("CP(N) Approximation 2 requires gauge links")

    latsize = tuple(int(length) for length in fields.shape[1:-1])
    field, diagnostics = solve_approx2_penalty_field(
        latsize,
        source,
        beta,
        cp_n=n,
        i=i,
        j=j,
        mask_radius=args.mask_radius,
        maxiter=args.approx2_maxiter,
        gtol=args.approx2_gtol,
        ftol=args.approx2_ftol,
    )
    model = CPnHomogeneousDeformations(
        n, action_type=action_type, device=device
    ).to(device)
    model.samples = {"all": fields}
    model.set_gauge_samples({"all": links})
    variance_loss = VarianceLoss(model, obs)
    if variance_loss.initial_var <= 0:
        raise ValueError("undeformed CP(N) OPF variance must be positive")

    profile = torch.as_tensor(field, dtype=fields.dtype, device=device)
    scale_variance = _local_gauge_variance_evaluator(
        fields,
        links,
        profile,
        model,
        beta=beta,
        i=i,
        j=j,
        source=source,
    )
    alpha, optimized_variance = optimize_field_scale(
        profile,
        variance_loss,
        bounds=tuple(args.alpha_bounds),
        grid_points=args.alpha_grid_points,
        xatol=args.alpha_xatol,
        variance_evaluator=scale_variance,
    )
    scaled_field = alpha * profile
    variance_ratio = float(optimized_variance / variance_loss.initial_var)
    result_path = Path(training_savename)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    field_array = profile.detach().cpu().numpy()
    scaled_array = scaled_field.detach().cpu().numpy()
    metadata = {
        "loss_function": "approx2",
        "objective_name": "small-fluctuation Approximation 2",
        "approximation": 2,
        "cp_n": int(n),
        "action_type": action_type,
        "ndim": int(len(latsize)),
        "beta": float(beta),
        "lattice_size": latsize,
        "source": tuple(int(coordinate) for coordinate in source),
        "i": int(i),
        "j": int(j),
        "field_at_source": float(field_array[*source, i]),
        "field_norm_at_source": float(np.linalg.norm(field_array[*source])),
        "mask_radius": int(args.mask_radius),
        "optimizer": "L-BFGS-B",
        "scale_optimizer": "grid plus bounded scalar minimization",
        "objective": "CP(N) small-fluctuation saddle penalty",
        "alpha_bounds": tuple(float(value) for value in args.alpha_bounds),
    }
    np.save(
        result_path,
        {
            "metadata": metadata,
            "loss_function": "approx2",
            "learned_field": field_array,
            "alpha_scale": float(alpha),
            "scaled_field": scaled_array,
            "variance_ratio": variance_ratio,
            "optimization_diagnostics": diagnostics,
        },
        allow_pickle=True,
    )
    print(
        f"-----CP(N) Approximation 2 ({i},{j})-----------------\n"
        f"penalty exponent = {diagnostics['penalty_exponent']:.8g}\n"
        f"alpha = {alpha:.8g}\n"
        f"Var: def/undef = {variance_ratio:.8g}"
    )
    return variance_ratio


def solve_approx3_loss(
    fields: torch.Tensor,
    links: torch.Tensor | None,
    *,
    n: int,
    action_type: str,
    ndim: int,
    beta: float,
    obs: CPnOnePtFn,
    args: argparse.Namespace,
    source: tuple[int, ...],
    i: int,
    j: int,
    training_savename: str | Path,
    device: torch.device,
    split_mask: torch.Tensor | None = None,
    approx3_signature: dict[str, object] | None = None,
) -> float:
    """Solve Approximation 3, optimize its scale, and save the result."""
    del split_mask
    if action_type != "gauge":
        raise ValueError("CP(N) Approximation 3 is defined for the gauge action")
    if links is None:
        raise ValueError("CP(N) Approximation 3 requires gauge links")

    latsize = tuple(int(length) for length in fields.shape[1:-1])
    if len(latsize) != ndim:
        raise ValueError("the CP(N) field shape does not match the dimension")
    approx2_profile, approx2_diagnostics = solve_approx2_penalty_field(
        latsize,
        source,
        beta,
        cp_n=n,
        i=i,
        j=j,
        mask_radius=args.mask_radius,
        maxiter=args.approx2_maxiter,
        gtol=args.approx2_gtol,
        ftol=args.approx2_ftol,
    )
    field, diagnostics = solve_approx3_penalty_field(
        latsize,
        source,
        beta,
        cp_n=n,
        i=i,
        j=j,
        mask_radius=args.mask_radius,
        initial_field=approx2_profile,
    )

    model = CPnHomogeneousDeformations(
        n, action_type=action_type, device=device
    ).to(device)
    model.samples = {"all": fields}
    model.set_gauge_samples({"all": links})
    variance_loss = VarianceLoss(model, obs)
    if variance_loss.initial_var <= 0:
        raise ValueError("undeformed CP(N) OPF variance must be positive")

    profile = torch.as_tensor(field, dtype=fields.dtype, device=device)
    scale_variance = _local_gauge_variance_evaluator(
        fields,
        links,
        profile,
        model,
        beta=beta,
        i=i,
        j=j,
        source=source,
    )
    alpha, optimized_variance = optimize_field_scale(
        profile,
        variance_loss,
        bounds=tuple(args.alpha_bounds),
        grid_points=args.alpha_grid_points,
        xatol=args.alpha_xatol,
        variance_evaluator=scale_variance,
    )
    scaled_field = alpha * profile
    variance_ratio = float(optimized_variance / variance_loss.initial_var)
    field_array = profile.detach().cpu().numpy()
    scaled_array = scaled_field.detach().cpu().numpy()
    metadata = {
        "loss_function": "approx3",
        "objective_name": "smooth-deformation Approximation 3",
        "approximation": 3,
        "cp_n": int(n),
        "action_type": action_type,
        "ndim": int(ndim),
        "beta": float(beta),
        "lattice_size": latsize,
        "source": tuple(int(coordinate) for coordinate in source),
        "i": int(i),
        "j": int(j),
        "field_at_source": float(field_array[*source, i]),
        "field_norm_at_source": float(np.linalg.norm(field_array[*source])),
        "mask_radius": int(args.mask_radius),
        "optimizer": "L-BFGS-B",
        "scale_optimizer": "grid plus bounded scalar minimization",
        "objective": "CP(N) smooth-deformation metric penalty",
        "alpha_bounds": tuple(float(value) for value in args.alpha_bounds),
        "ensemble_configurations": int(len(fields)),
        "approx3_signature": approx3_signature,
    }
    result_path = Path(training_savename)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        result_path,
        {
            "metadata": metadata,
            "loss_function": "approx3",
            "learned_field": field_array,
            "alpha_scale": float(alpha),
            "scaled_field": scaled_array,
            "variance_ratio": variance_ratio,
            "optimization_diagnostics": diagnostics,
            "approx2_initialization_diagnostics": approx2_diagnostics,
        },
        allow_pickle=True,
    )
    print(
        f"-----CP(N) Approximation 3 ({i},{j})-----------------\n"
        f"penalty exponent = {diagnostics['penalty_exponent']:.8g}\n"
        f"alpha = {alpha:.8g}\n"
        f"Var: def/undef = {variance_ratio:.8g}; "
        f"saved {result_path}"
    )
    return variance_ratio
