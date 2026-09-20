from __future__ import annotations

import argparse
import ast
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import XY_utils as xy
from actions import XYAction
from deformations import ThetaDefs
from losses import VarianceLoss
from networks import UNet
from observables import XYOnePtFn, XYTwoPtFn
from training import Trainer, init_weights, periodic_box_mask

SPLIT = 0.8  # training split
BEST_WINDOW = 20


def plot_scaling(
    betas: list[float],
    S_diffs: list[torch.Tensor],
    grad_f_sqs: list[float],
    alphas: list[torch.Tensor],
    lattices: list[str],
    Ls: list[int],
    mod_squares: list[float],
    separation: float,
) -> None:
    """Plot action fluctuations, observable scaling, and learned fields."""

    # Linv = [1/L for L in Ls]
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

    exp_S_diffs = [torch.exp(-diff).real for diff in S_diffs]
    S_ys = [exp.var().item() / (exp.mean().item() ** 2) for exp in exp_S_diffs]
    ax[0].scatter(betas, S_ys, marker="o")
    ax[0].set_ylabel(
        r"${\rm Var}(e^{-S_{\rm pen}})/\langle e^{-S_{\rm pen}} \rangle^2 $"
    )

    # ax[1].scatter(betas, grad_f_sqs, marker="o")
    # ax[1].set_ylabel(r"$-\int d^2x (\nabla f)^2$")

    ax[1].scatter(betas, mod_squares, marker="o")
    if separation > 0:
        scaling = [(1 / separation) ** (1 / (torch.pi * b)) for b in betas]
        ax[1].plot(betas, scaling, ls="--", c="r")
    ax[1].set_ylabel(r"$\langle|Q|^2\rangle$")

    ax[1].set_xlabel(r"$\beta$")
    ax[1].set_xticks(betas)
    ax[1].set_xticklabels([f"{b:.2g}" for b in betas])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    fig2, axes = plt.subplots(ncols=len(betas), nrows=1, figsize=(12, 3))
    ax2 = np.atleast_1d(axes)
    for b_idx, beta in enumerate(betas):
        data = alphas[b_idx]
        vmax = torch.max(torch.abs(data))
        im = ax2[b_idx].imshow(
            data,
            vmin=-vmax,
            vmax=vmax,
        )
        fig2.colorbar(im, ax=ax2[b_idx])
        ax2[b_idx].set_title(f"beta={beta:3g}, {lattices[b_idx]}")

    fig2.tight_layout()

    plt.show()


def main(args: argparse.Namespace) -> None:
    """Train two-dimensional XY observable deformations for input ensembles."""

    source = (0, 0)
    sink = tuple(map(sum, zip(source, args.delta)))
    separation = ((args.delta[0]) ** 2 + (args.delta[1]) ** 2) ** 0.5

    my_obs = XYOnePtFn if args.obs == "OPF" else XYTwoPtFn
    obs_str = f"OPF {source}" if args.obs == "OPF" else f"TPF ({source},{sink})"

    betas = []
    Ls = []
    lattices = []
    S_diffs = []
    alphas = []
    grad_f_sqs = []
    mod_squares = []
    for filename in sorted(args.files):
        fields, beta = xy.load_ensemble(filename)
        betas.append(beta)

        # if beta >= 0.9:
        #    LEARNING_RATE = 1e-5
        # else:
        latsize = fields.shape[-2:]
        Ls.append(latsize[0])
        action = XYAction(beta, ndims=2)
        N_cfgs = fields.shape[0]
        lattices.append(f"{latsize[0]}x{latsize[1]}")

        train_idx = torch.randperm(N_cfgs)[: int(SPLIT * N_cfgs)]
        mask = torch.zeros(N_cfgs, dtype=torch.bool)
        mask[train_idx] = True

        model = ThetaDefs()
        model.samples = {"all": fields, "train": fields[mask], "test": fields[~mask]}

        obs_kwargs = dict(latsize=latsize, source=(0, 0), action=action)
        if args.obs == "TPF":
            obs_kwargs["sink"] = sink
        obs = my_obs(**obs_kwargs)

        loss_fn = VarianceLoss(model, obs)
        alphanet = UNet(
            input_dim=obs.grid.shape[-1], output_dim=1, hd=args.unet_hidden_dim
        )
        alphanet.apply(init_weights)

        nn.init.zeros_(alphanet.output.weight)
        nn.init.zeros_(alphanet.output.bias)

        mean, err = loss_fn.target_exp, loss_fn.exp_err
        StN_undef = mean / err
        print("===undeformed " + obs_str + "================")
        print(f"exp = {mean:3g}," f"std={err:3g}," f"\nStN = {StN_undef:3g}")

        optimizer = optim.Adam(alphanet.parameters(), lr=args.learning_rate)
        batch_size = min(args.batch_size, len(model.samples["train"]))
        trainer = Trainer(model, loss_fn, optimizer, args.epochs, batch_size)
        alpha_mask = (
            periodic_box_mask(latsize, source, radius=5).unsqueeze(-1)
            if args.obs == "OPF"
            else None
        )
        trainer.train(
            alphanet,
            mask=alpha_mask,
            plot=args.plot,
            max_training_time=args.max_training_time,
            plateau_window=args.plateau_window,
            plateau_patience=args.plateau_patience,
            plateau_min_delta=args.plateau_min_delta,
            extension=args.training_extension,
        )

        best_alphas = trainer.best_alphas(BEST_WINDOW)
        # best_alphas = obs.periodic_field(plot=False).unsqueeze(-1)
        alphas.append(best_alphas)

        mean, err = loss_fn.get_stats(alphas=best_alphas, sampletype="all")
        StN_def = mean / err
        gain = (loss_fn.exp_err / err) ** 2
        print("=====deformed " + obs_str + "================")
        print(
            f"exp = {mean:3g}, std={err:3g},"
            f"\nStN = {StN_def:3g}"
            f"\nVar: undef/def = {gain:3g}"
        )

        S_diff = obs(model, deform=True, alphas=best_alphas, get_Sdiff=True)
        S_diffs.append(S_diff)
        mod_squares.append(loss_fn.mod_squared(alphas=best_alphas, sampletype="all"))

    plot_scaling(
        betas, S_diffs, grad_f_sqs, alphas, lattices, Ls, mod_squares, separation
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="+",
        help="Ensemble files, e.g. data/xy_b*_L64_ens.npy",
    )

    parser.add_argument(
        "obs",
        choices=("TPF", "OPF"),
        help="TPF or OPF",
    )
    parser.add_argument("--max-training-time", type=float, default=None)
    parser.add_argument("--plateau-window", type=int, default=20)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-min-delta", type=float, default=1e-4)
    parser.add_argument("--training-extension", type=int, default=100)
    parser.add_argument(
        "--delta",
        help='two point function delta, eg "(0,1)"',
        type=lambda s: tuple(ast.literal_eval(s)),
        default=(0, 6),
    )
    parser.add_argument("--epochs", help="epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--unet-hidden-dim", type=int, default=None)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    main(args)
