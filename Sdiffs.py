from __future__ import annotations

import matplotlib.pyplot as plt
import torch

import actions
import observables
from CPN_analysis import load_samples, parse_args
from deformations import HomogenousDeformations
from losses import VarianceLoss

if __name__ == "__main__":
    args = parse_args()
    N = args.N

    samples = load_samples(args).to(torch.float32)
    N_conf = len(samples)

    # Action
    my_action = actions.CPnAction(args.beta, args.N, args.latsize)
    volume = torch.tensor(args.latsize).prod().item()

    print(
        "\n".join(
            [
                "Ensemble details:",
                f"  source : {args.source}",
                f"  size   : {args.latsize}",
                f"  beta   : {args.beta}",
                f"  N_conf : {N_conf}",
                f"  E/V    : {my_action.energy(samples).mean()/volume:.6f}",
            ]
        )
    )

    sites = torch.tensor(
        [[i, j] for i in range(args.latsize[0]) for j in range(args.latsize[1])]
    )
    S_undef = args.beta * my_action.local_energy(samples, sites).view(
        *samples.shape[:-1]
    ).mean(0)

    # Train / test split
    train_idx = torch.randperm(N_conf)[: int(args.split * N_conf)]
    mask = torch.zeros(N_conf, dtype=torch.bool)
    mask[train_idx] = True

    # Observables and model
    model = HomogenousDeformations(N)
    model.samples = {"all": samples, "train": samples[mask], "test": samples[~mask]}

    OPF = observables.CPnOnePtFn(
        N=N,
        latsize=args.latsize,
        i=args.i,
        j=args.j,
        source=(0, 0),
        action=my_action,
    )

    fname = f"data/training_params_OPF_i{
        args.i}_j{args.j}_Nepochs500.pt"
    OPF_alphas = torch.load(fname, weights_only=False).to(torch.float32)

    fname = f"data/training_params_OPF_masked_i{
        args.i}_j{args.j}_Nepochs500.pt"
    OPF_alphas_masked = torch.load(fname, weights_only=False).to(torch.float32)

    Z = OPF(model, deform=True, alphas=OPF_alphas, return_fields=True)
    S_def = args.beta * my_action.local_energy(Z, sites).view(*samples.shape[:-1]).mean(
        0
    )
    Z = OPF(model, deform=True, alphas=OPF_alphas_masked, return_fields=True)
    S_def_masked = args.beta * my_action.local_energy(Z, sites).view(
        *samples.shape[:-1]
    ).mean(0)

    S_diff = S_def - S_undef
    S_diff_masked = S_def_masked - S_undef

    TPF = observables.CPnTwoPtFn(
        N=N,
        latsize=args.latsize,
        i=args.i,
        j=args.j,
        delta=args.delta,
        source=(0, 0),
        action=my_action,
    )

    alpha_rolled = torch.roll(OPF_alphas, args.delta[0], dims=(0))
    alpha_rolled[..., [0, 1]] = alpha_rolled[..., [1, 0]]
    tpf_alphas = OPF_alphas + alpha_rolled
    tpf_alphas[..., 2] = (tpf_alphas[..., 0] + tpf_alphas[..., 1]) / 2

    Z = TPF(model, deform=True, alphas=tpf_alphas, return_fields=True)
    S_def2 = args.beta * my_action.local_energy(Z, sites).view(
        *samples.shape[:-1]
    ).mean(0)
    S_diff2 = S_def2 - S_undef

    alpha_rolled = torch.roll(OPF_alphas_masked, args.delta[0], dims=(0))
    alpha_rolled[..., [0, 1]] = alpha_rolled[..., [1, 0]]
    tpf_alphas_masked = OPF_alphas_masked + alpha_rolled

    fname = f"data/training_params_TPF_i{
        args.i}_j{args.j}_delta_{args.delta[0]}_{args.delta[1]}_Nepochs200.pt"
    TPF_alphas = torch.load(fname, weights_only=False).to(torch.float32)

    Z = TPF(model, deform=True, alphas=TPF_alphas, return_fields=True)
    S_def_tpf = args.beta * my_action.local_energy(Z, sites).view(
        *samples.shape[:-1]
    ).mean(0)
    S_diff_tpf = S_def_tpf - S_undef

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    data_list = [S_diff, S_diff2, S_diff_tpf]

    title_list1 = [
        r"Re($\Delta S_{\mathbf{x},\mathbf{x}+\hat{\mathbf{\mu}}}$) from OPF",
        r"Re($\Delta S$) with $\tilde{\alpha}_\mathbf{x}$",
        r"Re($\Delta S_{\mathbf{x},\mathbf{x}+\hat{\mathbf{\mu}}}$) from TPF",
    ]

    title_list2 = [
        r"Im($\Delta S_{\mathbf{x},\mathbf{x}+\hat{\mathbf{\mu}}}$) from OPF",
        r"Im($\Delta S$) with $\tilde{\alpha_\mathbf{x}}$",
        r"Im($\Delta S_{\mathbf{x},\mathbf{x}+\hat{\mathbf{\mu}}}$) from TPF",
    ]
    for col in range(3):
        v = torch.abs(data_list[col].real).max()
        im = axs[0, col].imshow(
            data_list[col].real.detach().numpy(), vmin=-v, vmax=v, cmap="RdBu_r"
        )
        fig.colorbar(im, ax=axs[0, col], shrink=0.6)

        v = torch.abs(data_list[col].imag).max()
        im = axs[1, col].imshow(
            data_list[col].imag.detach().numpy(), vmin=-v, vmax=v, cmap="RdBu_r"
        )
        fig.colorbar(im, ax=axs[1, col], shrink=0.6)
        axs[0, col].set_title(title_list1[col])
        axs[1, col].set_title(title_list2[col])

    for row in range(2):
        axs[row, 0].set_ylabel(r"$x_1$")

    fig2, axs2 = plt.subplots(3, 3, figsize=(9, 9), constrained_layout=True)
    title_list1 = [
        r"$\alpha_{\mathbf{x}}^0$ from OPF(0,0)",
        r"$\alpha_{\mathbf{x}}^1$ from OPF(0,0)",
        r"$\alpha_{\mathbf{x}}^2$ from OPF(0,0)",
    ]
    title_list2 = [
        r"$\tilde{\alpha}_\mathbf{x}^0=\alpha_\mathbf{"
        + r"x}^0+\alpha_{{\mathbf{x}+("
        + str(args.delta[0])
        + r",0)}}^1$",
        r"$\tilde{\alpha}_\mathbf{x}^1=\alpha_\mathbf{"
        + r"x}^1+\alpha_{{\mathbf{x}+("
        + str(args.delta[0])
        + ",0)}}^0$",
        r"$\tilde{\alpha}_\mathbf{x}^2=\frac{1}{2}(\tilde{"
        + r"\alpha_\mathbf{x}}^0+\tilde{\alpha_\mathbf{x}}^1)$",
    ]
    title_list3 = [
        r"$\alpha_{\mathbf{x}}^0$ from TPF",
        r"$\alpha_{\mathbf{x}}^1$ from TPF",
        r"$\alpha_{\mathbf{x}}^2$ from TPF",
    ]
    data_list = [OPF_alphas, tpf_alphas, TPF_alphas]
    for col in range(3):
        for row in range(3):
            v = torch.abs(data_list[row]).max()
            im = axs2[row, col].imshow(
                data_list[row][:, :, col].detach().numpy(),
                vmin=-v,
                vmax=v,
                cmap="RdBu_r",
            )
            fig2.colorbar(im, ax=axs2[row, col], shrink=0.7)
        axs2[0, col].set_title(title_list1[col])
        axs2[1, col].set_title(title_list2[col])
        axs2[2, col].set_title(title_list3[col])
        axs2[-1, col].set_xlabel(r"$x_2$")

    for row in range(2):
        axs2[row, 0].set_ylabel(r"$x_1$")

    fig3, axs3 = plt.subplots(3, 3, figsize=(9, 9), constrained_layout=True)
    title_list1 = [
        r"$\alpha_{\mathbf{x}}^0$ from OPF(0,0)",
        r"$\alpha_{\mathbf{x}}^1$ from OPF(0,0)",
        r"$\alpha_{\mathbf{x}}^2$ from OPF(0,0)",
    ]
    title_list2 = [
        r"$\tilde{\alpha}_\mathbf{x}^0=\alpha_\mathbf{"
        + r"x}^0+\alpha_{{\mathbf{x}+("
        + str(args.delta[0])
        + r",0)}}^1$",
        r"$\tilde{\alpha}_\mathbf{x}^1=\alpha_\mathbf{"
        + r"x}^1+\alpha_{{\mathbf{x}+("
        + str(args.delta[0])
        + ",0)}}^0$",
        r"$\tilde{\alpha}_\mathbf{x}^2==\alpha_\mathbf{"
        + r"x}^2+\alpha_{{\mathbf{x}+("
        + str(args.delta[0])
        + r",0)}}^2$",
    ]
    title_list3 = [
        r"$\alpha_{\mathbf{x}}^0$ from TPF",
        r"$\alpha_{\mathbf{x}}^1$ from TPF",
        r"$\alpha_{\mathbf{x}}^2$ from TPF",
    ]
    data_list = [OPF_alphas_masked, tpf_alphas_masked, TPF_alphas]
    for col in range(3):
        for row in range(3):
            v = torch.abs(data_list[row]).max()
            im = axs3[row, col].imshow(
                data_list[row][:, :, col].detach().numpy(),
                vmin=-v,
                vmax=v,
                cmap="RdBu_r",
            )
            fig3.colorbar(im, ax=axs3[row, col], shrink=0.7)
        axs3[0, col].set_title(title_list1[col])
        axs3[1, col].set_title(title_list2[col])
        axs3[2, col].set_title(title_list3[col])
        axs3[-1, col].set_xlabel(r"$x_2$")

    for row in range(2):
        axs3[row, 0].set_ylabel(r"$x_1$")

    print(f"Two-pt fn (TPF) between (0,0) and ({args.delta[0]},{args.delta[1]})")
    loss_fn = VarianceLoss(model, TPF)
    print(
        f"undeformed: <TPF>={loss_fn.target_exp:.6f}, var(TPF)={
          loss_fn.initial_var:.6f}"
    )
    obs = loss_fn.get_obs(deform=True, alphas=tpf_alphas, sampletype="all")
    tpf_exp, tpf_var = obs.mean(), obs.var()
    print(
        f"deformed with OPF params: <TPF>={
          tpf_exp:.6f}, var(TPF)={tpf_var:.6f}"
    )
    obs = loss_fn.get_obs(deform=True, alphas=tpf_alphas_masked, sampletype="all")
    tpf_exp, tpf_var = obs.mean(), obs.var()
    print(
        f"deformed with masked OPF params: <TPF>={
          tpf_exp:.6f}, var(TPF)={tpf_var:.6f}"
    )
    obs = loss_fn.get_obs(deform=True, alphas=TPF_alphas, sampletype="all")
    TPF_exp, TPF_var = obs.mean(), obs.var()
    print(
        f"deformed with TPF params: <TPF>={
          TPF_exp:.6f}, var(TPF)={TPF_var:.6f}"
    )

    plt.show()
