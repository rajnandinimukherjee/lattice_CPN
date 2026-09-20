from __future__ import annotations

import matplotlib.pyplot as plt
import torch

import actions
import observables
from CPN_analysis import load_samples, parse_args
from deformations import HomogenousDeformations
from losses import VarianceLoss
from training import bootstrap

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
    loss_fn_OPF = VarianceLoss(model, OPF)
    obs = loss_fn_OPF.get_obs(deform=False, sampletype="all")
    opf_exp, opf_var = obs.mean().item(), obs.var().item()
    opf_exp_err = bootstrap(obs).var().item() ** 0.5

    fname = "data/training_data_OPF_i0_j1_Nepochs280.pt"
    f2name = "data/training_data_OPF_masked_i0_j1_Nepochs280.pt"
    data = torch.load(fname, weights_only=False)
    data2 = torch.load(f2name, weights_only=False)

    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw={"hspace": 0},
        figsize=(5, 5),
        constrained_layout=True,
    )
    ax[0].set_ylabel("exp")
    ax[1].set_ylabel("var")
    ax[1].set_xlabel("epoch")
    fig.suptitle(r"$z_0z_1^*(0,0)$, one-pt-fn deformations")

    xdata = list(range(50, 280))
    mid, err = data["train_exp"][xdata], data["train_exp_err"][xdata]
    ax[0].fill_between(
        xdata, (mid - err).numpy(), (mid + err).numpy(), color="r", alpha=0.4
    )
    mid, err = data2["train_exp"][xdata], data2["train_exp_err"][xdata]
    ax[0].fill_between(
        xdata, (mid - err).numpy(), (mid + err).numpy(), color="b", alpha=0.4
    )
    # mid, err = data["test_exp"][xdata], data["test_exp_err"][xdata]
    # ax[0].fill_between(xdata,
    #                   (mid-err).numpy(),
    #                   (mid+err).numpy(),
    #                   color="b", alpha=0.4
    #                   )
    ax[0].axhspan(opf_exp + opf_exp_err, opf_exp - opf_exp_err, color="k", alpha=0.1)
    ax[0].axhline(opf_exp, color="k")
    ax[0].plot(xdata, data["train_exp"][xdata], color="r")
    ax[0].plot(xdata, data2["train_exp"][xdata], color="b")
    # ax[0].plot(xdata, data["test_exp"][xdata], color="b")

    ax[1].axhline(opf_var, color="k", label="undeformed")
    ax[1].plot(xdata, data["train_var"][xdata], color="r", label="deformed")
    ax[1].plot(xdata, data2["train_var"][xdata], color="b", label="deformed with BCs")
    # ax[1].plot(xdata, data["test_var"][xdata], color="b")
    ax[1].legend()

    plt.show()
