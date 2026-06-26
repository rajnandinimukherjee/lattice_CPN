import pdb

import matplotlib.pyplot as plt
import torch
from plot_settings import plotparams
plt.rcParams.update(plotparams)


def periodic_box_mask(lattice, center, radius):
    Nx, Ny = lattice
    cx, cy = center
    x = torch.arange(Nx)
    y = torch.arange(Ny)
    dx = torch.minimum((x - cx) % Nx, (cx - x) % Nx)
    dy = torch.minimum((y - cy) % Ny, (cy - y) % Ny)
    return (dx[:, None] <= radius) & (dy[None, :] <= radius)


def st_dev(data, mean=None, **kwargs):
    """standard deviation function - finds stdev
    around data mean or mean provided as input"""

    n = len(data)
    if mean is None:
        mean = torch.mean(data, axis=0)
    return (torch.sum((data - mean)**2, axis=-1) / n) ** 0.5


def bootstrap(data, Nboot=100):
    # data is assumed shape (C, ...)
    Ndata = data.shape[0]
    # Generate slicing indices
    slicing = torch.randint(0, Ndata, size=(Ndata, Nboot))

    # Build the index tuple
    index_tuple = tuple(
        slicing.T if ax == 0 else slice(None)
        for ax in range(data.ndim)
    )

    # Gather and average
    samples = data[index_tuple].mean(dim=1).to(dtype=data.dtype)
    return samples


def plot_training(trainer, fname, show=True,
                  plot_label="", plot_error=True,):
    i, j = trainer.loss_fn.kwargs["i"], trainer.loss_fn.kwargs["j"]
    sub_str = f"{i}{j}"

    for attr in ["train_exp", "train_var", "train_exp_err",
                 "test_exp", "test_var", "test_exp_err"]:
        setattr(trainer, attr, torch.as_tensor(getattr(trainer, attr)))

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(3, 5))

    ax[0].plot(range(trainer.epochs), trainer.train_exp,
               label="train", c="tab:blue")
    ax[0].plot(range(trainer.epochs), trainer.test_exp,
               label="test", c="tab:orange")
    ax[0].axhline(trainer.loss_fn.target_exp, c='k', label='undeformed')
    ax[0].set_ylabel(
        r'$\mathrm{Re}\left[\mathtt{Exp}Q_{'+sub_str+r'}\right]$')

    if plot_error:
        ax[0].fill_between(range(trainer.epochs),
                           trainer.train_exp + trainer.train_exp_err,
                           trainer.train_exp - trainer.train_exp_err,
                           (trainer.train_var/trainer.batch_size)**0.5,
                           color="tab:blue", alpha=0.3)
        ax[0].fill_between(range(trainer.epochs),
                           trainer.test_exp + trainer.test_exp_err,
                           trainer.test_exp - trainer.test_exp_err,
                           color="tab:orange", alpha=0.3)
        ax[0].fill_between(range(trainer.epochs),
                           trainer.loss_fn.target_exp + trainer.loss_fn.exp_err,
                           trainer.loss_fn.target_exp - trainer.loss_fn.exp_err,
                           color="k", alpha=0.1)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, framealpha=1.0)

    ax[1].plot(range(trainer.epochs), trainer.train_var, label="train")
    ax[1].plot(range(trainer.epochs), trainer.test_var, label="test")
    ax[1].axhline(trainer.loss_fn.initial_var, c='k', label='undeformed')
    ax[1].set_ylabel(r'$\mathtt{Var}\left[Q_{'+sub_str+r'}\right]$')

    # ax[0].set_ylim([trainer.loss_fn.target_exp*0.9,
    #               trainer.loss_fn.target_exp*1.1])
    ymin, ymax = ax[1].get_ylim()
    ax[1].set_ylim([ymin, trainer.loss_fn.initial_var*1.1])

    fig.text(0.95, 0.5, plot_label, va='center', ha='left', rotation=270)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    call_PDF(fname, show=show)
    print(f"plot saved to {fname}")
