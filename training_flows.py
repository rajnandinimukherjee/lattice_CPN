import pdb

import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# import training_utils as tu
import actions
import observables
from argparsing import parseargs
# from plot_settings import plotparams
from deformations import FlowedDeformations
from losses import PINNLoss, bootstrap
from networks import UNet

# plt.rcParams.update(plotparams)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=1e-3)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Trainer:
    def __init__(self, model, loss_fn, optimizer, epochs, batch_size):
        self.model = model
        self.loss_fn = loss_fn
        self.grid = loss_fn.obs.grid

        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_size = self.model.samples["train"].size(0)
        self.test_size = self.model.samples["test"].size(0)
        self.name = f"{self.model}"

        self.init = bootstrap(self.loss_fn.obs(model=self.model, deform=False,
                                               sampletype="all").real)
        self.target_exp = self.init.mean().item()
        self.initial_var = self.init.var().item()

        self.train_var, self.train_exp, self.train_loss = [], [], []
        self.test_var, self.test_exp, self.test_loss = [], [], []

    def train(self, alphanet):
        plt.ion()
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True,
                               gridspec_kw={'hspace': 0},
                               figsize=(9, 6))
        EV, = ax[0].plot([], [], "bo-")
        EV_err = ax[0].fill_between([], [], [], color="b", alpha=0.4)
        EV2, = ax[0].plot([], [], "ro-")
        EV2_err = ax[0].fill_between([], [], [], color="r", alpha=0.4)
        EV_init, = ax[0].plot([], [], "k-")
        EV_init_err = ax[0].axhspan(0, 0, color='k', alpha=0.1)
        VAR, = ax[1].plot([], [], "bo-")
        VAR2, = ax[1].plot([], [], "ro-")
        VAR_init, = ax[1].plot([], [], "k-")
        PINN, = ax[2].plot([], [], "bo-")
        PINN2, = ax[2].plot([], [], "ro-")
        ax[1].set_xlabel("Iteration")
        ax[0].set_ylabel(r"Exp[Re$O_{ij}$]")
        ax[1].set_ylabel(r"Var[$O_{ij}$]")
        ax[2].set_ylabel(
            r"$(\nabla\cdot F_\mathrm{re}-F_\mathrm{re}\cdot\nabla S)/\sigma$")
        fig.suptitle(self.loss_fn.obs)

        pbar = tqdm(range(self.epochs))
        plt.show(block=False)
        EV_std, EV2_std = [], []
        PINN_diff_train, PINN_diff_test = [], []
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            batch_idx = torch.randperm(
                self.model.samples["train"].size(0))[:self.batch_size]

            alphas = alphanet(self.grid)
            loss = self.loss_fn(
                alphas=alphas, batch_idx=batch_idx, sampletype="train")
            loss.backward()
            self.optimizer.step()

            # =====record==================================================
            self.train_loss.append(loss.item())
            train_obs = bootstrap(self.loss_fn.get_obs(
                alphas=alphas,
                batch_idx=batch_idx,
                sampletype="train").real)
            self.train_var.append(train_obs.var().item())
            self.train_exp.append(train_obs.mean().item())
            pinn_diff_train = bootstrap(self.loss_fn.PINN(
                alphas,
                sampletype="train",
                batch_idx=batch_idx,
                FS_only=True).real)
            PINN_diff_train.append([pinn_diff_train.mean().item() /
                                    pinn_diff_train.var().item()**0.5])

            EV_std.append(train_obs.var().item()**0.5)

            if epoch > 50:
                test_obs = bootstrap(self.loss_fn.get_obs(
                    alphas=alphas, sampletype="test").real)
                self.test_var.append(test_obs.var().item())
                self.test_exp.append(test_obs.mean().item())
                self.test_loss.append(self.loss_fn(alphas=alphas,
                                                   sampletype="test").item())
                pinn_diff_test = bootstrap(self.loss_fn.PINN(
                    alphas, sampletype="test", FS_only=True).real)
                PINN_diff_test.append(pinn_diff_test.mean().item() /
                                      pinn_diff_test.var().item()**0.5)
                EV2_std.append(test_obs.var().item()**0.5)
            else:
                self.test_var.append(torch.nan)
                self.test_exp.append(torch.nan)
                self.test_loss.append(torch.nan)
                PINN_diff_test.append(torch.nan)
                EV2_std.append(torch.nan)

            # if train_obs.var().item() < 1e-4:
            #    for param_grp in self.optimizer.param_groups:
            #        param_grp['lr'] *= 0.1
            #        print(f"lr is now {param_grp['lr']}")

            # =====plotting================================================
            xdata = list(range(epoch+1))
            ev_init_ydata = [self.target_exp]*len(xdata)
            var_init_ydata = [self.initial_var]*len(xdata)
            ev_ydata = self.train_exp
            ev2_ydata = self.test_exp
            var_ydata = self.train_var
            var2_ydata = self.test_var
            # pinn_ydata = self.train_loss
            # pinn2_ydata = self.test_loss
            pinn_ydata = PINN_diff_train
            pinn2_ydata = PINN_diff_test

            EV_err.remove()
            ymax = (torch.tensor(self.train_exp)+torch.tensor(EV_std)).numpy()
            ymin = (torch.tensor(self.train_exp)-torch.tensor(EV_std)).numpy()

            EV_err = ax[0].fill_between(
                xdata, ymin, ymax, color='b', alpha=0.4)

            EV2_err.remove()
            ymax = (torch.tensor(self.test_exp)+torch.tensor(EV2_std)).numpy()
            ymin = (torch.tensor(self.test_exp)-torch.tensor(EV2_std)).numpy()

            EV2_err = ax[0].fill_between(
                xdata, ymin, ymax, color='r', alpha=0.4)

            EV_init_err.remove()
            err = self.initial_var**0.5
            EV_init_err = ax[0].axhspan(self.target_exp+err,
                                        self.target_exp-err,
                                        color='k', alpha=0.1)

            for line in [EV, EV2, VAR, VAR2, EV_init, VAR_init, PINN, PINN2]:
                line.set_xdata(xdata[-50:])
            EV.set_ydata(ev_ydata[-50:])
            EV2.set_ydata(ev2_ydata[-50:])
            VAR.set_ydata(var_ydata[-50:])
            VAR2.set_ydata(var2_ydata[-50:])
            EV_init.set_ydata(ev_init_ydata[-50:])
            VAR_init.set_ydata(var_init_ydata[-50:])
            PINN.set_ydata(pinn_ydata[-50:])
            PINN2.set_ydata(pinn2_ydata[-50:])

            for idx in range(3):
                ax[idx].relim()
                ax[idx].autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

            if epoch > 50:
                pbar.set_description(
                    f"loss: {self.test_var[-1]:.8f} vs {self.initial_var:.8f}")
            else:
                pbar.set_description(
                    f"loss: {loss.item():.8f} vs {self.initial_var:.8f}")
            pbar.update(1)

            if epoch % 50 == 0:
                fig1, ax1 = plt.subplots()
                ax1.imshow(alphas[:, :, 6].detach().numpy())

        pbar.close()
        plt.ioff()
        # plt.close('all')


if __name__ == "__main__":
    args, deformation = parseargs()
    N = args.N

    if args.source == "Raj":
        file = h5py.File(f"data/CP{N}_lat{'x'.join([str(n) for n in args.latsize])
                                          }_b{args.beta:.1f}.h5", 'r')["configs"]
        burn = 2000  # int(file["info"]["burn"][0])
        stepsize = 20  # int(file["info"]["stepsize"][0])
        samples = torch.tensor(
            file["vectors"][burn::stepsize],
            dtype=torch.complex128)
    else:
        samples = torch.from_file("data/cpn_b4.0_L64_Nc3_ens.dat",
                                  shared=False,
                                  size=1000*64*64*3,
                                  dtype=torch.complex128).reshape(1000, 64, 64, 3)[50:]
    N_conf = len(samples)
    print('\n'.join([
        "Ensemble details:",
        f"source: {args.source}",
        f"size: {args.latsize}",
        f"beta: {args.beta}",
        f"N_conf: {N_conf}",
        f"E/V: {actions.LatticeEnergy(samples, batched=True).mean() /
                (args.latsize[0]*args.latsize[1])}"
    ]))

    train_indices = torch.randperm(N_conf)[:int(args.split*N_conf)]
    mask = torch.zeros(len(samples), dtype=torch.bool)
    mask[train_indices] = True

    myAction = actions.LatticeAction(beta=args.beta)
    myObs = observables.TwoPtFn(N=N, latsize=args.latsize, i=args.i, j=args.j,
                                delta=args.delta, action=myAction)
    model = FlowedDeformations(N)
    model.samples = {
        "all": samples,
        "train": samples[mask],
        "test": samples[~mask]
    }
    loss_fn = PINNLoss(model, samples, N, myObs)

    netargs = {
        "input_dim": 2*(N+1),
        "output_dim": 2*model.DOF,
    }

    alphanet = UNet(**netargs)
    alphanet.apply(init_weights)
    optimizer = optim.Adam(alphanet.parameters(), lr=1e-3)
    trainer = Trainer(model, loss_fn, optimizer,
                      args.epochs, args.batch_size)

    if args.run:
        trainer.train(alphanet)
