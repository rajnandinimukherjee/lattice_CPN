import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pdb

import actions
import observables
from argparsing import parseargs
from deformations import HomogenousDeformations
from networks import UNet
from losses import VarianceLoss
from training_utils import bootstrap

WARMUP_EPOCHS = 50
BEST_WINDOW = 20
LEARNING_RATE = 1e-3
WEIGHT_INIT_STD = 1e-3


def init_weights(m: nn.Module) -> None:
    """Small normal weights and zero bias for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=WEIGHT_INIT_STD)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Trainer:
    def __init__(self, model, loss_fn, optimizer, epochs: int, batch_size: int):
        self.model = model
        self.loss_fn = loss_fn
        self.grid = loss_fn.obs.grid
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        self.train_var:        list[float] = []
        self.train_exp:        list[float] = []
        self.train_exp_err:    list[float] = []
        self.test_var:         list[float] = []
        self.test_exp:         list[float] = []
        self.test_exp_err:     list[float] = []
        self.alpha_record:     list = []

    def _init_figure(self):
        """Set up the live-updating figure; return (fig, ax, artists)."""
        plt.ion()
        fig, ax = plt.subplots(
            nrows=2, ncols=1, sharex=True,
            gridspec_kw={"hspace": 0},
            figsize=(7, 6),
        )
        ax[0].set_ylabel("exp")
        ax[1].set_ylabel("var")
        ax[1].set_xlabel("Iteration")
        fig.suptitle(self.loss_fn.obs)

        (ev_line,) = ax[0].plot([], [], "bo-")
        (ev2_line,) = ax[0].plot([], [], "ro-")
        (ev_init_line,) = ax[0].plot([], [], "k-")
        (var_line,) = ax[1].plot([], [], "bo-")
        (var2_line,) = ax[1].plot([], [], "ro-")
        (var_init_line,) = ax[1].plot([], [], "k-")

        ev_fill = ax[0].fill_between([], [], [], color="b", alpha=0.4)
        ev2_fill = ax[0].fill_between([], [], [], color="r", alpha=0.4)
        ev_init_fill = ax[0].axhspan(0, 0, color="k", alpha=0.1)

        plt.show(block=False)
        return fig, ax, dict(
            ev=ev_line, ev_init=ev_init_line,
            ev2=ev2_line,
            var=var_line, var2=var2_line, var_init=var_init_line,
            ev_fill=ev_fill, ev_init_fill=ev_init_fill,
            ev2_fill=ev2_fill,
        )

    def _redraw(self, fig, ax, artists, epoch: int) -> None:
        """Update all plot artists for the current epoch."""
        xdata = list(range(epoch-50, epoch + 1))

        exp_t = torch.tensor(self.train_exp)[xdata]
        std_t = torch.tensor(self.train_exp_err)[xdata]
        artists["ev_fill"].remove()
        artists["ev_fill"] = ax[0].fill_between(
            xdata,
            (exp_t - std_t).numpy(),
            (exp_t + std_t).numpy(),
            color="b", alpha=0.4,
        )

        exp_t2 = torch.tensor(self.test_exp)[xdata]
        std_t2 = torch.tensor(self.test_exp_err)[xdata]
        artists["ev2_fill"].remove()
        artists["ev2_fill"] = ax[0].fill_between(
            xdata,
            (exp_t2 - std_t2).numpy(),
            (exp_t2 + std_t2).numpy(),
            color="r", alpha=0.4,
        )

        artists["ev_init_fill"].remove()
        artists["ev_init_fill"] = ax[0].axhspan(
            self.loss_fn.target_exp - self.loss_fn.exp_err,
            self.loss_fn.target_exp + self.loss_fn.exp_err,
            color="k", alpha=0.1,
        )

        artists["ev"].set_data(xdata, torch.tensor(self.train_exp)[xdata])
        artists["ev2"].set_data(xdata, torch.tensor(self.test_exp)[xdata])
        artists["var"].set_data(xdata, torch.tensor(self.train_var)[xdata])
        artists["var2"].set_data(xdata, torch.tensor(self.test_var)[xdata])
        artists["ev_init"].set_data(
            xdata, [self.loss_fn.target_exp] * len(xdata))
        artists["var_init"].set_data(
            xdata, [self.loss_fn.initial_var] * len(xdata))

        for a in ax:
            a.relim()
            a.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()

    def train(self, alphanet) -> None:
        fig, ax, artists = self._init_figure()
        pbar = tqdm(range(self.epochs))
        lr_reduced = False

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            batch_idx = torch.randperm(
                self.model.samples["train"].size(0)
            )[: self.batch_size]
            alphas = alphanet(self.grid)
            self.alpha_record.append(alphas)

            loss = self.loss_fn(
                alphas=alphas, batch_idx=batch_idx, sampletype="train")
            loss.backward()
            self.optimizer.step()

            # Test metrics (only after warmup)
            warmed_up = epoch > WARMUP_EPOCHS
            if warmed_up:
                test_obs = self.loss_fn.get_obs(
                    alphas=alphas, sampletype="test")
                self.test_exp.append(test_obs.mean().item())
                self.test_var.append(test_obs.var(correction=0).item())
                self.test_exp_err.append(bootstrap(test_obs).var().item()**0.5)

                # Reduce lr if test_var < init_var
                lr_ratio = self.test_var[-1]/self.loss_fn.initial_var
                if (not lr_reduced) and (lr_ratio < 1.0):
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = LEARNING_RATE/10
                        lr_reduced = True

            else:
                self.test_exp.append(float("nan"))
                self.test_var.append(float("nan"))
                self.test_exp_err.append(float("nan"))

            # Train metrics
            train_obs = self.loss_fn.get_obs(
                alphas=alphas,
                batch_idx=batch_idx,
                sampletype="train")
            self.train_exp.append(train_obs.mean().item())
            self.train_var.append(train_obs.var(correction=0).item())
            self.train_exp_err.append(bootstrap(train_obs).var().item()**0.5)

            if warmed_up:
                self._redraw(fig, ax, artists, epoch)

            current_loss = self.test_var[-1] if warmed_up else loss.item()
            pbar.set_description(
                f"loss: {current_loss:.4f} vs {self.loss_fn.initial_var:.4f}"
            )
            pbar.update(1)

        pbar.close()
        plt.ioff()
        # plt.close("all")


def load_samples(args) -> torch.Tensor:
    """Load lattice gauge configurations and return a complex128 tensor."""
    if args.source == "Raj":
        lat_str = "x".join(str(n) for n in args.latsize)
        configs = h5py.File(
            f"data/CP{args.N}_lat{lat_str}_b{args.beta:.1f}.h5", "r"
        )["configs"]
        return torch.tensor(
            configs["vectors"][2000::20], dtype=torch.complex128
        )

    return (
        torch.from_file(
            "data/cpn_b4.0_L64_Nc3_big_ens.dat",
            shared=False,
            size=2000 * 64 * 64 * 3,
            dtype=torch.complex128,
        )
        .reshape(2000, 64, 64, 3)[50:]
    )


if __name__ == "__main__":
    args, _ = parseargs()
    N = args.N

    samples = load_samples(args)
    N_conf = len(samples)
    lat_vol = args.latsize[0] * args.latsize[1]

    print("\n".join([
        "Ensemble details:",
        f"  source : {args.source}",
        f"  size   : {args.latsize}",
        f"  beta   : {args.beta}",
        f"  N_conf : {N_conf}",
        f"  E/V    : {actions.LatticeEnergy(samples,
                                            batched=True).mean() / lat_vol:.6f}",
    ]))

    # Train / test split
    train_idx = torch.randperm(N_conf)[: int(args.split * N_conf)]
    mask = torch.zeros(N_conf, dtype=torch.bool)
    mask[train_idx] = True

    # Observables and model
    my_action = actions.LatticeAction(beta=args.beta)
    # my_obs = observables.OnePtFn(
    #    N=N, latsize=args.latsize,
    #    i=args.i, j=args.j,
    #    coord=(0, 0),
    #    action=my_action,
    # )
    my_obs = observables.TwoPtFn(
        N=N, latsize=args.latsize,
        i=args.i, j=args.j, delta=args.delta,
        coord=(0, 0), action=my_action,
    )
    model = HomogenousDeformations(N)
    model.samples = {"all": samples,
                     "train": samples[mask], "test": samples[~mask]}

    loss_fn = VarianceLoss(model, samples, N, my_obs)
    alphanet = UNet(
        input_dim=loss_fn.obs.grid.shape[-1],
        output_dim=model.DOF,
    )
    alphanet.apply(init_weights)

    optimizer = optim.Adam(alphanet.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(model, loss_fn, optimizer, args.epochs, args.batch_size)

    # Pre-training observable statistics
    print(f"<O> = {loss_fn.target_exp:.6f},\tVar(O) = {
          loss_fn.initial_var:.6f}")
    ud_O = my_obs(model)
    print(f"|<O>|^2 = {torch.abs(ud_O.mean()).item() ** 2:.6f}")
    print(f"<|O|>^2 = {ud_O.abs().mean().item() ** 2:.6f}")
    print(f"<|O|^2> = {(ud_O.abs() ** 2).mean().item():.6f}")

    if not args.run:
        raise SystemExit(0)

    trainer.train(alphanet)

    # Pick best checkpoint from the final window
    window = trainer.test_var[-BEST_WINDOW:]
    best_alphas = trainer.alpha_record[-BEST_WINDOW:][torch.argmin(
        torch.tensor(window)).item()]

    # analytic_loss_rat = (
    #    trainer.loss_fn.var(alphas=loss_fn.obs.invSqAlphas, sampletype="all")
    #    / loss_fn.initial_var
    # )
    best_training_loss_rat = (
        trainer.loss_fn.var(alphas=best_alphas, sampletype="all")
        / loss_fn.initial_var
    )

    if torch.isnan(best_training_loss_rat):
        raise RuntimeError(
            f"best_training_loss_rat is NaN for delta={args.delta}. "
            "Ensure the test window falls entirely after the warmup period."
        )

    fname = f"data/training_params_i{args.i}_j{args.j}_Nepochs{args.epochs}.pt"
    torch.save(alphanet, fname)
    # saved_data = torch.load(fname, weights_only=False)
    # saved_data[0, *args.delta] = analytic_loss_rat.item()
    # saved_data[1, *args.delta] = best_training_loss_rat.item()
    # torch.save(saved_data, fname)
    # print(f"Saved analytic and best training loss ratios to {
    #      fname} for {args.delta}.")
