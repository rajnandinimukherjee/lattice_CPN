from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from plot_settings import latex_font_params


plt.rcParams.update(latex_font_params)

WEIGHT_INIT_STD = 1e-3


def periodic_box_mask(
    lattice: Sequence[int],
    center: Sequence[int],
    radius: int | float,
) -> torch.Tensor:
    """Return a periodic boolean mask around ``center`` on a lattice."""
    if len(lattice) != len(center):
        raise ValueError(
            "lattice and center must have the same number of dimensions")

    mask = torch.ones(tuple(lattice), dtype=torch.bool)

    for L, c, dim in zip(lattice, center, range(len(lattice))):
        x = torch.arange(L)
        d = torch.minimum((x - c) % L, (c - x) % L)

        # Reshape so this dimension broadcasts against all others
        shape = [1] * len(lattice)
        shape[dim] = L

        mask &= d.reshape(shape) <= radius

    return mask


def st_dev(
    data: torch.Tensor,
    mean: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Return the population standard deviation over the first axis."""

    n = len(data)
    if mean is None:
        mean = torch.mean(data, dim=0)
    return (torch.sum((data - mean) ** 2, dim=0) / n) ** 0.5


def bootstrap(data: torch.Tensor, Nboot: int = 100) -> torch.Tensor:
    """Return bootstrap estimates of the mean over the first axis."""
    if data.shape[0] == 0:
        raise ValueError("cannot bootstrap an empty tensor")
    indices = torch.randint(
        data.shape[0],
        size=(Nboot, data.shape[0]),
        device=data.device,
    )
    return data[indices].mean(dim=1)


def plot_training(
    trainer: Trainer,
    fname: str | PathLike[str] | None = None,
    show: bool = True,
    plot_label: str = "",
    plot_error: bool = True,
) -> tuple[Any, Any]:
    """Plot the recorded training and validation expectation and variance."""
    i = getattr(trainer.loss_fn.obs, "i", "")
    j = getattr(trainer.loss_fn.obs, "j", "")
    sub_str = f"{i}{j}"

    for attr in [
        "train_exp",
        "train_var",
        "train_exp_err",
        "test_exp",
        "test_var",
        "test_exp_err",
    ]:
        setattr(trainer, attr, torch.as_tensor(getattr(trainer, attr)))

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(3, 5))

    ax[0].plot(range(trainer.epochs), trainer.train_exp,
               label="train", c="tab:blue")
    ax[0].plot(range(trainer.epochs), trainer.test_exp,
               label="test", c="tab:orange")
    ax[0].axhline(trainer.loss_fn.target_exp, c="k", label="undeformed")
    ax[0].set_ylabel(
        r"$\mathrm{Re}\left[\mathtt{Exp}Q_{" + sub_str + r"}\right]$")

    if plot_error:
        ax[0].fill_between(
            range(trainer.epochs),
            trainer.train_exp + trainer.train_exp_err,
            trainer.train_exp - trainer.train_exp_err,
            color="tab:blue",
            alpha=0.3,
        )
        ax[0].fill_between(
            range(trainer.epochs),
            trainer.test_exp + trainer.test_exp_err,
            trainer.test_exp - trainer.test_exp_err,
            color="tab:orange",
            alpha=0.3,
        )
        ax[0].fill_between(
            range(trainer.epochs),
            trainer.loss_fn.target_exp + trainer.loss_fn.exp_err,
            trainer.loss_fn.target_exp - trainer.loss_fn.exp_err,
            color="k",
            alpha=0.1,
        )

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, framealpha=1.0)

    ax[1].plot(range(trainer.epochs), trainer.train_var, label="train")
    ax[1].plot(range(trainer.epochs), trainer.test_var, label="test")
    ax[1].axhline(trainer.loss_fn.initial_var, c="k", label="undeformed")
    ax[1].set_ylabel(r"$\mathtt{Var}\left[Q_{" + sub_str + r"}\right]$")

    # ax[0].set_ylim([trainer.loss_fn.target_exp*0.9,
    #               trainer.loss_fn.target_exp*1.1])
    ymin, ymax = ax[1].get_ylim()
    ax[1].set_ylim([ymin, trainer.loss_fn.initial_var * 1.1])

    fig.text(0.95, 0.5, plot_label, va="center", ha="left", rotation=270)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_exported_training(
    data: Mapping[str, Any] | str | PathLike[str],
    fname: str | PathLike[str] | None = None,
    show: bool = True,
    plot_label: str = "",
    plot_error: bool = True,
) -> tuple[Any, Any]:
    """Plot training data exported by :meth:`Trainer.export_training_data`.

    ``data`` may be the exported dictionary or a filename produced with
    ``numpy.save``.  Unlike the live plot, every recorded epoch is shown and
    the variance panel uses a logarithmic y-axis.
    """
    if isinstance(data, (str, bytes)) or hasattr(data, "__fspath__"):
        data = np.load(data, allow_pickle=True).item()

    def values(key: str) -> np.ndarray:
        """Convert one exported series to a floating NumPy array."""
        return np.asarray(data[key], dtype=float)

    train_exp = values("train_exp")
    test_exp = values("test_exp")
    train_var = values("train_var")
    test_var = values("test_var")
    epochs = len(train_exp)
    x = np.arange(epochs)

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(7, 6))
    objective_name = data.get("objective_name", "variance")
    ax[0].plot(x, train_exp, label="train", c="tab:blue")
    ax[0].plot(x, test_exp, label="test", c="tab:red")
    ax[0].axhline(data["target_exp"], c="k", label="undeformed")
    if objective_name == "penalty":
        ax[0].set_ylabel(r"$\langle e^{-2\,\mathrm{Re}\,\Delta S}\rangle$")
        variance_label = r"$\mathrm{Var}(e^{-2\,\mathrm{Re}\,\Delta S})$"
    else:
        ax[0].set_ylabel(r"Re[Exp Q]")
        variance_label = r"Var[Q]"

    for lr_epoch in np.asarray(data.get("lr_change_epochs", []), dtype=int):
        for axis in ax:
            axis.axvspan(
                lr_epoch - 0.5, lr_epoch + 0.5, color="0.5", alpha=0.25, zorder=0
            )

    if plot_error:
        train_err = values("train_exp_err")
        test_err = values("test_exp_err")
        ax[0].fill_between(
            x, train_exp - train_err, train_exp + train_err, color="tab:blue", alpha=0.3
        )
        ax[0].fill_between(
            x, test_exp - test_err, test_exp + test_err, color="tab:red", alpha=0.3
        )
        target = data["target_exp"]
        error = data["exp_err"]
        ax[0].fill_between(x, target - error, target +
                           error, color="k", alpha=0.1)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, framealpha=1.0)

    ax[1].plot(x, train_var, label="train", c="tab:blue")
    ax[1].plot(x, test_var, label="test", c="tab:red")
    if data["initial_var"] > 0:
        ax[1].axhline(data["initial_var"], c="k", label="undeformed")
    ax[1].set_yscale("log")
    ax[1].set_ylabel(variance_label)
    ax[1].set_xlabel("Epoch")

    fig.text(0.95, 0.2, plot_label, va="center", ha="left", rotation=270)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def init_weights(m: nn.Module) -> None:
    """Initialize supported affine layer weights near zero."""
    if isinstance(
        m,
        (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ),
    ):
        nn.init.normal_(m.weight, mean=0.0, std=WEIGHT_INIT_STD)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Trainer:
    """Train deformation parameters against an observable loss."""

    warmup_epochs = 50

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        epochs: int,
        batch_size: int,
    ) -> None:
        """Initialize a trainer and its metric history."""
        self.model = model
        self.loss_fn = loss_fn
        if getattr(loss_fn, "obs", None) is None:
            raise ValueError("the loss must provide an observable with an input grid")
        self.grid = loss_fn.obs.grid
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        self.train_var: list[float] = []
        self.train_exp: list[float] = []
        self.train_exp_err: list[float] = []
        self.test_var: list[float] = []
        self.test_exp: list[float] = []
        self.test_exp_err: list[float] = []
        self.alpha_record: list[torch.Tensor] = []
        self.lr_change_epochs: list[int] = []
        self.lr_history: list[float] = []
        self.monitor_epochs: list[int] = []
        self.monitor_objective: list[float] = []
        self.stop_reason = "not_started"
        self.best_epoch: int | None = None
        self.training_config: dict[str, Any] = {}

    def export_training_data(self) -> dict[str, Any]:
        """Return the data needed to reproduce the training plot.

        The returned values are detached NumPy arrays/scalars so that the
        result can be passed directly to ``numpy.save`` (or serialized by a
        caller) without retaining the computation graph.
        """

        def as_numpy(values: Sequence[float]) -> np.ndarray:
            """Detach a numeric history and return it on the CPU."""
            return torch.as_tensor(values).detach().cpu().numpy()

        return dict(
            train_exp=as_numpy(self.train_exp),
            train_exp_err=as_numpy(self.train_exp_err),
            test_exp=as_numpy(self.test_exp),
            test_exp_err=as_numpy(self.test_exp_err),
            train_var=as_numpy(self.train_var),
            test_var=as_numpy(self.test_var),
            target_exp=float(self.loss_fn.target_exp),
            exp_err=float(self.loss_fn.exp_err),
            initial_var=float(self.loss_fn.initial_var),
            lr_change_epochs=torch.as_tensor(
                self.lr_change_epochs, dtype=torch.int64)
            .cpu()
            .numpy(),
            lr_history=as_numpy(self.lr_history),
            monitor_epochs=torch.as_tensor(
                self.monitor_epochs, dtype=torch.int64
            )
            .cpu()
            .numpy(),
            monitor_objective=as_numpy(self.monitor_objective),
            stop_reason=self.stop_reason,
            best_epoch=self.best_epoch,
            training_config=dict(self.training_config),
            # Keep these available for selecting the best trained network.
            alphas=[alpha.detach().cpu() for alpha in self.alpha_record],
            objective_name=getattr(
                self.loss_fn, "objective_name", type(self.loss_fn).__name__
            ),
            selection_metric=getattr(self.loss_fn, "selection_metric", "var"),
        )

    def _selection_history(self, sampletype: str) -> list[float]:
        """Return the history matching the loss's checkpoint objective."""
        metric = getattr(self.loss_fn, "selection_metric", "var")
        if metric not in {"exp", "var"}:
            raise ValueError(f"unsupported selection metric: {metric}")
        return getattr(self, f"{sampletype}_{metric}")

    def _initial_objective(self) -> float:
        """Return the undeformed value of the optimized quantity."""
        metric = getattr(self.loss_fn, "selection_metric", "var")
        return float(
            self.loss_fn.target_exp if metric == "exp" else self.loss_fn.initial_var
        )

    def _init_figure(self) -> tuple[Any, Any, dict[str, Any]]:
        """Set up the live-updating figure; return (fig, ax, artists)."""
        plt.ion()
        fig, ax = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
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
        return (
            fig,
            ax,
            dict(
                ev=ev_line,
                ev_init=ev_init_line,
                ev2=ev2_line,
                var=var_line,
                var2=var2_line,
                var_init=var_init_line,
                ev_fill=ev_fill,
                ev_init_fill=ev_init_fill,
                ev2_fill=ev2_fill,
            ),
        )

    def _redraw(
        self,
        fig: Any,
        ax: Any,
        artists: dict[str, Any],
        epoch: int,
        moving_window: bool = True,
    ) -> None:
        """Update all plot artists for the current epoch."""
        if moving_window:
            xdata = list(range(epoch - 50, epoch + 1))
        else:
            xdata = list(range(50, epoch + 1))

        exp_t = torch.tensor(self.train_exp)[xdata]
        std_t = torch.tensor(self.train_exp_err)[xdata]
        artists["ev_fill"].remove()
        artists["ev_fill"] = ax[0].fill_between(
            xdata,
            (exp_t - std_t).numpy(),
            (exp_t + std_t).numpy(),
            color="b",
            alpha=0.4,
        )

        exp_t2 = torch.tensor(self.test_exp)[xdata]
        std_t2 = torch.tensor(self.test_exp_err)[xdata]
        artists["ev2_fill"].remove()
        artists["ev2_fill"] = ax[0].fill_between(
            xdata,
            (exp_t2 - std_t2).numpy(),
            (exp_t2 + std_t2).numpy(),
            color="r",
            alpha=0.4,
        )

        artists["ev_init_fill"].remove()
        artists["ev_init_fill"] = ax[0].axhspan(
            self.loss_fn.target_exp - self.loss_fn.exp_err,
            self.loss_fn.target_exp + self.loss_fn.exp_err,
            color="k",
            alpha=0.1,
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

    def train(
        self,
        alphanet: nn.Module,
        mask: torch.Tensor | None = None,
        fixed_alphas: Mapping[tuple[int, ...], float] | None = None,
        plot: bool = False,
        max_training_time: float | None = None,
        plateau_window: int = 20,
        plateau_patience: int = 3,
        plateau_min_delta: float = 1e-4,
        plateau_threshold_mode: str = "rel",
        early_stopping: bool = False,
        extension: int = 100,
        lr_scheduler: str = "plateau",
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        lr_cooldown: int = 1,
        min_lr: float = 1e-5,
        validation_interval: int = 1,
        validation_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Train until the configured stopping condition is reached.

        Validation runs every ``validation_interval`` optimizer steps and can
        use a fixed random subset of the held-out split. Validation objectives
        are averaged over non-overlapping blocks of ``plateau_window`` such
        observations. Scheduler and early-stopping patience therefore count
        independent monitoring blocks rather than highly correlated sliding
        windows. If ``max_training_time`` is
        supplied, the initial epoch budget may be extended in blocks of
        ``extension`` while validation continues to improve. The time limit
        is enforced between epochs; an epoch already in progress finishes.
        Validation plateaus stop the run only when ``early_stopping`` is true.

        ``fixed_alphas`` maps complete output-field indices to hard values.
        Constraints are imposed after ``mask``, so they must refer to sites
        retained by that mask if both options are used.
        """
        if max_training_time is not None and max_training_time <= 0:
            raise ValueError("max_training_time must be positive")
        if plateau_window < 1 or plateau_patience < 1:
            raise ValueError(
                "plateau_window and plateau_patience must be positive")
        if plateau_min_delta < 0:
            raise ValueError("plateau_min_delta must be non-negative")
        if plateau_threshold_mode not in {"abs", "rel"}:
            raise ValueError(
                "plateau_threshold_mode must be 'abs' or 'rel'")
        if plateau_threshold_mode == "rel" and plateau_min_delta >= 1:
            raise ValueError(
                "relative plateau_min_delta must be smaller than 1")
        if extension < 1:
            raise ValueError("extension must be positive")
        if lr_scheduler not in {"constant", "cosine", "plateau"}:
            raise ValueError(
                "lr_scheduler must be 'constant', 'cosine', or 'plateau'")
        if not 0 < lr_factor < 1:
            raise ValueError("lr_factor must be between 0 and 1")
        if lr_patience < 1 or lr_cooldown < 0 or min_lr < 0:
            raise ValueError(
                "lr_patience must be positive; lr_cooldown and min_lr non-negative"
            )
        if validation_interval < 1:
            raise ValueError("validation_interval must be positive")
        if validation_samples is not None and validation_samples < 2:
            raise ValueError("validation_samples must be at least 2")
        initial_lrs = [
            float(group["lr"]) for group in self.optimizer.param_groups
        ]
        if lr_scheduler != "constant" and min_lr > min(initial_lrs):
            raise ValueError("min_lr cannot exceed the initial learning rate")

        initial_epochs = self.epochs
        planned_epochs = initial_epochs
        plateau_count = 0
        best_rolling_objective = float("inf")
        validation_block: list[float] = []
        start_time = time.monotonic()
        epoch_times = []
        plateau_scheduler = None
        cosine_scheduler = None
        if lr_scheduler == "plateau":
            plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=lr_factor,
                patience=lr_patience,
                threshold=plateau_min_delta,
                threshold_mode=plateau_threshold_mode,
                cooldown=lr_cooldown,
                min_lr=min_lr,
            )
        elif lr_scheduler == "cosine":
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(initial_epochs, 1),
                eta_min=min_lr,
            )

        self.stop_reason = "running"
        self.training_config = {
            "batch_size": int(self.batch_size),
            "n_train": int(self.model.samples["train"].size(0)),
            "n_test": int(self.model.samples["test"].size(0)),
            "initial_learning_rates": initial_lrs,
            "lr_scheduler": lr_scheduler,
            "lr_factor": float(lr_factor),
            "lr_patience": int(lr_patience),
            "lr_cooldown": int(lr_cooldown),
            "min_lr": float(min_lr),
            "plateau_window": int(plateau_window),
            "plateau_patience": int(plateau_patience),
            "plateau_min_delta": float(plateau_min_delta),
            "plateau_threshold_mode": plateau_threshold_mode,
            "early_stopping": bool(early_stopping),
            "validation_interval": int(validation_interval),
            "validation_samples": (
                None if validation_samples is None else int(validation_samples)
            ),
        }

        full_test_fields = self.model.samples["test"]
        validation_fields = full_test_fields
        link_samples = getattr(self.model, "link_samples", None)
        full_test_links = (
            link_samples.get("test") if link_samples is not None else None
        )
        validation_links = full_test_links
        if (
            validation_samples is not None
            and validation_samples < len(full_test_fields)
        ):
            validation_indices = torch.randperm(
                len(full_test_fields), device=full_test_fields.device
            )[:validation_samples]
            validation_fields = full_test_fields.index_select(
                0, validation_indices
            )
            if full_test_links is not None:
                validation_links = full_test_links.index_select(
                    0, validation_indices
                )

        pbar = tqdm(total=planned_epochs)
        if plot:
            fig, ax, artists = self._init_figure()

        epoch = 0
        while epoch < planned_epochs:
            epoch_start = time.monotonic()
            if (
                max_training_time is not None
                and epoch > 0
                and time.monotonic() - start_time >= max_training_time
            ):
                self.stop_reason = "time_limit"
                break

            self.lr_history.append(
                float(self.optimizer.param_groups[0]["lr"])
            )
            self.optimizer.zero_grad()
            batch_idx = torch.randperm(
                self.model.samples["train"].size(0),
                device=self.model.samples["train"].device,
            )[: self.batch_size]
            alphas = alphanet(self.grid)
            if mask is not None:
                alphas = alphas * mask
            if fixed_alphas:
                alphas = alphas.clone()
                for index, value in fixed_alphas.items():
                    if len(index) != alphas.ndim:
                        raise ValueError(
                            "fixed alpha indices must address every output axis"
                        )
                    alphas[index] = value
            self.alpha_record.append(alphas.detach().clone())

            loss = self.loss_fn(
                alphas=alphas, batch_idx=batch_idx, sampletype="train")
            loss.backward()
            self.optimizer.step()

            # Evaluate the held-out objective at the requested cadence.
            warmed_up = epoch > self.warmup_epochs
            validation_due = warmed_up and (
                (epoch - self.warmup_epochs) % validation_interval == 0
            )
            if validation_due:
                self.model.samples["test"] = validation_fields
                if link_samples is not None and validation_links is not None:
                    link_samples["test"] = validation_links
                try:
                    test_obs = self.loss_fn.get_obs(
                        alphas=alphas, sampletype="test"
                    )
                finally:
                    self.model.samples["test"] = full_test_fields
                    if link_samples is not None and full_test_links is not None:
                        link_samples["test"] = full_test_links
                self.test_exp.append(test_obs.mean().item())
                self.test_var.append(test_obs.var(correction=0).item())
                self.test_exp_err.append(
                    bootstrap(test_obs).var().item() ** 0.5)

            else:
                self.test_exp.append(float("nan"))
                self.test_var.append(float("nan"))
                self.test_exp_err.append(float("nan"))

            # Train metrics
            train_obs = self.loss_fn.get_obs(
                alphas=alphas, batch_idx=batch_idx, sampletype="train"
            )
            self.train_exp.append(train_obs.mean().item())
            self.train_var.append(train_obs.var(correction=0).item())
            self.train_exp_err.append(bootstrap(train_obs).var().item() ** 0.5)

            if validation_due and plot:
                self._redraw(fig, ax, artists, epoch, **kwargs)

            test_objective = self._selection_history("test")
            current_loss = (
                test_objective[-1]
                if test_objective and np.isfinite(test_objective[-1])
                else loss.item()
            )
            pbar.set_description(
                f"loss: {current_loss:.4f} vs {self._initial_objective():.4f}"
            )
            pbar.update(1)
            epoch += 1
            epoch_times.append(time.monotonic() - epoch_start)

            if cosine_scheduler is not None and epoch <= initial_epochs:
                cosine_scheduler.step()

            # Average disjoint validation blocks. Feeding overlapping rolling
            # means to a patience counter makes it reduce the learning rate on
            # many almost-identical observations.
            rolling_objective = None
            if warmed_up and np.isfinite(test_objective[-1]):
                validation_block.append(float(test_objective[-1]))
            if len(validation_block) == plateau_window:
                rolling_objective = sum(validation_block) / plateau_window
                validation_block.clear()
                self.monitor_epochs.append(epoch - 1)
                self.monitor_objective.append(rolling_objective)
                previous_lrs = [
                    param_group["lr"] for param_group in self.optimizer.param_groups
                ]
                if plateau_scheduler is not None:
                    plateau_scheduler.step(rolling_objective)
                if plateau_scheduler is not None and any(
                    param_group["lr"] != previous_lr
                    for param_group, previous_lr in zip(
                        self.optimizer.param_groups, previous_lrs
                    )
                ):
                    # The scheduler responds to the metric from the epoch
                    # just completed, whose index is ``epoch - 1`` here.
                    self.lr_change_epochs.append(epoch - 1)
                if plateau_threshold_mode == "abs":
                    improved = (
                        rolling_objective
                        < best_rolling_objective - plateau_min_delta
                    )
                else:
                    improved = (
                        rolling_objective
                        < best_rolling_objective * (1.0 - plateau_min_delta)
                    )
                if improved:
                    best_rolling_objective = rolling_objective
                    plateau_count = 0
                else:
                    plateau_count += 1

            if rolling_objective is not None:
                pbar.set_postfix(
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    rolling_objective=f"{rolling_objective:.4g}",
                )

                if early_stopping and plateau_count >= plateau_patience:
                    self.stop_reason = "early_stopping"
                    break

            if epoch >= planned_epochs:
                if max_training_time is None:
                    self.stop_reason = "epoch_budget"
                    break

                # Estimate the maximum number of epochs that fit in the
                # remaining wall-clock budget.  The wall-clock check above
                # remains authoritative if epoch times vary.
                mean_epoch_time = sum(epoch_times) / len(epoch_times)
                time_limited_epochs = max(
                    epoch,
                    int(max_training_time / mean_epoch_time),
                )
                if epoch >= time_limited_epochs:
                    self.stop_reason = "time_limit"
                    break

                planned_epochs = min(
                    planned_epochs + extension,
                    time_limited_epochs,
                )
                pbar.total = planned_epochs
                pbar.refresh()

        pbar.close()
        if self.stop_reason == "running":
            self.stop_reason = "epoch_budget"
        self.epochs = len(self.train_var)
        plt.ioff()
        # plt.close("all")

    def best_alphas(self, window: int = 20) -> torch.Tensor:
        """Return the best validation field over the complete training run.

        ``window`` smooths checkpoint scores over consecutive validation
        epochs. The field at the end of the lowest-scoring window is
        returned. Training scores are used only when no validation score is
        finite.
        """
        if not self.alpha_record:
            raise RuntimeError("training has not recorded any alphas")
        if window < 1:
            raise ValueError("window must be positive")

        def smoothed_scores(values: Sequence[float]) -> torch.Tensor:
            history = torch.as_tensor(values, dtype=torch.float64)
            scores = torch.full_like(history, torch.inf)
            observed = torch.nonzero(
                torch.isfinite(history), as_tuple=False
            ).flatten()
            for end in range(window - 1, len(observed)):
                block_indices = observed[end - window + 1:end + 1]
                scores[block_indices[-1]] = history[block_indices].mean()
            if not torch.isfinite(scores).any():
                scores = torch.where(
                    torch.isfinite(history), history, torch.inf
                )
            return scores

        scores = smoothed_scores(self._selection_history("test"))
        if not torch.isfinite(scores).any():
            scores = smoothed_scores(self._selection_history("train"))
        best_index = int(torch.argmin(scores).item())
        self.best_epoch = best_index
        return self.alpha_record[best_index]
