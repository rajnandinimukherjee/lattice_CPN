from __future__ import annotations

import argparse
from collections.abc import Sequence
from os import PathLike

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from actions import CPnAction
from CPN_analysis import parse_int_tuple
from CPN_utils import oneCPN, randCPN, randSUN, small_randSUN


def cartesian_indices(axes: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return Cartesian-product indices with a consistent final coordinate axis."""
    indices = torch.cartesian_prod(*axes)
    return indices.unsqueeze(-1) if len(axes) == 1 else indices


def metropolis(
    old_config: torch.Tensor,
    action: CPnAction,
    N: int,
    **actionkwargs: object,
) -> tuple[torch.Tensor, float]:
    """Perform a one-site-at-a-time Metropolis sweep."""
    latsize = old_config.shape[:-1]
    random_rotations = randSUN(N + 1, size=latsize, dtype=old_config.dtype)
    new_config = old_config.clone()
    sites = cartesian_indices([torch.arange(length) for length in latsize])
    accepts = 0

    for index in sites:
        site = tuple(index.tolist())
        old_action = action(new_config, **actionkwargs)
        previous_value = new_config[site].clone()
        new_config[site] = random_rotations[site] @ previous_value
        new_action = action(new_config, **actionkwargs)
        probability = torch.exp(-(new_action - old_action).real).clamp(max=1.0)
        if torch.rand(()).item() < probability.item():
            accepts += 1
        else:
            new_config[site] = previous_value

    return new_config, accepts / len(sites)


def checkerboard_sites(
    latsize: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return even and odd site indices for a bipartite lattice."""
    coordinates = cartesian_indices([torch.arange(length) for length in latsize])
    parities = coordinates.sum(-1) % 2
    return coordinates[parities == 0], coordinates[parities == 1]


def metropolis_even_odd(
    old_config: torch.Tensor,
    action: CPnAction,
    N: int,
    **actionkwargs: object,
) -> tuple[torch.Tensor, float]:
    """Perform batched Metropolis updates on even and odd sublattices."""
    config = old_config.clone()
    latsize = config.shape[:-1]
    if any(length % 2 for length in latsize):
        raise ValueError("checkerboard updates require even lattice lengths")
    even_sites, odd_sites = checkerboard_sites(latsize)
    random_rotations = small_randSUN(N + 1, size=latsize, eps=0.3).to(config.dtype)
    acceptance_rates = []

    for sites in (even_sites, odd_sites):
        site_index = tuple(sites.T)
        old_values = config[site_index].clone()
        old_action = action(config, local=True, sites=sites, **actionkwargs)
        proposals = torch.einsum("mij,mj->mi", random_rotations[site_index], old_values)
        config[site_index] = proposals
        new_action = action(config, local=True, sites=sites, **actionkwargs)
        probabilities = torch.exp(-(new_action - old_action).real).clamp(max=1.0)
        accepted = torch.rand_like(probabilities) < probabilities
        config[site_index] = torch.where(accepted[:, None], proposals, old_values)
        acceptance_rates.append(accepted.float().mean().item())

    return config, sum(acceptance_rates) / len(acceptance_rates)


def metropolis_block(
    old_config: torch.Tensor,
    action: CPnAction,
    N: int,
    block_size: Sequence[int],
    **actionkwargs: object,
) -> tuple[torch.Tensor, float]:
    """Perform Metropolis updates using rigid rotations of lattice blocks."""
    latsize = old_config.shape[:-1]
    if len(block_size) != len(latsize):
        raise ValueError("block_size and lattice must have the same dimension")
    if any(size <= 0 for size in block_size):
        raise ValueError("block sizes must be positive")
    starts = cartesian_indices(
        [torch.arange(0, length, step) for length, step in zip(latsize, block_size)]
    )
    new_config = old_config.clone()
    accepts = 0

    for block_start in starts:
        block_slice = tuple(
            slice(int(start), min(int(start + size), length))
            for start, size, length in zip(block_start, block_size, latsize)
        )
        rotation = randSUN(N + 1, dtype=old_config.dtype)
        proposal = new_config.clone()
        proposal[block_slice] = torch.einsum(
            "ij,...j->...i", rotation, new_config[block_slice]
        )
        delta_action = action(proposal, **actionkwargs) - action(
            new_config, **actionkwargs
        )
        probability = torch.exp(-delta_action.real).clamp(max=1.0)
        if torch.rand(()).item() < probability.item():
            new_config[block_slice] = proposal[block_slice]
            accepts += 1

    return new_config, accepts / len(starts)


def MCMC(
    latsize: Sequence[int],
    iters: int = 10_000,
    burn: int = 1_000,
    stepsize: int = 1,
    N: int = 2,
    save: bool = False,
    fname: str | PathLike[str] = "",
    beta: float = 4.0,
    block: bool = True,
    block_size: Sequence[int] | None = None,
    start: str = "hot",
    evenodd: bool = False,
    action: CPnAction | None = None,
    plot: bool = True,
) -> torch.Tensor | None:
    """Generate a CP(N) Markov chain with the requested update scheme."""
    if iters < 2:
        raise ValueError("iters must be at least two")
    if not 0 <= burn < iters:
        raise ValueError("burn must satisfy 0 <= burn < iters")
    if start not in {"hot", "cold"}:
        raise ValueError("start must be 'hot' or 'cold'")

    lattice = tuple(latsize)
    if not lattice or any(length <= 0 for length in lattice):
        raise ValueError("lattice lengths must be positive")
    lattice_action = action or CPnAction(beta, N, lattice)
    configurations = torch.zeros(size=(iters, *lattice, N + 1), dtype=torch.cdouble)
    configurations[0] = (
        oneCPN(N, size=lattice) if start == "cold" else randCPN(N, size=lattice)
    )
    if block_size is None:
        block_size = (2,) * len(lattice)

    figure = axes = energy_line = ess_line = energy_average = None
    if plot:
        plt.ion()
        figure, axes = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            gridspec_kw={"hspace": 0},
        )
        (energy_line,) = axes[0].plot([], [], "bo-")
        energy_average = axes[0].errorbar([], [], yerr=[], fmt="r^")
        (ess_line,) = axes[1].plot([], [], "ro-")
        axes[1].set_xlabel("Iteration")
        axes[0].set_ylabel("E/V")
        axes[1].set_ylabel("ESS")
        figure.suptitle(f"latsize {lattice}, beta={beta}")
        plt.show(block=False)

    energies: list[float] = []
    effective_sample_sizes: list[float] = []
    total_acceptance = 0.0
    progress = tqdm(total=iters - 1)
    volume = float(torch.tensor(lattice).prod().item())

    for iteration in range(1, iters):
        energy = lattice_action.energy(configurations[iteration - 1]).real / volume
        energies.append(energy.item())
        post_burn = energies[burn:]
        if iteration <= burn or len(post_burn) < 2:
            effective_sample_sizes.append(float("nan"))
        else:
            effective_sample_sizes.append((iteration - burn) / IACT(post_burn))

        if block:
            updated, acceptance = metropolis_block(
                configurations[iteration - 1],
                lattice_action,
                N,
                block_size,
            )
        elif evenodd:
            updated, acceptance = metropolis_even_odd(
                configurations[iteration - 1], lattice_action, N
            )
        else:
            updated, acceptance = metropolis(
                configurations[iteration - 1], lattice_action, N
            )
        configurations[iteration] = updated
        total_acceptance += acceptance
        progress.set_description(f"A rate={100 * total_acceptance / iteration:.1f}%")

        if plot:
            plot_energies = energies[-50:]
            plot_ess = effective_sample_sizes[-50:]
            x_values = list(range(iteration - len(plot_energies) + 1, iteration + 1))
            energy_line.set_data(x_values, plot_energies)
            ess_line.set_data(x_values, plot_ess)
            if iteration > burn + 1:
                sample = torch.tensor(energies[burn::5])
                energy_average.remove()
                energy_average = axes[0].errorbar(
                    [x_values[0]],
                    [sample.mean()],
                    yerr=[sample.std()],
                    fmt="r^",
                    capsize=4,
                )
            for axis in axes:
                axis.relim()
                axis.autoscale_view()
            figure.canvas.draw()
            figure.canvas.flush_events()
        progress.update(1)

    progress.close()
    if plot:
        plt.ioff()
        plt.close("all")
    print(f"acceptance rate: {100 * total_acceptance / (iters - 1):.1f}%")

    if save:
        output_name = str(fname) or (
            f"data/CP{N}_lat{'x'.join(map(str, lattice))}_b{beta:.1f}.h5"
        )
        with h5py.File(output_name, "a") as output:
            if "configs" in output:
                del output["configs"]
            group = output.create_group("configs")
            group.create_dataset("vectors", data=configurations.numpy())
            info = group.create_group("info")
            info.create_dataset("iterations", data=[iters])
            info.create_dataset("burn", data=[burn])
            info.create_dataset("stepsize", data=[stepsize])
        print(f"CPN configs saved to {output_name}")
        return None
    return configurations


def autocorrFFT(energies: Sequence[float]) -> torch.Tensor:
    """Compute a normalized autocorrelation using an FFT convolution."""
    if len(energies) == 0:
        raise ValueError("energies must not be empty")
    values = torch.as_tensor(energies, dtype=torch.float64)
    values = values - values.mean()
    if torch.all(values == 0):
        return torch.ones_like(values)
    fft_size = 1 << (2 * len(values) - 1).bit_length()
    transform = torch.fft.rfft(values, n=fft_size)
    autocorrelation = torch.fft.irfft(transform * torch.conj(transform), n=fft_size)[
        : len(values)
    ]
    return autocorrelation / autocorrelation[0]


def IACT(energies: Sequence[float], cutoff: int | None = None) -> float:
    """Estimate the integrated autocorrelation time of an energy series."""
    autocorrelation = autocorrFFT(energies)
    if cutoff is None:
        negative = (autocorrelation < 0).nonzero(as_tuple=True)[0]
        cutoff = int(negative[0].item()) if len(negative) else len(autocorrelation)
    tau = 1 + 2 * torch.sum(autocorrelation[1:cutoff]).item()
    return max(tau, np.finfo(float).eps)


def build_parser() -> argparse.ArgumentParser:
    """Build the CPN MCMC command-line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--latsize", type=parse_int_tuple, default=(64, 64))
    parser.add_argument("--start", choices=("hot", "cold"), default="cold")
    parser.add_argument("--iters", type=int, default=10_000)
    parser.add_argument("--burn", type=int, default=1_000)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument(
        "--evenodd", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--block", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--blocksize", type=parse_int_tuple, default=(2, 2))
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    """Run the CPN MCMC command-line entry point."""
    args = build_parser().parse_args()
    MCMC(
        iters=args.iters,
        burn=args.burn,
        latsize=args.latsize,
        beta=args.beta,
        N=args.N,
        save=args.save,
        start=args.start,
        evenodd=args.evenodd,
        block=args.block,
        block_size=args.blocksize,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
