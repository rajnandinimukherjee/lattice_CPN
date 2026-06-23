import h5py
import pdb
import torch
from CPN_utils import *
from tqdm import tqdm
from actions import LatticeDdimAction, LatticeEnergy
import matplotlib.pyplot as plt
import argparse
import ast
import numpy


def metropolis(old_config, action, N, **actionkwargs):
    """ metropolis accept/reject with one-site-at-a-time updates """

    latsize = old_config.shape[:-1]

    rand_rotations = randSUN(N+1, size=latsize)
    new_config = old_config.clone()
    sites = torch.cartesian_prod(*[torch.arange(dim) for dim in latsize])
    N_sites = len(sites)

    # do a sweep of metropolis accept/rejects for every site
    accepts = 0
    for index in sites:
        idx = tuple(index.tolist())
        S_old = action(new_config, **actionkwargs)

        # Propose new config
        temp = new_config[idx]
        new_config[idx] = rand_rotations[idx]@temp
        S_new = action(new_config, **actionkwargs)

        # Compute action difference
        deltaS = (S_new - S_old).real
        accept_prob = min(1.0, torch.exp(-deltaS))

        if torch.rand(1).item() < accept_prob:
            accepts += 1
        else:
            new_config[idx] = temp

    return new_config, accepts/N_sites


def checkerboard_sites(latsize):
    """Return even and odd site indices for given lattice size """
    coords = torch.cartesian_prod(*[torch.arange(L) for L in latsize])
    parities = coords.sum(-1) % 2
    even_sites = coords[parities == 0]
    odd_sites = coords[parities == 1]
    return even_sites, odd_sites


def metropolis_even_odd(config, action, N, **actionkwargs):
    """ Metropolis accept/reject with batched updates of even and odd sites """
    latsize = config.shape[:-1]
    even_sites, odd_sites = checkerboard_sites(latsize)

    rand_rotations = small_randSUN(N+1, size=latsize, eps=0.3)
    accepts = 0
    for sites in [even_sites, odd_sites]:
        old = config[tuple(sites.T)]
        S_old = action(config, local=True, sites=sites, **actionkwargs)

        new = torch.einsum('mij,mj->mi', rand_rotations[tuple(sites.T)], old)
        config[tuple(sites.T)] = new
        S_new = action(config, local=True, sites=sites, **actionkwargs)

        deltaS = (S_new - S_old).real
        accept_prob = torch.exp(-deltaS).clamp(max=1.0)
        rand = torch.rand_like(accept_prob)
        accept = rand < accept_prob

        final = torch.where(accept[:, None], new, old)
        config[tuple(sites.T)] = final
        accepts += accept.float().mean().item()

    return config, accepts/2


def metropolis_block(old_config, action, N, block_size, **actionkwargs):
    """ Metropolis accept/reject with block updates """
    latsize = old_config.shape[:-1]
    ndims = len(latsize)
    new_config = old_config.clone()

    # Total number of blocks in each dimension
    num_blocks = [latsize[d] // block_size[d] for d in range(ndims)]
    blocks = torch.cartesian_prod(*[torch.arange(dim) for dim in num_blocks])
    N_blocks = len(blocks)

    accepts = 0
    # Loop over all block indices
    for block_idx in blocks:
        # Compute site ranges in each dimension for the current block
        slices = [slice(bi * bs, (bi + 1) * bs)
                  for bi, bs in zip(block_idx, block_size)]

        # Create an n-dimensional index for the block
        block_slice = tuple(slices)

        # Generate a single SU(N+1) random rotation
        R = randSUN(N + 1)

        # Propose new config
        temp_config = new_config.clone()
        temp_config[block_slice] = torch.einsum(
            'ij,...j->...i', R, new_config[block_slice])

        # Compute action difference
        deltaS = action(temp_config, **actionkwargs) - \
            action(old_config, **actionkwargs)
        accept_prob = min(1.0, torch.exp(-deltaS).real)

        if torch.rand(1).item() < accept_prob:
            new_config[block_slice] = temp_config[block_slice]
            accepts += 1

    return new_config, accepts/N_blocks


def MCMC(latsize, iters=10000, burn=1000, stepsize=1,
         N=3, save=False, fname="", beta=4.0, block=True,
         block_size=None, start="hot", evenodd=False, **kwargs):

    configs = torch.zeros(size=(iters, *latsize, N+1), dtype=torch.cdouble)

    configs[0] = oneCPN(
        N, size=latsize) if start == "cold" else randCPN(N, size=latsize)

    accepts = 0
    if block:
        update = metropolis_block
    elif evenodd:
        update = metropolis_even_odd
    else:
        update = metropolis

    if block:
        ndims = len(latsize)
        if block_size is None:
            block_size = tuple(2 for _ in range(ndims))
        assert len(block_size) == ndims
        print(f"doing block updates with blocks of size {
              'x'.join([str(n) for n in block_size])}")

    plt.ion()  # interactive mode
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True,
                           gridspec_kw={'hspace': 0})
    EV, = ax[0].plot([], [], "bo-")
    EV_avg = ax[0].errorbar([], [], yerr=[], fmt='r^')
    ESS, = ax[1].plot([], [], "ro-")
    ax[1].set_xlabel("Iteration")
    ax[0].set_ylabel("E/V")
    ax[1].set_ylabel("ESS")
    fig.suptitle(f"latsize {latsize}, beta={beta}")

    energies = []
    eff_samp_size = []
    pbar = tqdm(range(1, iters))
    plt.show(block=False)

    for idx in range(1, iters):
        energy = LatticeEnergy(configs[idx-1]).real / \
            torch.prod(torch.tensor(latsize))
        energies.append(energy.item())
        if idx <= burn:
            eff_samp_size.append(numpy.nan)
        else:
            eff_samp_size.append((idx-burn)/IACT(energies[burn:]))

        configs[idx], accept = update(
            configs[idx-1],
            N=N,
            beta=beta,
            block_size=block_size, **kwargs)
        accepts += accept
        pbar.set_description(f"A rate={round(accepts*100/idx, 1)}%")

        # Update plot data
        # Only keep the last 50 points for plotting
        if len(energies) > 50:
            ev_y_plot = energies[-50:]
            ESS_y_plot = eff_samp_size[-50:]
            x_plot = list(range(idx - 49, idx + 1))
        else:
            ev_y_plot = energies
            ESS_y_plot = eff_samp_size
            x_plot = list(range(len(energies)))

        EV.set_ydata(ev_y_plot)
        EV.set_xdata(x_plot)

        if idx > burn+1:
            e = torch.tensor(energies[burn::5])
            EV_avg.remove()
            EV_avg = ax[0].errorbar([x_plot[0]], [e.mean()],
                                    yerr=[e.var()**0.5], fmt='r^',
                                    capsize=4)
        ESS.set_ydata(ESS_y_plot)
        ESS.set_xdata(x_plot)

        for idx in range(2):
            ax[idx].relim()
            ax[idx].autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()

        pbar.update(1)

    pbar.close()
    plt.ioff()
    plt.close('all')

    print(f'acceptance rate: {
          accepts}/{iters}={round(accepts*100/iters, 1)}%')
    if save:
        if fname == "":
            fname = f"data/CP{N}_lat{
                'x'.join([str(n) for n in latsize])}_b{
                beta:.1f}.h5"
        file = h5py.File(fname, 'a')
        if "configs" in file:
            del file["configs"]
        grp = file.create_group("configs")
        grp.create_dataset("vectors", data=configs.numpy())
        info = grp.create_group("info")
        info.create_dataset("iterations", data=[iters])
        info.create_dataset("burn", data=[burn])
        info.create_dataset("stepsize", data=[stepsize])
        file.close()
        print(f"CPN configs saved to {fname}")
    else:
        return configs


def autocorrFFT(energies):
    n = len(energies)
    e = torch.tensor(energies)
    e = e - e.mean()

    n_fft = 1 << (2 * n - 1).bit_length()
    f = torch.fft.rfft(e, n=n_fft)
    acf = torch.fft.irfft(f * torch.conj(f), n=n_fft)[:n]
    return acf/acf[0]


def IACT(energies, cutoff=None):
    acf = autocorrFFT(energies)
    if cutoff is None:
        # find first negative lag
        try:
            cutoff = (acf < 0).nonzero(as_tuple=True)[0][0].item()
        except IndexError:
            cutoff = len(acf)

    # IACT
    tau_int = 1 + 2 * torch.sum(acf[1:cutoff]).item()
    return tau_int


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="CP(N) dimension", type=int, default=3)
    parser.add_argument("--latsize", help='lattice dimensions, eg "(64,64)"',
                        type=lambda s: tuple(ast.literal_eval(s)), default=(64, 64))
    parser.add_argument("--start", help="MCMC initial config start (hot|cold)",
                        type=str, default="cold")
    parser.add_argument("--iters", help="MCMC iterations",
                        type=int, default=10000)
    parser.add_argument("--save", help="option to save data (True|False)",
                        type=bool, default=True)
    parser.add_argument(
        "--beta", help="LatticeDdimAction parameter beta", type=float, default=4.0)
    parser.add_argument("--evenodd", help="option to use checkerboard metropolis",
                        type=bool, default=False)
    parser.add_argument("--block", help="option to block metropolis (True|False)",
                        type=bool, default=False)
    parser.add_argument("--blocksize", help='blocking size, eg "(2,2)"',
                        type=lambda s: tuple(ast.literal_eval(s)), default=(2, 2))

    args = parser.parse_args()
    assert args.start in ["hot", "cold"], "start must either be hot or cold"

    MCMC(
        iters=args.iters,
        latsize=args.latsize,
        action=LatticeDdimAction,
        beta=args.beta,
        N=args.N,
        save=args.save,
        start=args.start,
        evenodd=args.evenodd,
        block=args.block,
        block_size=args.blocksize)
