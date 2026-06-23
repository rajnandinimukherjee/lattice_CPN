import matplotlib.pyplot as plt
import torch
import pdb
import h5py

import actions
import observables
from argparsing import parseargs
from deformations import HomogenousDeformations
from losses import VarianceLoss


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
            "data/cpn_b4.0_L64_Nc3_ens.dat",
            shared=False,
            size=1000 * 64 * 64 * 3,
            dtype=torch.complex128,
        )
        .reshape(1000, 64, 64, 3)[50:]
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
    model = HomogenousDeformations(N)
    model.samples = {"all": samples,
                     "train": samples[mask], "test": samples[~mask]}

    fname = f"data/training_params_i{args.i}_j{args.j}_Nepochs{args.epochs}.pt"
    alphanet = torch.load(fname, weights_only=False)

    trange = range(int(args.latsize[0]/2))
    tpf_exp = torch.zeros(len(trange))
    tpf_var = torch.zeros(len(trange))
    for t in range(trange):
        delta = (0, t)
        OPF1 = observables.OnePtFn(
            N=N, latsize=args.latsize,
            i=args.i, j=args.j,
            coord=(8, 8),
            action=my_action,
        )
        OPF2 = observables.OnePtFn(
            N=N, latsize=args.latsize,
            i=args.i, j=args.j,
            coord=(8+t, 8),
            action=my_action,
        )
