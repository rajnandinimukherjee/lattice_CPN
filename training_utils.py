import pdb

import matplotlib.pyplot as plt
import torch


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


class TrainingPlotter:

    def __init__(self, nrows=1, ncols=1, figsize=(3, 6)):
        plt.ion()
        fig, ax = plt.subplots(nrows=nrows,
                               ncols=ncols,
                               sharex=True,
                               gridspec_kw={'hspace': 0},
                               figsize=figsize)
        self.data = {"train": {}, "test": {}, "uncat": {}}

    def add_data(self, dtype, name: str, axis=0,
                 color="k", ls='-', **kwargs):
        assert dtype in ["train", "test", "uncat"]
        color = {"train": "b", "test": "r", "uncat": color}[dtype]
        ls = {"train": "o-", "test": "o-", "uncat": ls}[dtype]
        plot_args = {"color": color, "ls": ls}
        plot_args.update(kwargs)

        self.data[dtype][name] = {"ax": axis,
                                  "plot_args": plot_args,
                                  "xdata": [],
                                  "ydata": []}
