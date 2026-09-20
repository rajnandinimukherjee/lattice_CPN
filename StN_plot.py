from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def main() -> None:
    """Plot analytic and learned variance gains from saved training data."""
    epochs = 200
    figure, axis = plt.subplots(1, 1)
    minimum, maximum = 1, 33
    separations = torch.arange(minimum, maximum)
    data = torch.load(
        f"data/training_data_i0_j1_Nepochs{epochs}.pt",
        weights_only=False,
    )[:, minimum:maximum, 0]
    analytic_ratio = data[0]
    learned_ratio = data[1]

    axis.plot(separations, analytic_ratio, "bo-", label="al/og")
    axis.plot(separations, learned_ratio, "ro-", label="ml/og")
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"ratio of $Var\left[O(8,8)O^\dagger(8+t,8)\right]$")
    axis.set_title(r"analytic vs training gain, 200 epochs")
    axis.legend()
    plt.show()


if __name__ == "__main__":
    main()
