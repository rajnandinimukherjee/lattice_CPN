from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from actions import CPnAction
from CPN_utils import CNtoR2N
from deformations import HomogenousDeformations
from observables import CPnTwoPtFn
from plot_settings import plotparams

plt.rcParams.update(plotparams)


def main() -> None:
    """Compare common decompositions of deformed CPN observable variance."""
    N, beta = 2, 4.0
    latsize = (64, 64)
    complex_samples = torch.from_file(
        "data/cpn_b4.0_L64_Nc3_ens.dat",
        shared=False,
        size=1000 * 64 * 64 * 3,
        dtype=torch.complex128,
    ).reshape(1000, 64, 64, 3)[50:]
    samples = CNtoR2N(complex_samples).to(torch.float32)
    model = HomogenousDeformations(N=N)
    model.samples = {"all": samples}
    action = CPnAction(beta=beta, N=N, latsize=latsize)

    figure, axis = plt.subplots()
    separations = torch.arange(32)
    undeformed = torch.zeros(size=(3, 32))
    deformed = torch.zeros(size=(3, 32))

    for separation in separations:
        observable = CPnTwoPtFn(
            N=N,
            latsize=latsize,
            i=1,
            j=0,
            delta=(int(separation), 0),
            action=action,
        )
        bare_values = observable(model)
        undeformed[0, separation] = torch.abs(bare_values.mean()).square()
        undeformed[1, separation] = bare_values.abs().mean().square()
        undeformed[2, separation] = bare_values.abs().square().mean()

        values = observable(model, deform=True, alphas=observable.invSqAlphas)
        deformed[0, separation] = torch.abs(values.mean()).square()
        deformed[1, separation] = values.abs().mean().square()
        deformed[2, separation] = values.abs().square().mean()

    labels = [
        r"$|\langle O(t)\rangle|^2$",
        r"$\langle |O(t)|\rangle^2$",
        r"$\langle |O(t)|^2\rangle$",
    ]
    colors = ["r", "b", "g"]
    for index in range(3):
        axis.plot(
            separations,
            undeformed[index],
            colors[index] + "o-",
            label=labels[index],
        )
        axis.plot(separations, deformed[index], colors[index] + "x-")
    axis.legend()
    axis.set_xlabel(r"$t$")
    axis.set_title(labels[0] + r"$\leq$" + labels[1] + r"$\leq$" + labels[2])

    figure2, axis2 = plt.subplots()
    axis2.plot(separations, undeformed[1] - undeformed[0], "ro-", label="PM")
    axis2.plot(separations, undeformed[2] - undeformed[1], "bo-", label="MM")
    axis2.plot(separations, undeformed[2] - undeformed[0], "go-", label="var")
    axis2.plot(separations, deformed[1] - deformed[0], "rx-")
    axis2.plot(separations, deformed[2] - deformed[1], "bx-")
    axis2.plot(separations, deformed[2] - deformed[0], "gx-")
    axis2.legend()
    axis2.set_xlabel(r"$t$")
    axis2.set_title(
        r"PM="
        + labels[1]
        + "-"
        + labels[0]
        + r", MM="
        + labels[2]
        + "-"
        + labels[1]
        + r", $\mathrm{Var}(O)=$PM+MM"
    )
    plt.show()


if __name__ == "__main__":
    main()
