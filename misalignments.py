import torch
import os
import matplotlib.pyplot as plt

import observables
import actions
import deformations

from plot_settings import plotparams
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update(plotparams)


def call_PDF(filename, open=True, **kwargs):
    pdf = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pdf, format="pdf")
    pdf.close()
    plt.close("all")
    if open:
        os.system("open " + filename)


N, beta = 2, 4.0
samples = torch.from_file("data/cpn_b4.0_L64_Nc3_ens.dat",
                          shared=False,
                          size=1000*64*64*3,
                          dtype=torch.complex128).reshape(1000, 64, 64, 3)[50:]
myModel = deformations.HomogenousDeformations(N=N)
myModel.samples = {"all": samples}
myAction = actions.LatticeAction(beta=beta)

fig, ax = plt.subplots()
trange = torch.arange(32)
undefs = torch.zeros(size=(3, 32))
defs = torch.zeros(size=(3, 32))

for t in trange:
    Obs = observables.TwoPtFn(N=2, latsize=(
        64, 64), i=1, j=0, delta=(t, 0), action=myAction)
    ud_O = Obs(myModel)
    undefs[0, t] = torch.abs(torch.mean(ud_O)).item()**2
    undefs[1, t] = torch.mean(torch.abs(ud_O)).item()**2
    undefs[2, t] = torch.mean(torch.abs(ud_O)**2).item()

    d_O = Obs(myModel, deform=True, alphas=Obs.invSqAlphas)
    defs[0, t] = torch.abs(torch.mean(d_O)).item()**2
    defs[1, t] = torch.mean(torch.abs(d_O)).item()**2
    defs[2, t] = torch.mean(torch.abs(d_O)**2).item()

labels = [r"$|\langle O(t)\rangle|^2$",
          r"$\langle |O(t)|\rangle^2$",
          r"$\langle |O(t)|^2\rangle$"]

colors = ["r", "b", "g"]

for idx in range(3):
    ax.plot(trange, undefs[idx, :], colors[idx]+"o-", label=labels[idx])
    ax.plot(trange, defs[idx, :], colors[idx]+"x-")

ax.legend()
ax.set_xlabel(r"$t$")
ax.set_title(labels[0]+r"$\leq$"+labels[1]+r"$\leq$"+labels[2])

fig2, ax2 = plt.subplots()
ax2.plot(trange, undefs[1]-undefs[0], colors[0]+"o-", label="PM")
ax2.plot(trange, undefs[2]-undefs[1], colors[1]+"o-", label="MM")
ax2.plot(trange, undefs[2]-undefs[0], colors[2]+"o-", label="var")

ax2.plot(trange, defs[1]-defs[0], colors[0]+"x-")
ax2.plot(trange, defs[2]-defs[1], colors[1]+"x-")
ax2.plot(trange, defs[2]-defs[0], colors[2]+"x-")
ax2.legend()
ax2.set_xlabel(r"$t$")
ax2.set_title(r"PM="+labels[1]+r"-"+labels[0] +
              r", MM="+labels[2]+r"-"+labels[1] +
              r", $\mathrm{Var}(O)=$PM+MM")
filename = "plots/cpn_b4.0_L64_Nc3_variance.pdf"
call_PDF(filename, open=True)
