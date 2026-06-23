import torch
import matplotlib.pyplot as plt

N_epochs = 200
fig, ax = plt.subplots(1, 1)
tmin, tmax = 1, 33
trange = torch.arange(tmin, tmax)
data = torch.load(f"data/training_data_i0_j1_Nepochs{N_epochs}.pt",
                  weights_only=False)[:, tmin:tmax, 0]
rat_analytic = data[0]
rat_learnt = data[1]

ax.plot(trange, rat_analytic, "bo-", label="al/og")
ax.plot(trange, rat_learnt, "ro-", label="ml/og")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"ratio of $Var\left[O(8,8)O^\dagger(8+t,8)\right]$")
ax.set_title(r"analytic vs training gain, 200 epochs")
ax.legend()
plt.show()
