"""Shared configuration for the CP(N) OPF scaling study."""

from pathlib import Path

SPLIT = 0.8
BEST_WINDOW = 20

DATA_DIRECTORY = Path("data")
TRAINING_DATA_DIRECTORY = Path("training_data")

LOSS_FUNCTIONS = ("variance", "penalty", "approx2")
LOSS_LABELS = {
    "variance": r"direct variance",
    "penalty": r"Approx. 1 (weak corr.)",
    "approx2": r"Approx. 2 (small fluc.)",
}
LOSS_MARKERS = {
    "variance": "o",
    "penalty": "s",
    "approx2": "^",
}

# The d=3 ferromagnetic CP(2) gauge model has a weak first-order transition.
# Pelissetto and Vicari (arXiv:1905.03307) quote beta_c=0.6195 for a
# Hamiltonian with an explicit factor N=3. Our action uses -2*beta*Re(...)
# without that factor, so beta_c(ours)=3*0.6195 and T_c=1/beta_c(ours).
# In d=1 and d=2 there is no finite-temperature ordered phase.
GAUGE_CRITICAL_TEMPERATURES = {
    1: 0.0,
    2: 0.0,
    3: 1.0 / (3.0 * 0.6195),
}

