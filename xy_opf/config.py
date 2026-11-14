"""Shared constants for the XY one-point-function study."""

from pathlib import Path

SPLIT = 0.8
BEST_WINDOW = 20

DATA_DIRECTORY = Path("data")
TRAINING_DATA_DIRECTORY = Path("training_data")

LOSS_FUNCTIONS = ("variance", "penalty", "saddle-point", "approx3")
LOSS_FUNCTION_CHOICES = (*LOSS_FUNCTIONS, "field-dependent")
LOSS_LABELS = {
    "variance": r"trained on $\mathrm{Var}(Q)$",
    "field-dependent": (
        r"field-dependent, trained on $\mathrm{Var}(Q)$"
    ),
    "penalty": (
        r"trained on $\langle e^{-2\,\mathrm{Re}\,\Delta S}\rangle$; "
        r"scaled by optimized $\alpha$"
    ),
    "saddle-point": (
        r"SaddlePoint $f_x$; scaled by optimized $\alpha$"
    ),
    "approx3": r"Approximation 3 (smooth deformation)",
}
LOSS_MARKERS = {
    "variance": "o",
    "field-dependent": "D",
    "penalty": "s",
    "saddle-point": "^",
    "approx3": "D",
}
XY_CRITICAL_TEMPERATURES = {
    1: 0.0,
    2: 1.0 / 1.1199,
    3: 1.0 / 0.454165,
}
