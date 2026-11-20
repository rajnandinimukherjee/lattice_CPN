#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set some plot parameters

Created on Mon Aug 16 20:41:27 2021

@author: jflynn
"""
from __future__ import annotations

from math import sqrt

# Paper-wide semantic colors.  The four objective colors follow the
# color-vision-friendly Okabe--Ito palette; gray denotes the undeformed
# estimator and the variance-ratio-one reference throughout the draft.
UNDEFORMED_COLOR = "#4A4A4A"
OBJECTIVE_COLORS = {
    "variance": "#0072B2",       # direct variance
    "penalty": "#E69F00",        # Approximation 1
    "approx2": "#009E73",        # Approximation 2
    "saddle-point": "#009E73",   # alternate key for Approximation 2
    "approx3": "#CC79A7",        # Approximation 3
    "lattice-laplace": "#CC79A7", # alternate key for Approximation 3
}

# Figure 2 uses one red family for all three deformed estimates.
TWO_POINT_COLORS = {
    "glued": "#8F302D",
    "deformed": "#BC554C",
    "one_point_squared": "#D98276",
}

# Figure 7's auxiliary amplitude axis uses a separate olive tone.
TRAINING_AMPLITUDE_COLOR = "#79712B"

fig_width_pt = 455.24411  # Got this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (sqrt(5) + 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width / golden_mean  # height in inches
fig_size = [fig_width, fig_height]
plotparams = {
    "backend": "ps",
    "font.size": 10,
    # Match LaTeX's default Computer Modern Roman typeface for ordinary text
    # and Matplotlib's built-in Computer Modern math renderer.  This does not
    # require a LaTeX installation at plot time.
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    # Classic Computer Modern text fonts lack the Unicode minus sign used by
    # Matplotlib tick labels. Use the supported ASCII hyphen instead.
    "axes.unicode_minus": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.labelsize": 10,
    "figure.figsize": fig_size,
    "savefig.bbox": "tight",
    "legend.numpoints": 1,
    "legend.frameon": True,
    "legend.handletextpad": 0.3,
    "legend.labelspacing": 0.25,
}

# Plot modules may apply this subset after importing pyplot.  In particular,
# do not include ``backend`` here: backend selection must happen before pyplot
# is imported and is unrelated to the plot typography.
latex_font_params = {
    "font.family": plotparams["font.family"],
    "font.serif": plotparams["font.serif"],
    "mathtext.fontset": plotparams["mathtext.fontset"],
    "axes.formatter.use_mathtext": plotparams["axes.formatter.use_mathtext"],
    "axes.unicode_minus": plotparams["axes.unicode_minus"],
}
