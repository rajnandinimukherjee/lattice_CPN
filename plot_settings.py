#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set some plot parameters

Created on Mon Aug 16 20:41:27 2021

@author: jflynn
"""
from __future__ import annotations

from math import sqrt

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
