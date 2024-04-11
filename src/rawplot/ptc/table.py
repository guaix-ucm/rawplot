# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2021
#
# See the LICENSE file for details
# see the AUTHORS file for authors
# ----------------------------------------------------------------------

#--------------------
# System wide imports
# -------------------

import logging

# ---------------------
# Thrid-party libraries
# ---------------------

import matplotlib.pyplot as plt

from lica.cli import execute

# ------------------------
# Own modules and packages
# ------------------------

from .._version import __version__

# ----------------
# Module constants
# ----------------

COLUMN_LABELS = ["Curve", "Plot", "Units"]

DATA = [
    ["Curve 1", "read, shot, FPN (total noise) vs. signal", "log rms DN vs. log DN"],
    ["Curve 1", "read, shot, FPN (total noise) vs. signal", "log rms $e^{-}$ vs. log $e^{-}$"],
    ["Curve 2", "read + shot noise vs. signal",             "log rms DN vs. log DN"],
    ["Curve 2", "read + shot noise vs. signal",             "log rms $e^{-}$ vs. log $e^{-}$"],
    ["Curve 3", "shot noise vs. signal",                    "log rms DN vs. log DN"],
    ["Curve 3", "shot noise vs. signal",                    "log rms $e^{-}$ vs. log $e^{-}$"],
    ["Curve 4", "FPN vs. signal",                           "log rms DN vs. log DN"],
    ["Curve 4", "FPN vs. signal",                           "log rms $e^{-}$ vs. log $e^{-}$"],
    ["Curve 5", "read + shot noise variance vs. signal",    "DN vs. DN"],
    ["Curve 6", "SNR vs. signal",                           "log SNR vs. log DN"],
    ["Curve 6", "SNR vs. signal",                           "log SNR vs. log $e^{-}$"],
]


# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# ------------------
# Auxiliary fnctions
# ------------------

def ptc_curves(args):
    log.info("Displaying PTC charts")
    fig, ax = plt.subplots(nrows=1, ncols=1,  figsize=(7, 3))
    fig.suptitle("Available Photon Transfer Curves")
    ax.axis("tight")
    ax.axis("off")
    ax.set_url('https://www.google.com/')
    table = ax.table(
        cellText=DATA, 
        colLabels=COLUMN_LABELS, 
        colWidths=(1/6, 3/6, 2/6),
        colLoc="center",
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    #table.set_fontsize(11)
    plt.show()
