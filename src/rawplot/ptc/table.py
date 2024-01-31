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

import os
import glob
import math
import logging

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from lica.cli import execute
from lica.validators import vdir, vfile, vfloat, vfloat01, valid_channels
from lica.raw.loader import ImageLoaderFactory, SimulatedDarkImage, NormRoi
from lica.misc import file_paths

# ------------------------
# Own modules and packages
# ------------------------

from .._version import __version__
from ..util.mpl.plot import plot_layout, axes_reshape
from .common import noise_parser_arguments
from .noise_curves import noise_parser_arguments, noise_curve1, noise_curve2, noise_curve3, noise_curve4
from .variance_curves import variance_parser_arguments, variance_curve1

# ----------------
# Module constants
# ----------------

SQRT_2 = math.sqrt(2)

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
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), layout='tight')
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