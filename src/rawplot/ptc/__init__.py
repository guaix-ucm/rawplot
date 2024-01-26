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
from lica.validators import vdir, vfloat01, valid_channels
from lica.raw import ImageLoaderFactory, SimulatedDarkImage, NormRoi
from lica.misc import file_paths

# ------------------------
# Own modules and packages
# ------------------------

from .._version import __version__
from ..util.mpl.plot import plot_layout, axes_reshape

# ----------------
# Module constants
# ----------------

SQRT_2 = math.sqrt(2)

COLUMN_LABELS = ["Chart", "Plot", "Units"]

DATA = [
    ["Chart 1", "read, shot, FPN (total noise) vs. signal", "log rms DN vs. log DN"],
    ["Chart 2", "read, shot noise vs. signal",               "log rms DN vs. log DN"],
    ["Chart 3", "shot noise vs. signal",                    "log rms DN vs. log DN"],
    ["Chart 4", "FPN vs. signal",                           "log rms DN vs. log DN"],
    ["Chart 5", "read, shot, FPN (total noise) vs. signal", "log rms $e^{-}$ vs. log $e^{-}$"],
    ["Chart 6", "ead, shot noise vs. signal",               "log rms $e^{-}$ vs. log $e^{-}$"],
    ["Chart 7", "shot noise vs. signal",                    "log rms $e^{-}$ vs. log $e^{-}$"],
    ["Chart 8", "FPN vs. signal",                           "log rms $e^{-}$ vs. log $e^{-}$"],
]

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# ------------------
# Auxiliary fnctions
# ------------------

def ptc_charts(args):
    log.info("Displaying PTC charts")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), layout='tight')
    fig.suptitle("Available Photon Transfer Charts")
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
    table.set_url('https://www.google.com/')
    plt.show()

# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def ptc(args):
    command =  args.command
    if  command == 'charts':
        ptc_charts(args)



def add_args(parser):
    subparser = parser.add_subparsers(dest='command')
    parser_charts = subparser.add_parser('charts', help='Display avaliable PTC charts in matplotlib')


# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=ptc, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="Plot Sensor SNR per channel over a numbr of flat fields"
    )
