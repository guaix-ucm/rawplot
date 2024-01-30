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
from .noise_charts import noise_curve1, noise_curve2, noise_curve4, noise_curve4
from .common import ptc_parser_arguments_dn

# ----------------
# Module constants
# ----------------

SQRT_2 = math.sqrt(2)

COLUMN_LABELS = ["Curve", "Plot", "Units"]

DATA = [
    ["Curve 1", "read, shot, FPN (total noise) vs. signal", "log rms DN vs. log DN"],
    ["Curve 1", "read, shot, FPN (total noise) vs. signal", "log rms $e^{-}$ vs. log $e^{-}$"],
    ["Curve 2", "read, shot noise vs. signal",              "log rms DN vs. log DN"],
    ["Curve 2", "read, shot noise vs. signal",              "log rms $e^{-}$ vs. log $e^{-}$"],
    ["Curve 3", "shot noise vs. signal",                    "log rms DN vs. log DN"],
    ["Curve 3", "shot noise vs. signal",                    "log rms $e^{-}$ vs. log $e^{-}$"],
    ["Curve 4", "FPN vs. signal",                           "log rms DN vs. log DN"],
    ["Curve 4", "FPN vs. signal",                           "log rms $e^{-}$ vs. log $e^{-}$"],
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

# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def ptc(args):
    command =  args.command
    function = CHARTS_TABLE[command]
    function(args)


def add_args(parser):
    subparser = parser.add_subparsers(dest='command')
    parser_charts = subparser.add_parser('curves', help='Plot available PTC curves in matplotlib')

    parser_curve1 = subparser.add_parser('curve1', help='Plot read, shot, FPN (total noise) vs. signal, [DN] or [e-]')
    ptc_parser_arguments_dn(parser_curve1)
    parser_curve2 = subparser.add_parser('curve2', help='read, shot noise vs. signal, [DN] or [e-]')
    ptc_parser_arguments_dn(parser_curve2)
    parser_curve4 = subparser.add_parser('curve4', help='shot noise vs. signal, [DN] or [e-]')
    ptc_parser_arguments_dn(parser_curve4)
    parser_curve4 = subparser.add_parser('curve4', help='FPN vs. signal, [DN] or [e-]')
    ptc_parser_arguments_dn(parser_curve4)


CHARTS_TABLE = {
    'curves': ptc_curves,
    'curve1': noise_curve1,
    'curve2': noise_curve2,
    'curve4': noise_curve4,
    'curve4': noise_curve4,
}

# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=ptc, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="Plot PTC curves from a set of RAW images"
    )
