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
from .chart1 import ptc_chart1

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
    function = CHARTS_TABLE[command]
    function(args)


def add_args(parser):
    subparser = parser.add_subparsers(dest='command')
    parser_charts = subparser.add_parser('charts', help='Plot avaliable PTC charts in matplotlib')

    parser_chart1 = subparser.add_parser('chart1', help='Plot read, shot, FPN (total noise) vs. signal')
    parser_chart1.add_argument('-i', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser_chart1.add_argument('-f', '--image-filter', type=str, required=True, help='Images filter, glob-style (i.e. flat*, dark*)')
    parser_chart1.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser_chart1.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser_chart1.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser_chart1.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser_chart1.add_argument('-rd','--rd-noise', type=vfloat, metavar='<\u03C3>', default=1.0, help='Readout noise [DN] (default: %(default)s)')
    parser_chart1.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser_chart1.add_argument('--every', type=int, metavar='<N>', default=1, help='pick every n `file after sorting')
    group0 = parser_chart1.add_mutually_exclusive_group(required=False)
    group0.add_argument('-bl', '--bias-level',  type=vfloat, default=None, help='Bias level, common for all channels (default: %(default)s)')
    group0.add_argument('-bf', '--bias-file',  type=vfile, default=None, help='Bias image (3D FITS cube) (default: %(default)s)')




CHARTS_TABLE = {
    'charts': ptc_charts,
    'chart1': ptc_chart1,
}

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
