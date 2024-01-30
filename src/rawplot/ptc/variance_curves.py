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

from lica.validators import vdir, vfile, vfloat, vfloat01, valid_channels
from lica.raw.loader import ImageLoaderFactory, SimulatedDarkImage, NormRoi

# ------------------------
# Own modules and packages
# ------------------------

from .._version import __version__
from ..util.mpl.plot import mpl_main_plot_loop
from ..util.common import common_list_info, bias_from, make_plot_title_from, check_physical
from .common import signal_and_noise_variances_from, signal_and_noise_variances
# ----------------
# Module constants
# ----------------


# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------
def variance_parser_arguments(parser):
    parser.add_argument('-i', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser.add_argument('-f', '--image-filter', type=str, required=True, help='Images filter, glob-style (i.e. flat*, dark*)')
    parser.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser.add_argument('-rd','--read-noise', type=vfloat, metavar='<\u03C3>', default=0.0, help='Read noise [DN] (default: %(default)s)')
    parser.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser.add_argument('--every', type=int, metavar='<N>', default=1, help='pick every n `file after sorting')
    group0 = parser.add_mutually_exclusive_group(required=False)
    group0.add_argument('-bl', '--bias-level', type=vfloat, default=None, help='Bias level, common for all channels (default: %(default)s)')
    group0.add_argument('-bf', '--bias-file',  type=vfile, default=None, help='Bias image (3D FITS cube) (default: %(default)s)')
    parser.add_argument('-gn','--gain', type=vfloat, metavar='<g>', default=None, help='Gain [e-/DN] (default: %(default)s)')
   


def plot_read_noise_variance_line(axes, read_noise):
    '''Plot an horizontal line'''
    text = r"$\sigma_{READ}^2$"
    axes.axhline(read_noise**2, linestyle='-', label=text)

def plot_variance_vs_signal(axes, i, x, y, xtitle, ytitle, ylabel, channels, **kwargs):
    '''For Charts 1 to 8'''
    # Main plot goes here
    axes.plot(x[i], y[i], marker='o', linewidth=0, label=ylabel)
    # Additional plots go here
    base = 2 if kwargs.get('log2', False) else 10
    for key, value in kwargs.items():
        if key in ('total',) :
            label = r"$\sigma_{TOTAL}^2$"
            axes.plot(x[i], value[i], marker='o', linewidth=0, label=label)
        elif key == 'read' and value is not None:
            plot_read_noise_variance_line(axes, value) #  read noise is a scalar
    axes.set_title(f'channel {channels[i]}')
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.set_xlabel(xtitle)
    axes.set_ylabel(ytitle)
    if ylabel:
        axes.legend()


def variance_curve1(args):
    log.info(" === VARIANCE CHART 1: Shot + Readout Noise vs. Signal === ")
    units = "[DN]"
    gain = args.gain
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    read_noise = args.read_noise
    signal, total_var, shot_read_var, shot_var, fpn_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = bias, 
        read_noise = read_noise
    )
    title = make_plot_title_from(r"$\sigma_{READ+SHOT}^2$ vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
        plot_func = plot_variance_vs_signal,
        xtitle = f"Signal {units}",
        ytitle = f"Noise Variance {units}",
        ylabel =r"$\sigma_{READ+SHOT}^2$",
        x    = signal,
        y  = shot_read_var,
        channels = channels,
        # Optional arguments
        read = read_noise,
        total_var = total_var,
    )


