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

from lica.validators import vdir, vfile, vfloat, vfloat01, vflopath, valid_channels
from lica.raw.loader import ImageLoaderFactory,  NormRoi

# ------------------------
# Own modules and packages
# ------------------------

from .._version import __version__
from ..util.mpl.plot import mpl_main_plot_loop
from ..util.common import common_list_info, make_plot_title_from, assert_physical, assert_range
from .common import signal_and_noise_variances, fit
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
    parser.add_argument('-x', '--x0', type=vfloat01,  help='Normalized ROI start point, x0 coordinate [0..1]')
    parser.add_argument('-y', '--y0', type=vfloat01,  help='Normalized ROI start point, y0 coordinate [0..1]')
    parser.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser.add_argument('-rd','--read-noise', type=vfloat, metavar='<\u03C3>',  help='Read noise [DN] (default: %(default)s)')
    parser.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser.add_argument('--every', type=int, metavar='<N>', default=1, help='pick every n `file after sorting')
    parser.add_argument('-bi', '--bias',  type=vflopath,  help='Bias, either a single value for all channels or else a 3D FITS cube file (default: %(default)s)')
    parser.add_argument('-dk', '--dark',  type=vfloat,  help='Dark count rate in DN/sec. (default: %(default)s)')
    parser.add_argument('--fit',  action='store_true', help='Fit Shot+Read noise line')
    parser.add_argument('-fr','--from', dest='from_value', type=vfloat, metavar='<x0>',  help='Lower signal limit to fit [DN] (default: %(default)s)')
    parser.add_argument('-to','--to', dest='to_value', type=vfloat, metavar='<x1>',  help='Upper signal limit to fit [DN] (default: %(default)s)')



def plot_fitted(axes, fitted):
    '''All graphical elements for a fitting line'''
    slope = fitted['slope']
    score = fitted['score']
    intercept = fitted['intercept']
    fitted_x = fitted['x']
    fitted_y = fitted['y']
    label = rf"$\sigma_{{READ+SHOT}}^2$ (model)"
    P0 = (0, intercept) 
    P1 = ( -intercept/slope, 0)
    axes.plot(fitted_x, fitted_y, marker='o', linewidth=0, label=r"$\sigma_{READ+SHOT}^2$ (fitted)")
    axes.axline(P0, slope=slope, linestyle=':', label=label)
    if intercept >= 0:
        text_b = rf"$\sigma_{{READ}} = {math.sqrt(intercept):0.2f}$ [DN]"
    else:
        text_b = rf"$\sigma_{{READ}} = ?$"
    text = "\n".join((fr"$r^2 = {score:.3f}$", rf"$g = {1/slope:0.2f}\quad [e^{{-}}/DN$]", text_b))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes.text(0.4, 0.95, text, transform=axes.transAxes, va='top', bbox=props)



def plot_variance_vs_signal(axes, i, x, y, xtitle, ytitle, ylabel, channels, **kwargs):
    '''For Charts 5'''
    # Main plot goes here (signal_and_read noise...)
    axes.plot(x[i], y[i], marker='o', linewidth=0, label=ylabel)
    # Additional plots go here
    shot_noise = kwargs.get('shot_var', None)
    if shot_noise is not None:
        label = r"$\sigma_{SHOT}^2$"
        axes.plot(x[i], shot_noise[i], marker='o', linewidth=0, label=label)
    fitted = kwargs.get('fitted', None)
    if fitted is not None:
        plot_fitted(axes, fitted[i])
    read_noise = kwargs.get('read', None)
    if read_noise is not None:
        label = r"$\sigma_{READ}^2$"
        axes.axhline(read_noise**2, linestyle='--', label=label)
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
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    read_noise = args.read_noise if args.read_noise is not None else 0.0
    signal, total_var, shot_and_read_var, fpn_var, shot_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = args.bias, 
        dark = args.dark,
        read_noise = read_noise
    )
    if args.fit:
        assert_range(args)
        fit_params = fit(signal, shot_and_read_var, args.from_value, args.to_value, channels)
    else:
        fit_params = None
    title = make_plot_title_from(r"$\sigma_{READ+SHOT}^2$ vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
        plot_func = plot_variance_vs_signal,
        xtitle = f"Signal [DN]",
        ytitle = f"Noise Variance [DN]",
        ylabel =r"$\sigma_{READ+SHOT}^2$",
        x  = signal,
        y  = shot_and_read_var,
        channels = channels,
        # Optional arguments
        read = args.read_noise,
        shot_var = shot_var if args.read_noise else None,
        fitted = fit_params
    )


