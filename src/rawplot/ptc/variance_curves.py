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

from sklearn.linear_model import  TheilSenRegressor


from lica.validators import vdir, vfile, vfloat, vfloat01, valid_channels
from lica.raw.loader import ImageLoaderFactory, SimulatedDarkImage, NormRoi

# ------------------------
# Own modules and packages
# ------------------------

from .._version import __version__
from ..util.mpl.plot import mpl_main_plot_loop
from ..util.common import common_list_info, bias_from, make_plot_title_from, assert_physical, assert_range
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
    parser.add_argument('-rd','--read-noise', type=vfloat, metavar='<\u03C3>', default=None, help='Read noise [DN] (default: %(default)s)')
    parser.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser.add_argument('--every', type=int, metavar='<N>', default=1, help='pick every n `file after sorting')
    group0 = parser.add_mutually_exclusive_group(required=False)
    group0.add_argument('-bl', '--bias-level', type=vfloat, default=None, help='Bias level, common for all channels (default: %(default)s)')
    group0.add_argument('-bf', '--bias-file',  type=vfile, default=None, help='Bias image (3D FITS cube) (default: %(default)s)')
    parser.add_argument('-fr','--from-value', type=vfloat, metavar='<x0>', default=None, help='Lower signal limit to fit [DN] (default: %(default)s)')
    parser.add_argument('-to','--to-value', type=vfloat, metavar='<x1>', default=None, help='Upper signal limit to fit [DN] (default: %(default)s)')


def signal_and_noise_variances(file_list, n_roi, channels, bias, read_noise):
    signal, total_noise_var, fpn_corrected_noise_var = signal_and_noise_variances_from(file_list, n_roi, channels, bias)
    fixed_pattern_noise_var = total_noise_var - fpn_corrected_noise_var
    return signal, total_noise_var, fpn_corrected_noise_var, fixed_pattern_noise_var

def fit(x, y, x0, x1, label):
    assert x.shape[0] == 1, "Only one color plane is allowed. Set --channel to one color only" 
    mask = np.logical_and(x >= x0, x <= x1)
    sub_x = x[mask]
    sub_y = y[mask]
    sub_x = sub_x.reshape(-1,1)
    estimator = TheilSenRegressor(random_state=42,  fit_intercept=True)
    estimator.fit(sub_x, sub_y)
    score = estimator.score(sub_x, sub_y)
    log.info("[%s] %s fitting score is %f. y=%.4f*x%+.4f", label, estimator.__class__.__name__, score,  estimator.coef_[0], estimator.intercept_)
    intercept = estimator.intercept_
    slope = estimator.coef_[0]
    return slope, intercept, score, sub_x, sub_y


def plot_variance_vs_signal(axes, i, x, y, xtitle, ytitle, ylabel, channels, **kwargs):
    '''For Charts 5'''
    # Main plot goes here (signal_and_read noise...)
    axes.plot(x[i], y[i], marker='o', linewidth=0, label=ylabel)
    # Additional plots go here
    total_noise = kwargs.get('total_var', None)
    if total_noise is not None:
        label = r"$\sigma_{TOTAL}^2$"
        axes.plot(x[i], total_noise[i], marker='o', linewidth=0, label=label)
    fpn_noise = kwargs.get('fpn_var', None)
    if fpn_noise is not None:
        label = r"$\sigma_{FPN}^2$"
        axes.plot(x[i], fpn_noise[i], marker='o', linewidth=0, label=label)
    read_noise = kwargs.get('read', None)
    if read_noise is not None:
        label = r"$\sigma_{READ}^2$"
        axes.axhline(read_noise**2, linestyle='--', label=text)
    fitted = kwargs.get('fitted', None)
    if fitted is not None:
        label = rf"fitted: $r^2 = {fitted[2]:.3f},\quad g = {1/fitted[0]:0.2f}\quad e^{{-}}/DN$"
        P0 = (0, fitted[1]); P1 = ( -fitted[1]/fitted[0])
        axes.plot(fitted[3], fitted[4], marker='o', linewidth=0, label="selected")
        axes.axline(P0, slope=fitted[0], linestyle='--', label=label)
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
    assert_range(args)
    units = "[DN]"
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    read_noise = args.read_noise
    signal, total_var, shot_and_read_var, fpn_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = bias, 
        read_noise = read_noise
    )
    if args.from_value and args.to_value:
        fit_params =  tuple(fit(signal, shot_and_read_var, args.from_value, args.to_value, channels[0]))
    else:
        fit_params = None

    title = make_plot_title_from(r"$\sigma_{READ+SHOT}^2$ vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
        plot_func = plot_variance_vs_signal,
        xtitle = f"Signal {units}",
        ytitle = f"Noise Variance {units}",
        ylabel =r"$\sigma_{READ+SHOT}^2$",
        x  = signal,
        y  = shot_and_read_var,
        channels = channels,
        # Optional arguments
        read = args.read_noise,
        #total_var = total_var,
        fitted = fit_params
    )


