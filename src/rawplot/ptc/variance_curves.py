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

from sklearn.linear_model import  TheilSenRegressor, LinearRegression


from lica.validators import vdir, vfile, vfloat, vfloat01, valid_channels
from lica.raw.loader import ImageLoaderFactory, SimulatedDarkImage, NormRoi

# ------------------------
# Own modules and packages
# ------------------------

from .._version import __version__
from ..util.mpl.plot import mpl_main_plot_loop
from ..util.common import common_list_info, bias_from, make_plot_title_from, assert_physical, assert_range
from .common import signal_and_noise_variances
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


def fit(x, y, x0, x1, channels):
    estimator = TheilSenRegressor(random_state=42,  fit_intercept=True)
    #estimator = LinearRegression(fit_intercept=True)
    fit_params = list()
    mask = np.logical_and(x >= x0, x <= x1)
    for i, ch in enumerate(channels):
        m = mask[i]
        sub_x = x[i][m]
        sub_y = y[i][m]
        sub_x = sub_x.reshape(-1,1)
        estimator.fit(sub_x, sub_y)
        score = estimator.score(sub_x, sub_y)
        log.info("[%s] %s fitting score is %f. y=%.4f*x%+.4f", ch, estimator.__class__.__name__, score,  estimator.coef_[0], estimator.intercept_)
        fit_params.append({'score': score, 'slope': estimator.coef_[0], 'intercept': estimator.intercept_, 
            'x': sub_x, 'y': sub_y})
    return fit_params

def plot_fitted(axes, fitted):
    '''All graphical elements for a fitting line'''
    slope = fitted['slope']
    score = fitted['score']
    intercept = fitted['intercept']
    fitted_x = fitted['x']
    fitted_y = fitted['y']
    label = rf"$\sigma_{{READ+SHOT}}^2$ (model)"
    P0 = (0, intercept) 
    P1 = ( -intercept/slope)
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
    assert_range(args)
    units = "[DN]"
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    read_noise = args.read_noise if args.read_noise is not None else 0.0
    signal, total_var, shot_and_read_var, fpn_var, shot_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = bias, 
        read_noise = read_noise
    )
    if args.from_value and args.to_value:
        fit_params = fit(signal, shot_and_read_var, args.from_value, args.to_value, channels)
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
        shot_var = shot_var if args.read_noise else None,
        fitted = fit_params
    )

