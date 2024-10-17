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

import math
import logging

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
from lica.validators import vdir, vfloat, vfloat01, vflopath

# ------------------------
# Own modules and packages
# ------------------------

from ..util.mpl.plot import mpl_main_plot_loop
from ..util.common import common_list_info, make_plot_title_from, assert_physical, assert_range
from .common import signal_and_noise_variances, is_estimate, estimate, vfit, float_or_none

# ----------------
# Module constants
# ----------------


# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)


def noise_parser_arguments(parser):
    parser.add_argument('-i', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser.add_argument('-f', '--image-filter', type=str, required=True, help='Images filter, glob-style (i.e. flat*, dark*)')
    parser.add_argument('-x', '--x0', type=vfloat01,  help='Normalized ROI start point, x0 coordinate [0..1]')
    parser.add_argument('-y', '--y0', type=vfloat01,  help='Normalized ROI start point, y0 coordinate [0..1]')
    parser.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser.add_argument('--every', type=int, metavar='<N>', default=1, help='pick every n `file after sorting')
    parser.add_argument('-bi', '--bias',  type=vflopath,  help='Bias, either a single value for all channels or else a 3D FITS cube file (default: %(default)s)')
    parser.add_argument('-dk', '--dark',  type=vfloat,  help='Dark count rate in DN/sec. (default: %(default)s)')
    parser.add_argument('--p-fpn', type=vfit, metavar='<p>',  help='Fixed Pattern Noise Percentage factor: [0..1] or "estimate" (default: %(default)s)')
    parser.add_argument('-rd','--read-noise', type=vfit, metavar='<\u03C3>',  help='Read noise [DN] or "estimate" (default: %(default)s)')
    parser.add_argument('-gn','--gain', type=vfloat, metavar='<g>',  help='Gain [e-/DN] (default: %(default)s)')
    parser.add_argument('-ph','--physical-units',  action='store_true', help='Display in [e-] physical units instead of [DN]. Requires --gain')
    parser.add_argument('--log2',  action='store_true', help='Display plot using log2 instead of log10 scale')
    parser.add_argument('-fr','--from', dest='from_value', type=vfloat, metavar='<x0>',  help='Lower signal limit to fit [DN] (default: %(default)s)')
    parser.add_argument('-to','--to', dest='to_value', type=vfloat, metavar='<x1>',  help='Upper signal limit to fit [DN] (default: %(default)s)')


def p_fpn_estimator(x, y):
    return y/x

def rdnoise_estimator(x, y):
    return y

def plot_fitted_line(axes, fitted):
    label = fitted['label']
    fitted_y = fitted['y']
    fitted_x = fitted['x']
    p_fpn = fitted['mean']
    axes.plot(fitted_x, fitted_y, marker='o', linewidth=0, label=f"{label} (selected)")
    axes.axline( (1, p_fpn), (1/p_fpn, 1), linestyle='--', label=r"$\sigma_{FPN}, m=1$")

def plot_fitted_box(axes, fitted):
    '''All graphical elements for a fitting line'''
    mean = fitted['mean']
    std = fitted['std']
    label = fitted['label']
    text = "\n".join( (f"{label}", fr"$\mu = {mean:0.2e}$", fr"$\sigma = {std:0.2e}$"))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes.text(0.4, 0.95, text, transform=axes.transAxes, va='top', bbox=props)

def plot_noise_vs_signal(axes, i, x, y, xtitle, ytitle, ylabel, channels, **kwargs):
    '''For Curves 1 to 8'''
    phys = kwargs.get('phys', False) # rae we dealing with physical values?
    # Main data plot goes here
    axes.plot(x[i], y[i], marker='o', linewidth=0, label=ylabel)
    # Additional data plots go here
    read_noise = kwargs.get('read', None)
    fpn_noise = kwargs.get('fpn', None)
    shot_noise = kwargs.get('shot', None)
    if shot_noise is not None:
        label = r"$\sigma_{SHOT}$" if read_noise is not None else r"$\sigma_{SHOT+READ}$"
        axes.plot(x[i], shot_noise[i], marker='o', linewidth=0, label=label)
    if fpn_noise is not None:
        axes.plot(x[i], fpn_noise[i], marker='o', linewidth=0, label=r"$\sigma_{FPN}$")
    fitted = kwargs.get('fitted', None)
    if fitted is not None:
        plot_fitted_box(axes, fitted[i])
        plot_fitted_line(axes, fitted[i])
    # Optional theoretical model lines
    read_noise = kwargs.get('read', None)
    if read_noise is not None:
        axes.axhline(read_noise, linestyle=':', label=r"$\sigma_{READ}$")
    gain = kwargs.get('gain', None)
    if gain is not None:
        # Points following Poison error
        P0 = (1,1) if phys else (1, 1/math.sqrt(gain))
        P1 = (4,2) if phys else (gain, 1)
        axes.axline(P0, P1, linestyle='-.', label=r"$\sigma_{SHOT}, m=\frac{1}{2}$")
    p_fpn = kwargs.get('p_fpn', None)
    if p_fpn is not None:
        axes.axline( (1, p_fpn), (1/p_fpn, 1), linestyle='--', label=r"$\sigma_{FPN}, m=1$")
    # Optional (vertical) Zones
    if read_noise is not None and gain is not None:
        Y = read_noise**2 if phys else gain*(read_noise**2)
        axes.axvline(Y, linestyle='--', linewidth=2, color='k')
    if gain is not None and p_fpn is not None:
        Y = 1 / (p_fpn**2) if phys else 1 / (gain*p_fpn**2)
        axes.axvline(Y, linestyle='--', linewidth=2, color='k')
    # Titles, scales and grids
    axes.set_title(f'channel {channels[i]}')
    base = 2 if kwargs.get('log2', False) else 10
    axes.set_xscale('log', base=base)
    axes.set_yscale('log', base=base)
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    units = r"$[e^{-}]$" if phys else "[DN]"
    axes.set_xlabel(f"{xtitle} {units}")
    axes.set_ylabel(f"{ytitle} {units}")
    if ylabel:
        axes.legend()

# ------------------------
# AUXILIARY MAIN FUNCTIONS
# ------------------------

def noise_curve1(args):
    log.info(" === NOISE CHART 1: Individual Noise Sources vs. Signal === ")
    assert_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    read_noise = args.read_noise if type(args.read_noise) is float else 0.0
    signal, total_var, shot_read_var, fpn_var, shot_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = args.bias, 
        dark = args.dark,
        read_noise = read_noise
    )
    total_noise = np.sqrt(total_var)
    shot_noise = np.sqrt(shot_var)
    fpn_noise = np.sqrt(fpn_var)
    if args.gain and args.physical_units:
        total_noise *= args.gain
        shot_noise *= args.gain
        fpn_noise *= args.gain
        read_noise *= args.gain
        signal *= args.gain
    if is_estimate(args.read_noise):
        assert_range(args)
        fit_params = estimate(signal, total_noise, args.from_value, args.to_value, channels, 
            rdnoise_estimator, label=r"$\sigma_{READ}$")
    elif is_estimate(args.p_fpn):
        assert_range(args)
        fit_params = estimate(signal, fpn_noise, args.from_value, args.to_value, channels, 
            p_fpn_estimator, label=r"$\sigma_{FPN}$")
    else:
        fit_params = None
    title = make_plot_title_from("Individual Noise Sources vs. Signal",metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        plot_func = plot_noise_vs_signal, # 2D (channel, data) Numpy array
        xtitle = "Signal",
        ytitle = "Noise",
        x     = signal,
        y     = total_noise,
        ylabel =r"$\sigma_{TOTAL}$",
        channels = channels,
        # Optional arguments
        shot  = shot_noise, # 2D (channel, data) Numpy array
        fpn   = fpn_noise,  # 2D (channel, data) Numpy array
        read = float_or_none(args.read_noise),
        p_fpn = float_or_none(args.p_fpn),
        gain = args.gain,
        phys = args.physical_units,
        log2 = args.log2,
        fitted = fit_params
    )


def noise_curve2(args):
    log.info(" === NOISE CHART 2: Shot plus Readout Noise vs. Signal === ")
    assert_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    read_noise = args.read_noise or 0.0
    signal, total_var, shot_read_var, fpn_var, shot_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = args.bias, 
        dark = args.dark,
        read_noise = read_noise
    )
    shot_read_noise = np.sqrt(shot_read_var)
    if args.gain and args.physical_units:
        shot_read_noise *= args.gain
        signal *= args.gain
        read_noise *= args.gain
    if is_estimate(args.read_noise):
        assert_range(args)
        fit_params = estimate(signal, shot_read_noise, args.from_value, args.to_value, channels, 
            rdnoise_estimator, label=r"$\sigma_{READ}$")
    else:
        fit_params = None
    title = make_plot_title_from(r"$\sigma_{SHOT+READ}$ vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        plot_func = plot_noise_vs_signal,
        xtitle = "Signal",
        ytitle = "Noise",
        ylabel =r"$\sigma_{SHOT+READ}$",
        x     = signal,
        y  = shot_read_noise,
        channels = channels,
        # Optional arguments
        read = float_or_none(args.read_noise),
        phys = args.physical_units,
        log2 = args.log2,
        fitted = fit_params
    )


def noise_curve3(args):
    log.info(" === NOISE CHART 3: Shot Noise vs. Signal === ")
    assert_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    read_noise = args.read_noise or 0.0
    signal, total_var, shot_read_var, fpn_var, shot_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = args.bias, 
        dark = args.dark,
        read_noise = read_noise,
    )
    shot_noise = np.sqrt(shot_var)
    if args.gain and args.physical_units:
        shot_noise *= args.gain
        signal *= args.gain
        read_noise *= args.gain
    title = make_plot_title_from(r"$\sigma_{SHOT}$ vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        plot_func = plot_noise_vs_signal,
        xtitle = "Signal",
        ytitle = "Noise",
        x     = signal,
        y  = shot_noise,
        ylabel =r"$\sigma_{SHOT}$",
        channels = channels,
        # Optional arguments
        read = float_or_none(args.read_noise),
        gain = args.gain,
        phys = args.physical_units,
        log2 = args.log2,
    )


def noise_curve4(args):
    log.info(" === NOISE CHART 4: Fixed Pattern Noise vs. Signal === ")
    assert_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    read_noise = args.read_noise or 0.0
    signal, total_var, shot_read_var, fpn_var, shot_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = args.bias,
        dark = args.dark, 
        read_noise = read_noise,
    )
    fpn_noise = np.sqrt(fpn_var)
    if args.gain and args.physical_units:
        fpn_noise *= args.gain
        signal *= args.gain
        read_noise *= args.gain
    if is_estimate(args.p_fpn):
        assert_range(args)
        fit_params = estimate(signal, fpn_noise, args.from_value, args.to_value, channels, 
            p_fpn_estimator, label=r"$\sigma_{FPN}$")
    else:
        fit_params = None
    title = make_plot_title_from(r"$\sigma_{FPN}$ vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        plot_func = plot_noise_vs_signal,
        xtitle = "Signal",
        ytitle = "Noise",
        x     = signal,
        y  = fpn_noise,
        ylabel =r"$\sigma_{FPN}$",
        channels = channels,
        # Optional arguments
        read = float_or_none(args.read_noise),
        p_fpn = float_or_none(args.p_fpn),
        phys = args.physical_units,
        log2 = args.log2,
        fitted = fit_params,
    )
