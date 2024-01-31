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
from ..util.common import common_list_info, bias_from, make_plot_title_from, assert_physical
from .common import signal_and_noise_variances_from, signal_and_noise_variances

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
    parser.add_argument('--log2',  action='store_true', help='Display plot using log2 instead of log10 scale')
    parser.add_argument('--p-fpn', type=vfloat01, metavar='<p>', default=None, help='Fixed Pattern Noise Percentage factor [0..1] (default: %(default)s)')
    parser.add_argument('-gn','--gain', type=vfloat, metavar='<g>', default=None, help='Gain [e-/DN] (default: %(default)s)')
    parser.add_argument('-ph','--physical-units',  action='store_true', help='Display in [e-] physical units instead of [DN]. Requires --gain')
    

def plot_noise_vs_signal(axes, i, x, y, xtitle, ytitle, ylabel, channels, **kwargs):
    '''For Curves 1 to 8'''
    # Main data plot goes here
    axes.plot(x[i], y[i], marker='o', linewidth=0, label=ylabel)
    # Additional data plots go here
    read_noise = kwargs.get('read', None)
    fpn_noise = kwargs.get('fpn', None)
    shot_noise = kwargs.get('shot', None)
    phys = kwargs.get('phys', False)
    if shot_noise is not None:
        label = r"$\sigma_{SHOT}$" if read_noise is not None else r"$\sigma_{SHOT+READ}$"
        axes.plot(x[i], shot_noise[i], marker='o', linewidth=0, label=label)
    if fpn_noise is not None:
        axes.plot(x[i], fpn_noise[i], marker='o', linewidth=0, label=r"$\sigma_{FPN}$")
    # Optional theoretical model lines
    read_noise = kwargs.get('read', None)
    if read_noise is not None:
        axes.axhline(read_noise, linestyle=':', label=r"$\sigma_{READ}$")
    gain = kwargs.get('gain', None)
    if gain is not None:
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
    bias = bias_from(args)
    read_noise = args.read_noise if args.read_noise is not None else 0.0
    signal, total_var, shot_read_var, shot_var, fpn_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = bias, 
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
    title = make_plot_title_from("Individual Noise Sources vs. Signal",metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
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
        read = None if read_noise == 0.0 else read_noise,
        p_fpn = args.p_fpn,
        gain = args.gain,
        log2 = args.log2,
        phys = args.physical_units,
    )


def noise_curve2(args):
    log.info(" === NOISE CHART 2: Shot plus Readout Noise vs. Signal === ")
    assert_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    read_noise = args.read_noise if args.read_noise is not None else 0.0
    signal, total_var, shot_read_var, shot_var, fpn_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = bias, 
        read_noise = read_noise
    )
    shot_read_noise = np.sqrt(shot_read_var)
    if args.gain and args.physical_units:
        shot_read_noise *= args.gain
        signal *= args.gain
        read_noise *= args.gain
    title = make_plot_title_from(r"$\sigma_{SHOT+READ}$ vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
        plot_func = plot_noise_vs_signal,
        xtitle = "Signal",
        ytitle = "Noise",
        ylabel =r"$\sigma_{SHOT+READ}$",
        x     = signal,
        y  = shot_read_noise,
        channels = channels,
        # Optional arguments
        read = None if read_noise == 0.0 else read_noise,
        log2 = args.log2,
        phys = args.physical_units,
    )


def noise_curve3(args):
    log.info(" === NOISE CHART 3: Shot Noise vs. Signal === ")
    assert_physical(args)
    assert args.read_noise is not None
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    read_noise = args.read_noise if args.read_noise is not None else 0.0
    signal, total_var, shot_read_var, shot_var, fpn_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = bias, 
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
        figsize  = (12, 9),
        plot_func = plot_noise_vs_signal,
        xtitle = "Signal",
        ytitle = "Noise",
        x     = signal,
        y  = shot_noise,
        ylabel =r"$\sigma_{SHOT}$",
        channels = channels,
        # Optional arguments
        read = None if read_noise == 0.0 else read_noise,
        gain = args.gain,
        phys = args.physical_units,
    )


def noise_curve4(args):
    log.info(" === NOISE CHART 4: Fixed Pattern Noise vs. Signal === ")
    assert_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    read_noise = args.read_noise if args.read_noise is not None else 0.0
    signal, total_var, shot_read_var, shot_var, fpn_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = bias, 
        read_noise = read_noise,
    )
    fpn_noise = np.sqrt(fpn_var)
    if args.gain and args.physical_units:
        fpn_noise *= args.gain
        signal *= args.gain
        read_noise *= args.gain
    title = make_plot_title_from(r"$\sigma_{FPN}$ vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
        plot_func = plot_noise_vs_signal,
        xtitle = "Signal",
        ytitle = "Noise",
        x     = signal,
        y  = fpn_noise,
        ylabel =r"$\sigma_{FPN}$",
        channels = channels,
        # Optional arguments
        read = None if read_noise == 0.0 else read_noise,
        p_fpn = args.p_fpn,
        log2 = args.log2,
        phys = args.physical_units,
    )
