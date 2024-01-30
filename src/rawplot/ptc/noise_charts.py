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

def plot_read_noise_line(axes, read_noise):
    '''Plot an horizontal line'''
    text = r"$\sigma_{READ}$"
    axes.axhline(read_noise, linestyle='-', label=text)

def plot_fpn_line(axes, p_fpn):
    P0 = (1, p_fpn)
    P1 = (1/p_fpn, 1)
    text = r"$\sigma_{FPN}, m=1$"
    axes.axline(P0, P1, linestyle='--', label=text)

def plot_shot_line(axes, gain):
    P0 = (1, 1/math.sqrt(gain))
    P1 = (gain, 1)
    text = r"$\sigma_{SHOT}, m=\frac{1}{2}$"
    axes.axline(P0, P1, linestyle=':', label=text)

def plot_noise_vs_signal(axes, i, x, y, xtitle, ytitle, ylabel, channels, **kwargs):
    '''For Charts 1 to 8'''
    # Main plot goes here
    axes.plot(x[i], y[i], marker='o', linewidth=0, label=ylabel)
    # Additional plots go here
    base = 2 if kwargs.get('log2', False) else 10
    phys = kwargs.get('phys', False)
    for key, value in kwargs.items():
        if key in ('shot', 'fpn',) :
            label = rf"$\sigma_{ {key.upper()} }$"
            axes.plot(x[i], value[i], marker='o', linewidth=0, label=label)
        elif key == 'read' and value is not None:
            plot_read_noise_line(axes, value) #  read noise is a scalar
        elif key == 'p_fpn' and value is not None and not phys:
            plot_fpn_line(axes, value)
        elif key == 'gain' and value is not None and not phys:
            plot_shot_line(axes, value)
    axes.set_title(f'channel {channels[i]}')
    axes.set_xscale('log', base=base)
    axes.set_yscale('log', base=base)
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.set_xlabel(xtitle)
    axes.set_ylabel(ytitle)
    if ylabel:
        axes.legend()


def noise_curve1(args):
    log.info(" === NOISE CHART 1: Individual Noise Sources vs. Signal === ")
    units, gain, phys = check_physical(args)
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
    total_noise = np.sqrt(total_var)
    shot_noise = np.sqrt(shot_var)
    fpn_noise = np.sqrt(fpn_var)
    if gain and phys:
        total_noise *= gain
        shot_noise *= gain
        fpn_noise *= gain
        read_noise *= gain
        signal  *= gain
    title = make_plot_title_from("Individual Noise Sources vs. Signal",metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
        plot_func = plot_noise_vs_signal,
        xtitle = f"Signal {units}",
        ytitle = f"Noise {units}",
        x     = signal,
        y     = total_noise,
        ylabel =r"$\sigma_{TOTAL}$",
        channels = channels,
        # Optional arguments
        shot  = shot_noise,
        fpn   = fpn_noise,
        read  = read_noise, # Still an scalar
        p_fpn = args.p_fpn,
        gain = gain,
        log2 = args.log2,
        phys = phys,
    )


def noise_curve2(args):
    log.info(" === NOISE CHART 2: Shot plus Readout Noise vs. Signal === ")
    units, gain, phys = check_physical(args)
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
    shot_read_noise = np.sqrt(shot_read_var)
    if gain and phys:
        shot_read_noise *= gain
        signal *= gain
        read_noise *= gain
    title = make_plot_title_from(r"$\sigma_{SHOT+READ}$ vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
        plot_func = plot_noise_vs_signal,
        xtitle = f"Signal {units}",
        ytitle = f"Noise {units}",
        ylabel =r"$\sigma_{SHOT+READ}$",
        x     = signal,
        y  = shot_read_noise,
        channels = channels,
        # Optional arguments
        read = read_noise,
        log2 = args.log2,
        phys = phys,
    )


def noise_curve3(args):
    log.info(" === NOISE CHART 3: Shot Noise vs. Signal === ")
    units, gain, phys = check_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    read_noise = args.read_noise
    signal, total_var, shot_read_var, shot_var, fpn_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = bias, 
        read_noise = read_noise,
        log2 = args.log2,
    )
    shot_noise = np.sqrt(shot_var)
    if gain and phys:
        shot_noise *= gain
        signal *= gain
        read_noise *= gain
    title = make_plot_title_from(r"$\sigma_{SHOT}$ vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
        plot_func = plot_noise_vs_signal,
        xtitle = f"Signal {units}",
        ytitle = f"Noise {units}",
        x     = signal,
        y  = shot_noise,
        ylabel =r"$\sigma_{SHOT}$",
        channels = channels,
        # Optional arguments
        read = read_noise,
        phys = phys,
    )


def noise_curve4(args):
    log.info(" === NOISE CHART 4: Fixed Pattern Noise vs. Signal === ")
    units, gain, phys = check_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    read_noise = args.read_noise
    signal, total_var, shot_read_var, shot_var, fpn_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = bias, 
        read_noise = read_noise,
        log2 = args.log2,
    )
    fpn_noise = np.sqrt(fpn_var)
    if gain and phys:
        fpn_noise *= gain
        signal *= gain
        read_noise *= gain
    title = make_plot_title_from(r"$\sigma_{FPN}$ vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
        plot_func = plot_noise_vs_signal,
        xtitle = f"Signal {units}",
        ytitle = f"Noise {units}",
        x     = signal,
        y  = fpn_noise,
        ylabel =r"$\sigma_{FPN}$",
        channels = channels,
        # Optional arguments
        read = read_noise,
        log2 = args.log2,
         phys = phys,
    )
