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
from lica.raw.analyzer.image import ImageStatistics, ImagePairStatistics


# ------------------------
# Own modules and packages
# ------------------------

from .._version import __version__
from ..util.mpl.plot import mpl_main_plot_loop
from ..util.common import common_list_info, bias_from, make_plot_title_from

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

def check_physical(args):
    gain = args.gain
    phys = args.physical_units
    if gain is None and phys:
        raise ValueError("Can'use physycal units [-e] if --gain is not set")
    units = r"$[e^{-}]$" if gain is not None and phys else "[DN]"
    return units, gain, phys

def signal_and_total_noise_from(file_list, n_roi, channels, bias):
    file_list = file_list[::2]
    N = len(file_list)
    signal_list = list()
    noise_list = list()
    for i, path in enumerate (file_list, start=1):
        analyzer = ImageStatistics(path, n_roi, channels, bias)
        analyzer.run()
        signal_list.append(analyzer.mean())
        noise_var = analyzer.variance()
        noise_list.append(noise_var)
        log.info("[%d/%d] \u03C3\u00b2(total) for image %s = %s", i, N, analyzer.name(), noise_var)
    return np.stack(signal_list, axis=-1), np.stack(noise_list, axis=-1)


def read_and_shot_noise_from(file_list, n_roi, channels, bias):
    file_pairs = list(zip(file_list, file_list[1:]))[::2]
    N = len(file_pairs)
    noise_list = list()
    for i, (path_a, path_b) in enumerate(file_pairs, start=1):
        analyzer = ImagePairStatistics(path_a, path_b, n_roi, channels, bias)
        analyzer.run()
        noise_var = analyzer.variance()
        noise_list.append(noise_var)
        log.info("[%d/%d] \u03C3\u00b2(sh+rd) for image pair %s = %s",  i, N, analyzer.names(), noise_var)
    return  np.stack(noise_list, axis=-1)


def signal_and_noise_variances(file_list, n_roi, channels, bias, read_noise):
    signal, total_noise_var = signal_and_total_noise_from(file_list, n_roi, channels, bias)
    read_noise_var = np.full_like(signal, read_noise**2)
    shot_read_noise_var = read_and_shot_noise_from(file_list, n_roi, channels, bias)
    fixed_pattern_noise_var = total_noise_var - shot_read_noise_var
    shot_noise_var = shot_read_noise_var - read_noise_var
    return signal, total_noise_var, shot_read_noise_var, shot_noise_var, fixed_pattern_noise_var, read_noise_var


def plot_fpn_line(axes, p_fpn):
    P0 = (1, p_fpn)
    P1 = (1/p_fpn, 1)
    axes.axline(P0, P1, linestyle='--', label=r"$\sigma_{FPN}$ ideal")

def plot_shot_line(axes, gain):
    P0 = (1, 1/math.sqrt(gain))
    P1 = (gain, 1)
    axes.axline(P0, P1, linestyle=':', label=r"$\sigma_{SHOT}$ ideal")

def plot_noise_vs_signal(axes, i, x, y, xtitle, ytitle, ylabel, channels, **kwargs):
    '''For Charts 1 to 8'''
    # Main plot goes here
    axes.plot(x[i], y[i], marker='o', linewidth=0, label=ylabel)
    # Additional plots go here
    base = 2 if kwargs.get('log2', False) else 10
    for key, value in kwargs.items():
        if key in ('shot', 'fpn', 'read') :
            label = rf"$\sigma_{ {key.upper()} }$"
            axes.plot(x[i], value[i], marker='o', linewidth=0, label=label)
        elif key == 'p_fpn' and value is not None:
            plot_fpn_line(axes, value)
        elif key == 'gain' and value is not None:
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


def noise_chart1(args):
    log.info(" === NOISE CHART 1: Individual Noise Sources vs. Signal === ")
    units, gain, phys = check_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    signal, total_var, shot_read_var, shot_var, fpn_var, read_noise_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = bias, 
        read_noise = args.read_noise
    )
    total_noise = np.sqrt(total_var)
    shot_noise = np.sqrt(shot_var)
    fpn_noise = np.sqrt(fpn_var)
    read_noise = np.sqrt(read_noise_var) # Now, read_noise is a numpy array
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
        read  = read_noise, 
        p_fpn = args.p_fpn,
        gain = gain,
        log2 = args.log2,
    )


def noise_chart2(args):
    log.info(" === NOISE CHART 2: Shot plus Readout Noise vs. Signal === ")
    units, gain, phys = check_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    signal, total_var, shot_read_var, shot_var, fpn_var, read_noise_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = bias, 
        read_noise = args.read_noise
    )
    shot_read_noise = np.sqrt(shot_read_var)
    if gain and phys:
        shot_read_noise *= gain
        signal *= gain
    title = make_plot_title_from(r"$\sigma_{SHOT+READ}$ vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
        plot_func = plot_noise_vs_signal,
        xtitle = f"Signal {units}",
        ytitle = f"Noise {units}",
        x     = signal,
        y  = shot_read_noise,
        channels = channels,
        log2 = args.log2,
    )


def noise_chart3(args):
    log.info(" === NOISE CHART 3: Shot Noise vs. Signal === ")
    units, gain, phys = check_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    signal, total_var, shot_read_var, shot_var, fpn_var, read_noise_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = bias, 
        read_noise = args.read_noise,
        log2 = args.log2,
    )
    shot_noise = np.sqrt(shot_var)
    if gain and phys:
        shot_noise *= gain
        signal *= gain
    title = make_plot_title_from(r"$\sigma_{SHOT}$ vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
        plot_func = plot_noise_vs_signal,
        xtitle = f"Signal {units}",
        ytitle = f"Noise {units}",
        x     = signal,
        y  = shot_noise,
        channels = channels,
    )


def noise_chart4(args):
    log.info(" === NOISE CHART 4: Fixed Pattern Noise vs. Signal === ")
    units, gain, phys = check_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    signal, total_var, shot_read_var, shot_var, fpn_var, read_noise_var = signal_and_noise_variances(
        file_list = file_list, 
        n_roi = n_roi, 
        channels = channels, 
        bias = bias, 
        read_noise = args.read_noise,
        log2 = args.log2,
    )
    fpn_noise = np.sqrt(fpn_var)
    if gain and phys:
        fpn_noise *= gain
        signal *= gain
    title = make_plot_title_from(r"$\sigma_{FPN}$ vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
        plot_func = plot_noise_vs_signal,
        xtitle = f"Signal {units}",
        ytitle = f"Noise {units}",
        x     = signal,
        y  = fpn_noise,
        channels = channels,
        log2 = args.log2,
    )
