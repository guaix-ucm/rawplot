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
from lica.validators import vdir, vfloat01, valid_channels
from lica.raw.loader import ImageLoaderFactory, SimulatedDarkImage, NormRoi
from lica.misc import file_paths

# ------------------------
# Own modules and packages
# ------------------------

from .._version import __version__
from ..util.mpl.plot import plot_layout, axes_reshape
from ..util.common import common_list_info, bias_from
from lica.raw.analyzer.image import ImageStatistics, ImagePairStatistics
from .common import plot_noise_vs_signal, signal_and_total_noise_from, read_and_shot_noise_from

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


def noise_chart1(args):
    log.info(" === NOISE CHART 1 === ")
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    signal, total_noise_var = signal_and_total_noise_from(file_list, n_roi, channels, bias)
    shrd_noise_var = read_and_shot_noise_from(file_list, n_roi, channels, bias)
    fpn_var = total_noise_var - shrd_noise_var
    if args.bias_file is not None:
        analyzer = ImageStatistics(args.bias_file, n_roi, channels)
        analyzer.run()
        rdnoise_var = np.full_like(signal, analyzer.mean().reshape(len(channels),-1))
    else:
        rdnoise_var = np.full_like(signal, args.rd_noise**2)
    shot_var = shrd_noise_var - rdnoise_var
    total_noise = np.sqrt(total_noise_var)
    shot_noise = np.sqrt(shot_var)
    fpn_noise = np.sqrt(fpn_var)
    rd_noise = np.sqrt(rdnoise_var)
    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(12, 9), layout='tight')
    fig.suptitle(f"Total Noise Sources vs. Signal\n"
            f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}\n"
            f"Color Plane Size: {metadata['width']} cols x {metadata['height']} rows\n"
            f"ROI: {roi} {roi.width()} cols x {roi.height()} rows")
    axes = axes_reshape(axes, channels)
    for row in range(0,display_rows):
        for col in range(0,display_cols):
            i = 2*row+col
            if len(channels) == 3 and row == 1 and col == 1: # Skip the empty slot in 2x2 layout with 3 items
                axes[row][col].set_axis_off()
                break
            plot_noise_vs_signal(axes[row][col], signal[i], channels[i], 'Noise', '[DN]',
                total=total_noise[i],
                fpn=fpn_noise[i],
                shot=shot_noise[i],
                rdnoise=rd_noise[i]
            )
    plt.show()


def noise_chart2(args):
    log.info(" === NOISE CHART 2 === ")
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    signal, total_noise_var = signal_and_total_noise_from(file_list, n_roi, channels, bias)
    shrd_noise_var = read_and_shot_noise_from(file_list, n_roi, channels, bias)
    fpn_var = total_noise_var - shrd_noise_var
    if args.bias_file is not None:
        analyzer = ImageStatistics(args.bias_file, n_roi, channels)
        analyzer.run()
        rdnoise_var = np.full_like(signal, analyzer.mean().reshape(len(channels),-1))
    else:
        rdnoise_var = np.full_like(signal, args.rd_noise**2)
    shot_var = shrd_noise_var - rdnoise_var
    total_noise = np.sqrt(shrd_noise_var)
    shot_noise = np.sqrt(shot_var)
    rd_noise = np.sqrt(rdnoise_var)
    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(12, 9), layout='tight')
    fig.suptitle(f"Shot plus Readout Noise vs. Signal\n"
            f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}\n"
            f"Color Plane Size: {metadata['width']} cols x {metadata['height']} rows\n"
            f"ROI: {roi} {roi.width()} cols x {roi.height()} rows")
    axes = axes_reshape(axes, channels)
    for row in range(0,display_rows):
        for col in range(0,display_cols):
            i = 2*row+col
            if len(channels) == 3 and row == 1 and col == 1: # Skip the empty slot in 2x2 layout with 3 items
                axes[row][col].set_axis_off()
                break
            plot_noise_vs_signal(axes[row][col], signal[i], channels[i], 'Noise', '[DN]',
                total=total_noise[i],
                shot=shot_noise[i],
                rdnoise=rd_noise[i]
            )
    plt.show()


def noise_chart3(args):
    log.info(" === NOISE CHART 3 === ")
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    signal, total_noise_var = signal_and_total_noise_from(file_list, n_roi, channels, bias)
    shrd_noise_var = read_and_shot_noise_from(file_list, n_roi, channels, bias)
    fpn_var = total_noise_var - shrd_noise_var
    if args.bias_file is not None:
        analyzer = ImageStatistics(args.bias_file, n_roi, channels)
        analyzer.run()
        rdnoise_var = np.full_like(signal, analyzer.mean().reshape(len(channels),-1))
    else:
        rdnoise_var = np.full_like(signal, args.rd_noise**2)
    shot_var = shrd_noise_var - rdnoise_var 
    shot_noise = np.sqrt(shot_var)
    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(12, 9), layout='tight')
    fig.suptitle(f"Shot Noise vs. Signal\n"
            f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}\n"
            f"Color Plane Size: {metadata['width']} cols x {metadata['height']} rows\n"
            f"ROI: {roi} {roi.width()} cols x {roi.height()} rows")
    axes = axes_reshape(axes, channels)
    for row in range(0,display_rows):
        for col in range(0,display_cols):
            i = 2*row+col
            if len(channels) == 3 and row == 1 and col == 1: # Skip the empty slot in 2x2 layout with 3 items
                axes[row][col].set_axis_off()
                break
            plot_noise_vs_signal(axes[row][col], signal[i], channels[i], 'Noise', '[DN]',              
                shot=shot_noise[i],
            )
    plt.show()


def noise_chart4(args):
    log.info(" === NOISE CHART 4 === ")
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    signal, total_noise_var = signal_and_total_noise_from(file_list, n_roi, channels, bias)
    shrd_noise_var = read_and_shot_noise_from(file_list, n_roi, channels, bias)
    fpn_var = total_noise_var - shrd_noise_var
    fpn_noise = np.sqrt(fpn_var)
    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(12, 9), layout='tight')
    fig.suptitle(f"Fixed Pattern Noise (FPN) vs. Signal\n"
            f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}\n"
            f"Color Plane Size: {metadata['width']} cols x {metadata['height']} rows\n"
            f"ROI: {roi} {roi.width()} cols x {roi.height()} rows")
    axes = axes_reshape(axes, channels)
    for row in range(0,display_rows):
        for col in range(0,display_cols):
            i = 2*row+col
            if len(channels) == 3 and row == 1 and col == 1: # Skip the empty slot in 2x2 layout with 3 items
                axes[row][col].set_axis_off()
                break
            plot_noise_vs_signal(axes[row][col], signal[i], channels[i], 'Noise', '[DN]',
                fpn=fpn_noise[i]
            )
    plt.show()
