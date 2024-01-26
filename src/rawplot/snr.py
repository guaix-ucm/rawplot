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
from lica.raw import ImageLoaderFactory, SimulatedDarkImage, NormRoi
from lica.misc import file_paths

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.mpl.plot import plot_layout, axes_reshape
from .util.common import preliminary_tasks

# ----------------
# Module constants
# ----------------

SQRT_2 = math.sqrt(2)

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# ------------------
# Auxiliary fnctions
# ------------------

def plot_snr(axes, signal, sn_ratio, channel, use_stops, full_scale):
    axes.set_title(fr'channel {channel}')
    full_scale = 1 if full_scale is None else full_scale
    signal = signal / full_scale
    if use_stops:
        axes.set_xscale('log', base=2)
        axes.set_yscale('log', base=2)
        units = "[stops]"
    else:
        units = "[DN]"
    relative = "(relative to full scale)" if full_scale is not None else ""
    title = f'Signal {relative} {units}'
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.set_xlabel(title)
    axes.plot(signal, sn_ratio,   marker='o', linewidth=0)
    axes.set_ylabel(f'Signal to Noise Ratio {units}')


def measure_snr_for(file_list, n_roi, channels):
    file_pairs = list(zip(file_list, file_list[1:]))[::2]
    signals =  list()
    snrs = list()
    factory =  ImageLoaderFactory()
    for path_a, path_b in file_pairs:
        image_a = factory.image_from(path_a, n_roi, channels)
        image_b = factory.image_from(path_b, n_roi, channels) 
        signal, sn_ratio = measure_snr(image_a, image_b)
        signals.append(signal)
        snrs.append(sn_ratio)
    signals = np.array(signals).transpose()
    snrs = np.array(snrs).transpose()
    return signals, snrs

def measure_readout_noise_for(file_list, n_roi, channels):
    file_pairs = list(zip(file_list, file_list[1:]))[::2]
    noises = list()
    isos = list()
    factory =  ImageLoaderFactory()
    for path_a, path_b in file_pairs:
        image_a = factory.image_from(path_a, n_roi, channels)
        image_b = factory.image_from(path_b, n_roi, channels) 
        iso, noise = measure_readout_noise(image_a, image_b)
        noises.append(noise)
    noises = np.array(noises).transpose()
    isos = np.array(isos).transpose()
    return isos, noises

def measure_snr(image_a, image_b):
    channels = image_a.channels() # Common to both
    # We must take the bias off to get valid results
    section_a = image_a.load().astype(float, copy=False) - np.array(image_a.black_levels()).reshape(len(channels), 1, 1)
    section_b = image_b.load().astype(float, copy=False) - np.array(image_a.black_levels()).reshape(len(channels), 1, 1)
    signal = np.mean((section_a + section_b), axis=(1,2)) / 2  
    noise = np.std((section_a - section_b), axis=(1,2)) / SQRT_2
    sn_ratio = signal / noise
    log.info("signal is %s, noise is %s", signal, noise)
    log.info("From %s, %s snr is %s", image_a.name(), image_b.name(), dict(zip(channels, sn_ratio.tolist())))
    return signal, sn_ratio


def measure_readout_noise(image_a, image_b):
    channels = image_a.channels() # Common to both
    iso = image_a.metadata()['iso']
    # We must take the bias off to get valid results
    section_a = image_a.load().astype(float, copy=False)
    section_b = image_b.load()().astype(float, copy=False)
    noise = np.std((section_a - section_b), axis=(1,2)) / SQRT_2
    iso = np.array([iso for ch in channels])
    log.info("From %s, %s , estimated noise at ISO gain %s is %s", image_a.name(), image_b.name(), iso, dict(zip(channels, noise.tolist())))
    return iso, noise


# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def snr(args):
    file_list, roi, n_roi, channels, metadata = preliminary_tasks(args)
    use_stops = args.stops
    full_scale = args.full_scale    
    signals, snrs = measure_snr_for(file_list, n_roi, channels)
    if(args.bias_filter):
        isos, noises = measure_readout_noise_for(file_list, n_roi, channels)
    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(12, 9), layout='tight')
    fig.suptitle(f"SNR vs Signal\n"
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
            plot_snr(axes[row][col], signals[i], snrs[i], channels[i], use_stops, full_scale)
    plt.show()


# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):
    parser.add_argument('-i', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser.add_argument('-f', '--image-filter', type=str, required=True, help='Images filter, glob-style (i.e. flat*, dark*)')
    parser.add_argument('-b', '--bias-filter', type=str, default=None, help='Bias Images filter, glob-style (i.e. bias*')
    parser.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser.add_argument('--stops',   action='store_true', help='Plot X asis in stops (log2(x))')
    parser.add_argument('--full-scale', type=int, metavar="<MAX DN>", default=None, help='Normalize X axes relative to full scale value')
    parser.add_argument('--every', type=int, metavar='<N>', default=1, help='pick every n `file after sorting')

# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=snr, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="Plot Sensor SNR per channel over a numbr of flat fields"
        )
