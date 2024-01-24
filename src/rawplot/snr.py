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
from lica.rawimage import RawImage, imageset_metadata
from lica.mpl import plot_layout,  axes_reshape
from lica.misc import file_paths

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__

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


def measure_snr_from(path_a, path_b, x0, y0, width, height, channels):
    image_a = RawImage(path_a)
    image_b = RawImage(path_b)
    roi = image_a.roi(x0, y0, width, height) # Common to both
    # We must take the bias off to get valid results
    section_a = image_a.debayered(roi, channels).astype(float, copy=False) - np.array(image_a.black_levels(channels)).reshape(len(channels), 1, 1)
    section_b = image_b.debayered(roi, channels).astype(float, copy=False) - np.array(image_a.black_levels(channels)).reshape(len(channels), 1, 1)
    signal = np.mean((section_a + section_b), axis=(1,2)) / 2  
    noise = np.std((section_a - section_b), axis=(1,2)) / SQRT_2
    sn_ratio = signal / noise
    log.info("signal is %s, noise is %s", signal, noise)
    log.info("From %s, %s snr is %s", image_a.name(), image_b.name(), dict(zip(channels, sn_ratio.tolist())))
    return (signal, sn_ratio)


def measure_readout_noise_from(path_a, path_b, x0, y0, width, height, channels, iso):
    image_a = RawImage(path_a)
    image_b = RawImage(path_b)
    roi = image_a.roi(x0, y0, width, height) # Common to both
    # We must take the bias off to get valid results
    section_a = image_a.debayered(roi, channels).astype(float, copy=False)
    section_b = image_b.debayered(roi, channels).astype(float, copy=False)
    noise = np.std((section_a - section_b), axis=(1,2)) / SQRT_2
    iso = np.array([iso for ch in channels])
    log.info("From %s, %s , estimated noise at ISO gain %s is %s", image_a.name(), image_b.name(), iso, dict(zip(channels, noise.tolist())))
    return iso, noise

def measure_snr(input_dir, flat_filter, x0, y0, width, height, channels):
    file_list = sorted(file_paths(input_dir, flat_filter))
    file_pairs = list(zip(file_list, file_list[1:]))[::2]
    metadata = imageset_metadata(file_list[0], x0, y0, width, height, channels)
    signals =  list()
    snrs = list()
    for path_a, path_b in file_pairs:
        signal, sn_ratio = measure_snr_from(path_a, path_b, x0, y0, width, height, channels)
        signals.append(signal)
        snrs.append(sn_ratio)
    signals = np.array(signals).transpose()
    snrs = np.array(snrs).transpose()
    return signals, snrs, metadata

def measure_readout_noise(input_dir, bias_filter, x0, y0, width, height, channels):
    file_list = sorted(file_paths(input_dir, bias_filter))
    metadata = imageset_metadata(file_list[0], x0, y0, width, height, channels)
    file_pairs = list(zip(file_list, file_list[1:]))[::2]
    noises = list()
    isos = list()
    for path_a, path_b in file_pairs:
        iso, noise = measure_readout_noise_from(path_a, path_b, x0, y0, width, height, channels, metadata['iso'])
        noises.append(noise)
    noises = np.array(noises).transpose()
    isos = np.array(isos).transpose()
    return isos, noises

# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def snr(args):
    channels = valid_channels(args.channels)
    use_stops = args.stops
    full_scale = args.full_scale
    signals, snrs, metadata = measure_snr(args.input_dir, args.flat_filter, args.x0, args.y0, args.width, args.height, channels)
    if(args.bias_filter):
        isos, noises = measure_readout_noise(args.input_dir, args.bias_filter, args.x0, args.y0, args.width, args.height, channels)
    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(12, 9), layout='tight')
    fig.suptitle(f"SNR plot\n"
            f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}\n"
            f"Sensor: {metadata['camera']}, ISO: {metadata['iso']}\n"
            f"Color Plane Size: {metadata['rows']} rows x {metadata['cols']} cols\n" 
            f"ROI Section: {metadata['roi']}, {metadata['roi'].height()} rows x {metadata['roi'].width()} cols")
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
    parser.add_argument('-d', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser.add_argument('-f', '--flat-filter', type=str, required=True, help='Flat Images filter, glob-style')
    parser.add_argument('-b', '--bias-filter', type=str, default=None, help='Bias Images filter, glob-style')
    parser.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser.add_argument('-c','--channels', default=['R', 'G1', 'G2','B'], nargs='+',
                    choices=['R', 'G1', 'G2', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser.add_argument('--stops',   action='store_true', help='Plot X asis in stops (log2(x))')
    parser.add_argument('--full-scale', type=int, metavar="<MAX DN>", default=None, help='Normalize X axes relative to full scale value')

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
