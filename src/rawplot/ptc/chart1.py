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

from .._version import __version__
from ..util.mpl.plot import plot_layout, axes_reshape
from ..util.common import preliminary_tasks

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


def plot_noise_vs_signal(axes, signal, noise, channel, ylabel, units):
    axes.set_title(fr'channel {channel}')
    axes.set_xscale('log')
    axes.set_yscale('log')
    units = "[DN]"
    title = f'Signal {units}'
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.set_xlabel(title)
    axes.plot(signal, noise,   marker='o', linewidth=0)
    axes.set_ylabel(f'{ylabel} {units}')


def measure_total_noise_for(file_list, n_roi, channels, bias):
    signal_list = list()
    total_noise_list = list()
    factory =  ImageLoaderFactory()
    for path in file_list:
        image = factory.image_from(path, n_roi, channels)
        pixels = image.load().astype(float, copy=False) - bias
        signal = np.mean(pixels, axis=(1,2))
        noise = np.std(pixels, axis=(1,2))
        log.info("From %s, signal %s, noise is %s", image.name(), dict(zip(channels, signal.tolist())), dict(zip(channels, noise.tolist())))
        signal_list.append(signal)
        total_noise_list.append(noise)
    signals = np.array(signal_list).transpose()
    noises = np.array(total_noise_list).transpose()
    log.info("TOTAL noises SHAPE is %s", noises.shape)
    return signals, noises


# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def ptc_chart1(args):
    file_list, roi, n_roi, channels, metadata = preliminary_tasks(args)
    bias = ImageLoaderFactory().image_from(args.master_bias, n_roi, channels).load()
    signal, noise = measure_total_noise_for(file_list, n_roi, channels, bias)

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
            plot_noise_vs_signal(axes[row][col], signal[i], noise[i], channels[i], 'Total noise', '[DN]')
    plt.show()
