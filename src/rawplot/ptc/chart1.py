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
from ..util.analyzer.image import ImageAnalyzer, ImagePairAnalyzer

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


def plot_noise_vs_signal(axes, signal, channel, ylabel, units, **kwargs):
    for key, noise in kwargs.items():
        axes.plot(signal, noise, marker='o', linewidth=0, label=f"$\\sigma_{{ {key.upper()} }}$")
    axes.set_title(fr'channel {channel}')
    axes.set_xscale('log', base=2)
    axes.set_yscale('log', base=2)
    units = "[DN]"
    title = f'Signal {units}'
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.set_xlabel(title)
    axes.set_ylabel(f'{ylabel} {units}')
    axes.legend()


def signal_and_total_noise_from(file_list, n_roi, channels, bias):
    file_list = file_list[::2]
    signal_list = list()
    noise_list = list()
    for path in file_list:
        analyzer = ImageAnalyzer(path, n_roi, channels, bias)
        analyzer.run()
        signal_list.append(analyzer.mean())
        noise_var = analyzer.variance()
        noise_list.append(noise_var)
        log.info("\u03C3\u00b2(total) for image %s = %s", analyzer.name(), noise_var)
    return np.stack(signal_list, axis=-1), np.stack(noise_list, axis=-1)

def read_and_shot_noise_from(file_list, n_roi, channels, bias):
    file_pairs = list(zip(file_list, file_list[1:]))[::2]
    noise_list = list()
    for path_a, path_b in file_pairs:
        analyzer = ImagePairAnalyzer(path_a, path_b, n_roi, channels, bias)
        analyzer.run()
        noise_var = analyzer.variance()
        noise_list.append(noise_var)
        log.info("\u03C3\u00b2(sh+rd) for image pair %s = %s", analyzer.names(), noise_var)
    return  np.stack(noise_list, axis=-1)


# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------


def ptc_chart1(args):
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    signal, total_noise_var = signal_and_total_noise_from(file_list, n_roi, channels, bias)
    shrd_noise_var = read_and_shot_noise_from(file_list, n_roi, channels, bias)
    fpn_var = total_noise_var - shrd_noise_var
    if args.bias_file is not None:
        analyzer = ImageAnalyzer(args.bias_file, n_roi, channels)
        analyzer.run()
        rdnoise_var = np.full_like(signal, analyzer.mean().reshape(len(channels),-1))
    else:
        rdnoise_var = np.full_like(signal, args.rd_noise**2)
    shot_var = shrd_noise_var - rdnoise_var
    total_noise = np.sqrt(total_noise_var)
    shot_noise = np.sqrt(shot_var)
    fpn_noise = np.sqrt(fpn_var)
    rd_noise = np.sqrt(rdnoise_var)

    log.info("TOTAL noises SHAPE is %s", total_noise.shape)
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
            plot_noise_vs_signal(axes[row][col], signal[i], channels[i], 'Noise', '[DN]',
                total=total_noise[i],
                fpn=fpn_noise[i],
                shot=shot_noise[i],
                rdnoise=rd_noise[i]
            )
    plt.show()
