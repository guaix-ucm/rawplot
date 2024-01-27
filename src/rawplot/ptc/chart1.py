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


def measure_signal_for(file_list, n_roi, channels, bias):
    file_pairs = list(zip(file_list, file_list[1:]))[::2]
    signal_list = list()
    factory =  ImageLoaderFactory()
    for path_a, path_b in file_pairs:
        image_a = factory.image_from(path_a, n_roi, channels)
        image_b = factory.image_from(path_b, n_roi, channels)
        pixels_a = image_a.load().astype(float, copy=False) - bias
        pixels_b = image_b.load().astype(float, copy=False) - bias
        signal = np.mean(pixels_a + pixels_b, axis=(1,2)) / 2
        log.info("Average signal for image pair (%s, %s) = %s", image_a.name(), image_b.name(), signal)
        signal_list.append(signal)
    return np.stack(signal_list, axis=-1)

def measure_total_noise_for(file_list, n_roi, channels, bias):
    file_list = file_list[::2] # get first image of the pair
    noise_list = list()
    factory =  ImageLoaderFactory()
    for path in file_list:
        image = factory.image_from(path, n_roi, channels)
        pixels = image.load().astype(float, copy=False) - bias
        noise_var = np.var(pixels, axis=(1,2))
        log.info("\u03C3\u00b2(total) for %s = %s", image.name(), noise_var)
        noise_list.append(noise_var)
    return np.stack(noise_list, axis=-1)

def measure_shot_plus_rdnoise_for(file_list, n_roi, channels, bias):
    file_pairs = list(zip(file_list, file_list[1:]))[::2]
    noise_list = list()
    factory =  ImageLoaderFactory()
    for path_a, path_b in file_pairs:
        image_a = factory.image_from(path_a, n_roi, channels)
        image_b = factory.image_from(path_b, n_roi, channels)
        pixels_a = image_a.load().astype(float, copy=False) - bias
        pixels_b = image_b.load().astype(float, copy=False) - bias
        noise_var = np.var(pixels_a - pixels_b, axis=(1,2)) / 2
        log.info("\u03C3\u00b2(sh+rd) for image pair (%s, %s) = %s", image_a.name(), image_b.name(), noise_var)
        noise_list.append(noise_var)
    return np.stack(noise_list, axis=-1)

def measure_fpn_noise(total_noise_var, shrd_noise_var):
    diff = total_noise_var - shrd_noise_var
    log.info("\u03C3\u00b2(fpn) shape is %s". diff.shape)
    return diff

# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def ptc_chart1(args):
    file_list, roi, n_roi, channels, metadata = preliminary_tasks(args)
    bias = ImageLoaderFactory().image_from(args.master_bias, n_roi, channels).load()
    signal = measure_signal_for(file_list, n_roi, channels, bias)
    total_noise_var = measure_total_noise_for(file_list, n_roi, channels, bias)
    shrd_noise_var = measure_shot_plus_rdnoise_for(file_list, n_roi, channels, bias)
    
    fpn_var = total_noise_var - shrd_noise_var
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
