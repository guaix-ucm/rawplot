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
import functools

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.linear_model import  TheilSenRegressor, LinearRegression

from lica.cli import execute
from lica.validators import vdir, vfloat01, valid_channels
from lica.rawimage import RawImage, imageset_metadata
from lica.mpl import plot_layout, axes_reshape, plot_linear_equation
from lica.misc import file_paths

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__

# ----------------
# Module constants
# ----------------

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# ------------------
# Auxiliary fnctions
# ------------------

def fit_estimator(estimator, exptime, signal, channel):
    T = exptime.reshape(-1,1)
    fitted = estimator.fit(T, signal)
    score = estimator.score(T, signal)
    log.info("[%s] %s fitting score is %f. y=%.4f*x%+.4f", channel, estimator.__class__.__name__, score,  estimator.coef_[0], estimator.intercept_)
    predicted = estimator.predict(T)
    return score, estimator.coef_[0], estimator.intercept_


# For single image in all channels at once
# retuns 1D arrays, one row for each color channel
def compute_signal_noise(path, x0, y0, width, height, channels):
    image = RawImage(path)
    roi = image.roi(x0, y0, width, height)
    section = image.debayered(roi, channels).astype(np.float32, copy=False) - np.array(image.black_levels(channels)).reshape(len(channels), 1, 1)
    exptime = float(image.exif()['exposure'])
    exptime = np.array([exptime for ch in channels])
    signal = np.mean(section, axis=(1,2))
    noise = np.std(section, axis=(1,2))
    log.info("Image: %s signal: %s, exptime: %s", image.name(), signal, exptime)
    return exptime, signal, noise

# For an image list
# returns 2D arrays [channel, datapoints ifor each channel]
def compute_signal_noise_for(file_list, x0, y0, width, height, channels):
    signal_list = list()
    noise_list = list()
    exp_list =list()
    for path in file_list:
        exptime, signal, noise = compute_signal_noise(path, x0, y0, width, height, channels)
        signal_list.append(signal)
        noise_list.append(noise)
        exp_list.append(exptime)
    noise = np.array(noise_list).transpose()
    signal = np.array(signal_list).transpose()
    exptime = np.array(exp_list).transpose()
    return exptime, signal, noise


# The saturation analysis is made on the assupotion that the measured noise
# should be dominated by shot noise, whose sdtdev = sqrt(signal)
# So we compute the threshold noise / sqrt(signal)
# and discard values below a certain threshold (0.5 seems a reasonable compromise)
def saturation_analysis(exptime, signal, noise, channels, threshold=0.5):
    estimated_poisson_noise = np.sqrt(signal)
    ratio =  noise / estimated_poisson_noise
    sat_exptime = np.full_like(exptime, np.nan)
    sat_signal = np.full_like(signal, np.nan)
    # They are not matrices as each channel may have its own length
    good_exptime_list=list()
    good_signal_list=list()
    sat_exptime_list=list()
    sat_signal_list=list()
    for ch in range(len(channels)):
        good_exptime = exptime[ch].copy()
        good_signal = signal[ch].copy()
        sat_exptime = np.full_like(exptime[ch], np.nan)
        sat_signal = np.full_like(signal[ch], np.nan)
        for i in range(ratio.shape[1]):
            if ratio[ch,i] < threshold:
                sat_signal[i] = good_signal[i]
                sat_exptime[i] = good_exptime[i]
                # Mark with non valid values
                good_signal[i] = np.nan
                good_exptime[i] = np.nan
        # Wipe the NaNs out, thus making the arrays smaller
        # This is needed for the least squares fitting routine
        good_exptime = good_exptime[~np.isnan(good_exptime)]
        sat_exptime = sat_exptime[~np.isnan(sat_exptime)]
        log.info("[%s]. Good signal for only %d points", channels[ch], good_exptime.shape[0])
        log.info("[%s]. Saturated signal in %d points", channels[ch], sat_exptime.shape[0])
        good_exptime_list.append(good_exptime)
        sat_exptime_list.append(sat_exptime)
        good_signal_list.append(good_signal[~np.isnan(good_signal)])
        sat_signal_list.append(sat_signal[~np.isnan(sat_signal)])
    return good_exptime_list, good_signal_list, sat_exptime_list, sat_signal_list


def plot_linearity(axes, exptime, signal, good_exptime, good_signal, sat_exptime, sat_signal, channel):
    estimator = TheilSenRegressor(random_state=42,  fit_intercept=True)
    score, slope, intercept = fit_estimator(estimator, good_exptime, good_signal, channel)
    fit_signal = estimator.predict(exptime.reshape(-1,1)) # For the whole range
    text = rf"Theil-Sen fit: $R^2 = {score:.2f}$"
    axes.plot(exptime, signal,  marker='o', linewidth=0, label="data")
    axes.plot(sat_exptime, sat_signal,  marker='o', linewidth=0, label="saturated")
    axes.plot(exptime, fit_signal, label=text)
    plot_linear_equation(axes, exptime, fit_signal, slope, intercept, xlabel='t', ylabel='S(t)')
    axes.set_xlabel('Exposure time [s]')
    axes.set_ylabel('Mean Signal [DN]')
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.legend()

# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def linearity(args):
    channels = valid_channels(args.channels)
    log.info("Working with %d channels: %s", len(channels), channels)
    # Take the A files only by decimating by a  user specified 'every' factor
    file_list = sorted(file_paths(args.input_dir, args.flat_filter))[::args.every]
    metadata = imageset_metadata(file_list[0], args.x0, args.y0, args.width, args.height, channels)
    exptime, signal, noise = compute_signal_noise_for(file_list, args.x0, args.y0, args.width, args.height, channels)
    log.info("estimated signal & noise for %d points", exptime.shape[1])
    good_exptime, good_signal, sat_exptime, sat_signal = saturation_analysis(exptime, signal, noise, channels, 0.5)
    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(12, 9), layout='tight')
    fig.suptitle(f"Linearity plot\n"
            f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}\n"
            f"Color Plane Size: {metadata['rows']} rows x {metadata['cols']} cols\n" 
            f"ROI Section: {metadata['roi']}, {metadata['roi'].height()} rows x {metadata['roi'].width()} cols")
    axes = axes_reshape(axes, channels)
    for row in range(0,display_rows):
        for col in range(0,display_cols):
            i = 2*row+col
            if len(channels) == 3 and row == 1 and col == 1: # Skip the empty slot in 2x2 layout with 3 items
                axes[row][col].set_axis_off()
                break
            plot_linearity(axes[row][col], exptime[i], signal[i], good_exptime[i], good_signal[i], sat_exptime[i], sat_signal[i], channels[i])
    plt.show()


# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):
    parser.add_argument('-d', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser.add_argument('-f', '--flat-filter', type=str, required=True, help='Flat Images filter, glob-style')
    parser.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser.add_argument('--every', type=int, metavar='<N>', default=1, help='pick every n `file after sorting')

# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=linearity, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="Plot sensor exposure linearity per channel"
        )
