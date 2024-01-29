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
from lica.misc import file_paths
from lica.validators import vdir, vfile, vfloat, vfloat01, valid_channels
from lica.raw.loader import ImageLoaderFactory, SimulatedDarkImage, NormRoi
from lica.raw.analyzer.image import ImageStatistics

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.mpl.plot import mpl_main_plot_loop
from .util.common import common_list_info, bias_from, make_plot_title_from

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

def check_physical(args):
    gain = args.gain
    phys = args.physical_units
    if gain is None and phys:
        raise ValueError("Can'use physycal units [-e] if --gain is not set")
    units = r"$[e^{-}]$" if gain is not None and phys else "[DN]"
    return units, gain, phys


def fit_estimator(estimator, exptime, signal, channel):
    T = exptime.reshape(-1,1)
    fitted = estimator.fit(T, signal)
    score = estimator.score(T, signal)
    log.info("[%s] %s fitting score is %f. y=%.4f*x%+.4f", channel, estimator.__class__.__name__, score,  estimator.coef_[0], estimator.intercept_)
    predicted = estimator.predict(T)
    return score, estimator.coef_[0], estimator.intercept_

# The saturation analysis is made on the assumption that the measured noise
# should be dominated by shot noise if the ROI is small
# We are neglecting FPN, which would make the noise even larger anyway.
# In any case, beyornd the saturation point, the noise decreases 
# and this is what we are looking for.
# So we compute the threshold noise / sqrt(signal)
# and discard values below a certain threshold (0.5 seems a reasonable compromise)
def saturation_analysis(exptime, signal, noise, channels, threshold=0.5):
    estimated_poisson_noise = np.sqrt(signal)
    ratio =  noise / estimated_poisson_noise
    sat_exptime = np.full_like(exptime, np.nan)
    sat_signal = np.full_like(signal, np.nan)
    # They are not matrices as each channel may have its own row length
    good_exptime_list=list()
    good_signal_list=list()
    sat_exptime_list=list()
    sat_signal_list=list()
    N = len(channels)
    for ch in range(N):
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


def signal_exptime_and_total_noise_from(file_list, n_roi, channels, bias, every=2):
    file_list = file_list[::every]
    N = len(file_list)
    signal_list = list()
    noise_list = list()
    exptime_list = list()
    for i, path in enumerate(file_list, start=1):
        analyzer = ImageStatistics(path, n_roi, channels, bias)
        analyzer.run()
        signal = analyzer.mean()
        signal_list.append(signal)
        exptime = np.full_like(signal, analyzer.loader().exposure())
        exptime_list.append(exptime)
        noise = analyzer.std()
        noise_list.append(noise)
        log.info("[%d/%d] \u03C3\u00b2(total) for image %s = %s", i, N, analyzer.name(), noise)
    return np.stack(exptime_list, axis=-1), np.stack(signal_list, axis=-1), np.stack(noise_list, axis=-1)


def plot_linear_equation(axes, xdata, ydata, slope, intercept, xlabel='x', ylabel='y'):
    angle = math.atan(slope)*(180/math.pi)
    x0 = np.min(xdata); x1 = np.max(xdata)
    y0 = np.min(ydata); y1 = np.max(ydata)
    x = x0 + 0.35*(x1-x0)
    y = y0 + 0.45*(y1-y0)
    text = f"${ylabel} = {slope:.2f}{xlabel}{intercept:+.2f}$"
    axes.text(x, y, text,
        rotation_mode='anchor',
        rotation=angle,
        transform_rotates_text=True,
        ha='left', va='top'
    )


def plot_linearity(axes, i, x, y, xtitle, ytitle, ylabel, channels,  **kargs):
    exptime = x[i]
    signal = y[i]
    good_exptime = kargs['good_exptime'][i]
    good_signal = kargs['good_signal'][i]
    sat_exptime = kargs['sat_exptime'][i]
    sat_signal = kargs['sat_signal'][i]
    estimator = TheilSenRegressor(random_state=42,  fit_intercept=True)
    score, slope, intercept = fit_estimator(estimator, good_exptime, good_signal, channels[i])
    fit_signal = estimator.predict(exptime.reshape(-1,1)) # For the whole range
    text = rf"Theil-Sen fit: $R^2 = {score:.2f}$"
    axes.plot(exptime, signal,  marker='o', linewidth=0, label=ylabel)
    axes.plot(sat_exptime, sat_signal,  marker='o', linewidth=0, label="saturated")
    axes.plot(exptime, fit_signal, label=text)
    plot_linear_equation(axes, exptime, fit_signal, slope, intercept, xlabel='t', ylabel='S(t)')
    axes.set_xlabel(xtitle)
    axes.set_ylabel(ytitle)
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.legend()

# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------


def linearity(args):
    log.info(" === LINEARITY PLOT === ")
    units, gain, phys = check_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    bias = bias_from(args)
    exptime, signal, noise = signal_exptime_and_total_noise_from(file_list, n_roi, channels, bias)
    log.info("estimated signal & noise for %s points", exptime.shape)
    good_exptime, good_signal, sat_exptime, sat_signal = saturation_analysis(exptime, signal, noise, channels, 0.5)
    if gain and phys:
        signal *= gain
        good_signal *= gain
        sat_signal *= gain
    title = make_plot_title_from("Linearity plot",metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
        channels = channels,
        plot_func = plot_linearity,
        xtitle = "Exposure time [s]",
        ytitle = f"Signal {units}",
        ylabel = "good",
        x  = exptime,
        y  = signal,
        # Optional arguments tpo be handled by the plotting function
        good_exptime = good_exptime,
        good_signal  = good_signal,
        sat_exptime  = sat_exptime,
        sat_signal   = sat_signal
    )

# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):
    parser.add_argument('-i', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser.add_argument('-f', '--image-filter', type=str, required=True, help='Images filter, glob-style (i.e. flat*, dark*)')
    parser.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1] (default: %(default)s)')
    parser.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1] (default: %(default)s) ')
    parser.add_argument('-c','--channels', default=('R', 'Gr', 'Gb','B'), nargs='+',
                    choices=('R', 'Gr', 'Gb', 'G', 'B'),
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser.add_argument('--every', type=int, metavar='<N>', default=1, help='pick every n `file after sorting')
    group0 = parser.add_mutually_exclusive_group(required=False)
    group0.add_argument('-bl', '--bias-level', type=vfloat, default=None, help='Bias level, common for all channels (default: %(default)s)')
    group0.add_argument('-bf', '--bias-file',  type=vfile, default=None, help='Bias image (3D FITS cube) (default: %(default)s)')
    parser.add_argument('-gn','--gain', type=vfloat, metavar='<g>', default=None, help='Gain [e-/DN] (default: %(default)s)')
    parser.add_argument('-ph','--physical-units',  action='store_true', help='Display in [-e] physical units instead of [DN]. Requires --gain')

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
