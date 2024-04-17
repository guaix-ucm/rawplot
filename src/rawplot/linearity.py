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

import logging

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import  TheilSenRegressor, LinearRegression

from lica.cli import execute
from lica.misc import file_paths
from lica.validators import vdir, vfile, vfloat, vfloat01, vflopath
from lica.raw.loader import ImageLoaderFactory
from lica.raw.analyzer.image import ImageStatistics

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.mpl.plot import mpl_main_plot_loop
from .util.common import common_list_info, make_plot_title_from, assert_physical

# ----------------
# Module constants
# ----------------

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# -----------------
# Matplotlib styles
# -----------------

# Load global style sheets
plt.style.use("rawplot.resources.global")

# ------------------
# Auxiliary fnctions
# ------------------

def fit(exptime, signal, channels):
    fit_params = list()
    estimator = TheilSenRegressor(random_state=42,  fit_intercept=True)
    for i, ch in enumerate(channels):
        T = exptime[i].reshape(-1,1)
        fitted = estimator.fit(T, signal[i])
        score = estimator.score(T, signal[i])
        log.info("[%s] %s fitting score is %f. y=%.4f*x%+.4f", ch, estimator.__class__.__name__, score,  estimator.coef_[0], estimator.intercept_)
        intercept = estimator.intercept_
        slope = estimator.coef_[0]
        fit_params.append({'score': score, 'slope': estimator.coef_[0], 'intercept': estimator.intercept_})
    return fit_params

# The saturation analysis is made on the assumption that the measured noise
# should be dominated by shot noise if the ROI is small
# We are neglecting FPN, which would make the noise even larger anyway.
# In any case, beyornd the saturation point, the noise decreases 
# and this is what we are looking for.
# So we compute the threshold noise / sqrt(signal)
# and discard values below a certain threshold (0.5 seems a reasonable compromise)
def saturation_analysis(exptime, signal, noise, channels, threshold):
    estimated_poisson_noise = np.sqrt(signal)
    ratio =  noise / estimated_poisson_noise
    bad_mask = ratio < threshold
    good_mask = ratio >= threshold
    good_exptime_list=list()
    good_signal_list=list()
    sat_exptime_list=list()
    sat_signal_list=list()
    for i, ch in enumerate(channels):
        b_msk = bad_mask[i]
        g_msk = good_mask[i]
        sat_exptime = exptime[i][b_msk]
        sat_signal = signal[i][b_msk]
        good_exptime = exptime[i][g_msk]
        good_signal = signal[i][g_msk]
        log.info("[%s]. Good signal for only %d points", ch, good_exptime.shape[0])
        good_exptime_list.append(good_exptime)
        sat_exptime_list.append(sat_exptime)
        good_signal_list.append(good_signal)
        sat_signal_list.append(sat_signal)
    return good_exptime_list, good_signal_list, sat_exptime_list, sat_signal_list


def signal_exptime_and_total_noise_from(file_list, n_roi, channels, bias, dark, every=2):
    file_list = file_list[::every]
    N = len(file_list)
    signal_list = list()
    noise_list = list()
    exptime_list = list()
    for i, path in enumerate(file_list, start=1):
        analyzer = ImageStatistics.from_path(path, n_roi, channels, bias, dark)
        analyzer.run()
        signal = analyzer.mean()
        signal_list.append(signal)
        exptime = np.full_like(signal, analyzer.loader().exptime())
        exptime_list.append(exptime)
        noise = analyzer.std()
        noise_list.append(noise)
        log.info("[%d/%d] \u03C3\u00b2(total) for image %s = %s", i, N, analyzer.name(), noise)
    return np.stack(exptime_list, axis=-1), np.stack(signal_list, axis=-1), np.stack(noise_list, axis=-1)


def plot_fitted(axes, fitted, fitted_x, fitted_y):
    '''All graphical elements for a fitting line'''
    slope = fitted['slope']
    score = fitted['score']
    intercept = fitted['intercept']
    label = rf"$S(t)$ (fitted)"
    P0 = (0, intercept) 
    P1 = ( -intercept/slope)
    axes.plot(fitted_x, fitted_y, marker='o', linewidth=0, label=r"$S(t)$ (to fit)")
    axes.axline(P0, slope=slope, linestyle=':', label=label)
    text = "\n".join((fr"$r^2 = {score:.3f}$", rf"$S(t) = {slope:0.2f}t{intercept:+0.2f}$"))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes.text(0.5, 0.2, text, transform=axes.transAxes, va='top', bbox=props)


def plot_linearity(axes, i, x, y, xtitle, ytitle, ylabel, channels, **kwargs):
    exptime = x[i]
    signal = y[i]
    phys = kwargs.get('phys', False)
    units = r"$[e^{-}]$" if phys else "[DN]" 
    good_exptime = kwargs['good_exptime'][i]
    good_signal = kwargs['good_signal'][i]
    sat_exptime = kwargs['sat_exptime'][i]
    sat_signal = kwargs['sat_signal'][i]
    fitted = kwargs['fitted'][i]
    plot_fitted(axes, fitted, good_exptime, good_signal)
    axes.plot(sat_exptime, sat_signal,  marker='o', linewidth=0, label="saturated")
    axes.set_xlabel(xtitle)
    axes.set_ylabel(f"{ytitle} {units}")
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.legend()

# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------


def linearity(args):
    log.info(" === LINEARITY PLOT === ")
    assert_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    exptime, signal, noise = signal_exptime_and_total_noise_from(file_list, n_roi, channels, args.bias, args.dark)
    log.info("estimated signal & noise for %s points", exptime.shape)
    good_exptime, good_signal, sat_exptime, sat_signal = saturation_analysis(exptime, signal, noise, channels, threshold=0.5)
    if args.gain and args.physical_units:
        signal = [args.gain * s for s in signal]
        good_signal = [args.gain * s for s in good_signal]
        sat_signal = [args.gain * s for s in sat_signal]
    title = make_plot_title_from("Linearity plot",metadata, roi)
    fit_params = fit(good_exptime, good_signal, channels)
    mpl_main_plot_loop(
        title    = title,
        channels = channels,
        plot_func = plot_linearity,
        xtitle = "Exposure time [s]",
        ytitle = f"Signal",
        ylabel = "good",
        x  = exptime,
        y  = signal,
        # Optional arguments tpo be handled by the plotting function
        good_exptime = good_exptime,
        good_signal  = good_signal,
        sat_exptime  = sat_exptime,
        sat_signal   = sat_signal,
        fitted = fit_params,
        phys = args.physical_units,
    )

# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):
    parser.add_argument('-i', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser.add_argument('-f', '--image-filter', type=str, required=True, help='Images filter, glob-style (i.e. flat*, dark*)')
    parser.add_argument('-x', '--x0', type=vfloat01,  help='Normalized ROI start point, x0 coordinate [0..1]')
    parser.add_argument('-y', '--y0', type=vfloat01,  help='Normalized ROI start point, y0 coordinate [0..1]')
    parser.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1] (default: %(default)s)')
    parser.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1] (default: %(default)s) ')
    parser.add_argument('-c','--channels', default=('R', 'Gr', 'Gb','B'), nargs='+',
                    choices=('R', 'Gr', 'Gb', 'G', 'B'),
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser.add_argument('--every', type=int, metavar='<N>', default=1, help='pick every n `file after sorting')
    parser.add_argument('-bi', '--bias',  type=vflopath,  help='Bias, either a single value for all channels or else a 3D FITS cube file (default: %(default)s)')
    parser.add_argument('-dk', '--dark',  type=vfloat,  help='Dark count rate in DN/sec. (default: %(default)s)')
    parser.add_argument('-gn','--gain', type=vfloat, metavar='<g>',  help='Gain [e-/DN] (default: %(default)s)')
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
