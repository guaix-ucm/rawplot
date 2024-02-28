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

import re
import logging

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt

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

WAVELENGTH_REG_EXP = re.compile(r'(\w+)_(\d+)nm_g(\d+)_(\d+)_(\d+)_(\w+).jpg')

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# ------------------
# Auxiliary fnctions
# ------------------

def mpl_spectra_plot_loop(title, figsize, x, y, xtitle, ytitle, plot_func, channels, ylabel, **kwargs):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize, layout='tight')
    fig.suptitle(title)
    axes.set_xlabel(xtitle)
    axes.set_ylabel(f"{ytitle} [DN]")
    filters = kwargs.get('filters', None)
    for i in range(len(channels)):
        plot_func(axes, i, x, y, channels, **kwargs)
    if filters is not None:
        for filt in filters:
            axes.axvline(filt['wave'], linestyle=filt['style'], label=filt['label'])
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.legend()
    plt.show()


def plot_raw_spectral(axes, i, x, y, channels, **kwargs):
    wavelength = x[i]
    signal = y[i]
    if channels[i] == 'R':
        color = 'red'
        marker = 'o'
    elif  channels[i] == 'B':
        color = 'blue'
        marker = 'o'
    elif  channels[i] == 'Gr':
        color = (0, 0.5, 0)
        marker = '1'
    elif  channels[i] == 'Gb':
        color = (0, 0.25, 0)
        marker = '2'
    else:
        color = 'green'
    axes.plot(wavelength, signal,  marker=marker, color=color, linewidth=1, label=channels[i])
   


def signal_from(file_list, n_roi, channels, bias, dark, every=2):
    file_list = file_list[::every]
    N = len(file_list)
    signal_list = list()
    exptime_list = list()
    for i, path in enumerate(file_list, start=1):
        analyzer = ImageStatistics(path, n_roi, channels, bias, dark)
        analyzer.run()
        signal = analyzer.mean()
        signal_list.append(signal)
        exptime = np.full_like(signal, analyzer.loader().exptime())
        exptime_list.append(exptime)
        log.info("[%d/%d] \u03BC signal for image %s = %s", i, N, analyzer.name(), signal)
    return np.stack(exptime_list, axis=-1), np.stack(signal_list, axis=-1)

def get_wavelengths(file_list, channels):
    M = len(channels)
    data = list()
    for file in file_list:
        matchobj = WAVELENGTH_REG_EXP.search(file)
        if matchobj:
            item = { key:  matchobj.group(i) for i, key in enumerate(('tag', 'wave', 'gain', 'seq', 'exptime', 'filter'), start=1)}
            item['wave'] = int(item['wave'])
            item['gain'] = int(item['gain'])
            item['seq'] = int(item['seq'])
            item['exptime'] = int(item['exptime'])
            data.append(item)
    log.info("Matched %d files", len(data))
    result = np.array([item['wave'] for item in data])
    result = np.tile(result, M).reshape(M,len(data))
    log.info("Wavelengthss array shape is %s", result.shape)
    return result

# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------


def draft_spectrum(args):
    log.info(" === DRAFT SPECTRUM PLOT === ")
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    title = make_plot_title_from("Draft Spectral Response plot",metadata, roi)
    wavelength = get_wavelengths(file_list, channels)
    exptime, signal = signal_from(file_list, n_roi, channels, args.bias, args.dark, args.every)
    log.info("Exptime array shape is %s", exptime.shape)
    mpl_spectra_plot_loop(
        title    = title,
        figsize  = (12, 9),
        channels = channels,
        plot_func = plot_raw_spectral,
        xtitle = "Wavelength [nm]",
        ytitle = f"Signal",
        ylabel = "good",
        x  = wavelength,
        y  = signal,
        # Optional arguments tpo be handled by the plotting function
        filters=[ 
            {'label':'from BG38 to OG570',  'wave': 570, 'style': '--'}, 
            {'label':'from OG570 to RG830', 'wave': 830, 'style': '-.'},
        ] # where filters were changesd
    )

def complete_spectrum(args):
    log.info(" === COMPLETE SPECTRUM PLOT === ")

COMMAND_TABLE = {
    'draft': draft_spectrum,
    'complete': complete_spectrum, 
}

def spectral(args):
    command =  args.command
    func = COMMAND_TABLE[command]
    func(args)

# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):

    subparser = parser.add_subparsers(dest='command')

    parser_draft = subparser.add_parser('draft', help='Draft spectrum')
    parser_good  = subparser.add_parser('complete', help='Complete, reduced spectrum')

    parser_draft.add_argument('-i', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser_draft.add_argument('-f', '--image-filter', type=str, required=True, help='Images filter, glob-style (i.e. flat*, dark*)')
    parser_draft.add_argument('-x', '--x0', type=vfloat01,  help='Normalized ROI start point, x0 coordinate [0..1]')
    parser_draft.add_argument('-y', '--y0', type=vfloat01,  help='Normalized ROI start point, y0 coordinate [0..1]')
    parser_draft.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1] (default: %(default)s)')
    parser_draft.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1] (default: %(default)s) ')
    parser_draft.add_argument('-c','--channels', default=('R', 'Gr', 'Gb','B'), nargs='+',
                    choices=('R', 'Gr', 'Gb', 'G', 'B'),
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser_draft.add_argument('--every', type=int, metavar='<N>', default=1, help='pick every n `file after sorting')
    parser_draft.add_argument('-bi', '--bias',  type=vflopath,  help='Bias, either a single value for all channels or else a 3D FITS cube file (default: %(default)s)')
    parser_draft.add_argument('-dk', '--dark',  type=vfloat,  help='Dark count rate in DN/sec. (default: %(default)s)')

# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=spectral, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="Draft plot of sensor spectral response"
    )