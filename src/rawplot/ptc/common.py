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

from lica.validators import vdir, vfile, vfloat, vfloat01, valid_channels
from lica.raw.loader import ImageLoaderFactory, NormRoi

# ------------------------
# Own modules and packages
# ------------------------

from lica.raw.analyzer.image import ImageStatistics, ImagePairStatistics

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)


def ptc_parser_arguments_dn(parser):
    parser.add_argument('-i', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser.add_argument('-f', '--image-filter', type=str, required=True, help='Images filter, glob-style (i.e. flat*, dark*)')
    parser.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser.add_argument('-rd','--rd-noise', type=vfloat, metavar='<\u03C3>', default=1.0, help='Readout noise [DN] (default: %(default)s)')
    parser.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser.add_argument('--every', type=int, metavar='<N>', default=1, help='pick every n `file after sorting')
    group0 = parser.add_mutually_exclusive_group(required=False)
    group0.add_argument('-bl', '--bias-level', type=vfloat, default=None, help='Bias level, common for all channels (default: %(default)s)')
    group0.add_argument('-bf', '--bias-file',  type=vfile, default=None, help='Bias image (3D FITS cube) (default: %(default)s)')



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
        analyzer = ImageStatistics(path, n_roi, channels, bias)
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
        analyzer = ImagePairStatistics(path_a, path_b, n_roi, channels, bias)
        analyzer.run()
        noise_var = analyzer.variance()
        noise_list.append(noise_var)
        log.info("\u03C3\u00b2(sh+rd) for image pair %s = %s", analyzer.names(), noise_var)
    return  np.stack(noise_list, axis=-1)
