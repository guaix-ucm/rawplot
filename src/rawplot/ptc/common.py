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
from lica.raw.analyzer.image import ImagePairStatistics

# ------------------------
# Own modules and packages
# ------------------------

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)


def signal_and_noise_variances_from(file_list, n_roi, channels, bias):
    file_pairs = list(zip(file_list, file_list[1:]))[::2]
    N = len(file_pairs)
    total_noise_list = list() # Only from the first images of the pairs
    signal_list = list()      # Only from the first images of the pairs
    fpn_corrected_noise_list = list()
    for i, (path_a, path_b) in enumerate(file_pairs, start=1):
        analyzer = ImagePairStatistics(path_a, path_b, n_roi, channels, bias)
        analyzer.run()
        signal = analyzer.mean()
        total_noise_var = analyzer.variance()
        fpn_corrected_noise_var = analyzer.adj_pair_variance() # Already corrected by 1/2 factor
        signal_list.append(signal)
        total_noise_list.append(total_noise_var)
        fpn_corrected_noise_list.append(fpn_corrected_noise_var)
        log.info("[%d/%d] \u03C3\u00b2(total)     for image %s = %s", i, N, analyzer.name(), total_noise_var)
        log.info("[%d/%d] \u03C3\u00b2(total-fpn) for image pair %s = %s",  i, N, analyzer.names(), fpn_corrected_noise_var)
    return  np.stack(signal_list, axis=-1), np.stack(total_noise_list, axis=-1), np.stack(fpn_corrected_noise_list, axis=-1)


def signal_and_noise_variances(file_list, n_roi, channels, bias, read_noise):
    signal, total_noise_var, fpn_corrected_noise_var = signal_and_noise_variances_from(file_list, n_roi, channels, bias)
    fixed_pattern_noise_var = total_noise_var - fpn_corrected_noise_var
    shot_noise_var = fpn_corrected_noise_var - read_noise**2
    return signal, total_noise_var, fpn_corrected_noise_var, shot_noise_var, fixed_pattern_noise_var

def ptc_parser_arguments_dn(parser):
    parser.add_argument('-i', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser.add_argument('-f', '--image-filter', type=str, required=True, help='Images filter, glob-style (i.e. flat*, dark*)')
    parser.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser.add_argument('-rd','--read-noise', type=vfloat, metavar='<\u03C3>', default=0.0, help='Read noise [DN] (default: %(default)s)')
    parser.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser.add_argument('--every', type=int, metavar='<N>', default=1, help='pick every n `file after sorting')
    group0 = parser.add_mutually_exclusive_group(required=False)
    group0.add_argument('-bl', '--bias-level', type=vfloat, default=None, help='Bias level, common for all channels (default: %(default)s)')
    group0.add_argument('-bf', '--bias-file',  type=vfile, default=None, help='Bias image (3D FITS cube) (default: %(default)s)')
    parser.add_argument('--log2',  action='store_true', help='Display plot using log2 instead of log10 scale')
    parser.add_argument('--p-fpn', type=vfloat01, metavar='<p>', default=None, help='Fixed Pattern Noise Percentage factor [0..1] (default: %(default)s)')
    parser.add_argument('-gn','--gain', type=vfloat, metavar='<g>', default=None, help='Gain [e-/DN] (default: %(default)s)')
    parser.add_argument('-ph','--physical-units',  action='store_true', help='Display in [e-] physical units instead of [DN]. Requires --gain')