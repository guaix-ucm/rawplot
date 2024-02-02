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
    return signal, total_noise_var, fpn_corrected_noise_var, fixed_pattern_noise_var, shot_noise_var,
