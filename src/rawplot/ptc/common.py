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

from sklearn.linear_model import  TheilSenRegressor, LinearRegression

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
    return signal, total_noise_var, fpn_corrected_noise_var, fixed_pattern_noise_var, shot_noise_var


def fit(X, Y, x0, x1, channels, loglog=False):
    estimator = TheilSenRegressor(random_state=42,  fit_intercept=True)
    #estimator = LinearRegression(fit_intercept=True)
    mask = np.logical_and(X >= x0, X <= x1)
    if loglog:
        X = np.log(X)
        Y = np.log(Y)
    fit_params = list()
    for i, ch in enumerate(channels):
        m = mask[i]
        sub_x = X[i][m]
        sub_y = Y[i][m]
        sub_x = sub_x.reshape(-1,1)
        estimator.fit(sub_x, sub_y)
        score = estimator.score(sub_x, sub_y)
        log.info("[%s] %s fitting score is %f. y=%.4f*x%+.4f", ch, estimator.__class__.__name__, score,  estimator.coef_[0], estimator.intercept_)
        fit_params.append({'score': score, 'slope': estimator.coef_[0], 'intercept': estimator.intercept_, 
            'x': sub_x, 'y': sub_y, 'mask': mask})
    return fit_params