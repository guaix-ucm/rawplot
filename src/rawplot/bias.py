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

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.cli import execute
from .util.validators import vdir, vfloat01, valid_channels
from .util.rawimage import RawImage, imageset_metadata
from .util.mpl import plot_layout, axes_reshape, plot_linear_equation
from .util.misc import file_paths

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


def bias_create(args):
    channels = valid_channels(args.channels)
    log.info("Working with %d channels: %s", len(channels), channels)
    file_list = sorted(file_paths(args.input_dir, args.flat_filter))
    metadata = imageset_metadata(file_list[0], args.x0, args.y0, args.width, args.height, channels)
    _img = RawImage(file_list[0])
    roi = _img.roi(args.x0, args.y0, args.width, args.height)
    # ROI from the fist image
    images = [RawImage(path) for path in file_list]
    log.info("Begin loading %d images into RAM", len(images))
    sections = [image.debayered(roi, channels).astype(float, copy=False) for image in images]
    log.info("Loaded %d images into RAM", len(sections))
    stack4d = np.stack(sections)
    log.info("Stack4d shape is %s", stack4d.shape)
    log.info("Calculating Stack average")
    #aver = np.mean(stack4d, axis=(0,1))
    #log.info("Average shape is %s", aver.shape)



    

# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def bias(args):   
    if  args.command == 'create':
        bias_create(args)



# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):

    subparser = parser.add_subparsers(dest='command')

    parser_create = subparser.add_parser('create', help='Create a 3D FITS cube master bias')
   
    parser_create.add_argument('-d', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser_create.add_argument('-f', '--flat-filter', type=str, required=True, help='Flat Images filter, glob-style')
    parser_create.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser_create.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser_create.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser_create.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser_create.add_argument('-c','--channels', default=['R', 'G1', 'G2','B'], nargs='+',
                    choices=['R', 'G1', 'G2', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')

# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=bias, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="Master bias creation and analysis"
        )
