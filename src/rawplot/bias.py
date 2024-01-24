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
import sys
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
from astropy.io import fits

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

def fill_header(header, metadata, img):
    header['CHANNELS'] = (" ".join(metadata['channels']), 'Actual channels being used')
    header['SECTION'] = (str(metadata['roi']), 'NumPy format')
    header['EXPTINE'] = (float(img.exposure()), '[s]')
    header['BAYER'] = (img.cfa_pattern(), 'Color Filter Array Pattern')
    header['MAKER'] = (metadata['maker'], 'Camera Manufacturer')
    header['SENSOR'] = (metadata['camera'], 'Sensor')
    header['comment'] = f'Created with {os.path.basename(sys.argv[0])}'


def bias_create(args):
    channels = valid_channels(args.channels)
    if args.output_dir is None:
        output_dir = os.getcwd()
    log.info("Working with %d channels: %s", len(channels), channels)
    file_list = sorted(file_paths(args.input_dir, args.flat_filter))
    metadata = imageset_metadata(file_list[0], args.x0, args.y0, args.width, args.height, channels)
    img0 = RawImage(file_list[0])
    h, w =  img0.shape()
    h, w = h//2, w//2
    roi = img0.roi(args.x0, args.y0, args.width, args.height)
    # ROI from the fist image
    images = [RawImage(path) for path in file_list]
    log.info("Begin loading %d images into RAM with %s channels, %d x %d each", len(images), ",".join(channels), w, h)
    sections = [image.debayered(roi, channels).astype(float, copy=False) for image in images]
    log.info("Loaded %d images into RAM", len(sections))
    stack4d = np.stack(sections)
    log.info("Calculating Stack average")
    master_aver = np.mean(stack4d, axis=0)
    filename = f"{args.output_prefix}_aver_x{roi.x1:04d}_y{roi.y1:04d}_{metadata['cols']:04d}x{metadata['rows']:04d}_{''.join(channels)}.fit"
    path = os.path.join(output_dir, filename)
    log.info("Saving master bias in %s", path)
    hdu = fits.PrimaryHDU(master_aver)
    fill_header(hdu.header, metadata, img0)
    hdu.writeto(filename, overwrite=True)
    if args.stdev_map:
        stdev_map = np.std(stack4d, axis=0)
        filename = f"{args.output_prefix}_stdev_x{roi.x1:04d}_y{roi.y1:04d}_{metadata['cols']:04d}x{metadata['rows']:04d}_{''.join(channels)}.fit"
        path = os.path.join(output_dir, filename)
        log.info("Saving stdev map in %s", path)
        hdu = fits.PrimaryHDU(master_aver)
        fill_header(hdu.header, metadata, img0)
        hdu.writeto(filename, overwrite=True)
    if args.max_map:
        max_map = np.max(stack4d, axis=0)
        filename = f"{args.output_prefix}_max_x{roi.x1:04d}_y{roi.y1:04d}_{metadata['cols']:04d}x{metadata['rows']:04d}_{''.join(channels)}.fit"
        path = os.path.join(output_dir, filename)
        log.info("Saving max map in %s", path)
        hdu = fits.PrimaryHDU(master_aver)
        fill_header(hdu.header, metadata, img0)
        hdu.writeto(filename, overwrite=True)


   
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
   
    parser_create.add_argument('-i', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser_create.add_argument('-f', '--flat-filter', type=str, required=True, help='Flat Images filter, glob-style')
    parser_create.add_argument('-p', '--output-prefix', type=str, required=True, help='Output file prefix')
    parser_create.add_argument('-o', '--output-dir', type=vdir, default=None, help='Output directory defaults to current dir.')
    parser_create.add_argument('--stdev-map',  action='store_true', help='Also create standard deviation map')
    parser_create.add_argument('--max-map',  action='store_true', help='Also create max. pixel value map')
    parser_create.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser_create.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser_create.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser_create.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser_create.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
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
