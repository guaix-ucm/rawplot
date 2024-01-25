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
from lica.validators import vdir, vfile, vfloat01, valid_channels
from lica.raw import ImageLoaderFactory, NRect, CHANNELS
from lica.misc import file_paths

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.mpl.plot import plot_layout, plot_cmap, plot_edge_color,  plot_image, plot_histo, axes_reshape

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


def fill_header(header, metadata, img, comment=None):
    header['CHANNELS'] = (metadata['channels'], 'Channel stored order')
    header['SECTION'] = (metadata['roi'], 'NumPy format')
    header['EXPTIME'] = (float(img.exposure()), '[s]')
    header['MAKER'] = (metadata['maker'], 'Manufacturer')
    header['CAMERA'] = (metadata['camera'], 'Sensor')
    header['ISO'] = (metadata['iso'], 'Sensor')
    header['DATE-OBS'] = (metadata['datetime'], 'When taken')
    if comment:
         header['comment'] = comment
    header['comment'] = f'Created with {os.path.basename(sys.argv[0])}'


def output_path(output_dir, prefix, metadata, roi, tag):
    width = metadata['width'] // 2
    height = metadata['height'] // 2
    channels = metadata['channels'].replace(' ', '_')
    filename = f"{prefix}_{tag}_x{roi.x0:04d}_y{roi.y0:04d}_{width:04d}x{height:04d}_{channels}.fit"
    return os.path.join(output_dir, filename)


def bias_create(args):
    channels = CHANNELS
    if args.output_dir is None:
        output_dir = os.getcwd()
    n_roi = NRect(args.x0, args.y0, args.width, args.height)
    file_list = sorted(file_paths(args.input_dir, args.flat_filter))
    factory = ImageLoaderFactory()
    img0 = factory.image_from(file_list[0], n_roi, channels)
    roi = img0.roi()
    metadata = img0.metadata()
    h, w = metadata['width'] // 2, metadata['height'] // 2

    # ROI from the fist image
    images = [factory.image_from(path, n_roi, channels) for path in file_list]
    log.info("Begin loading %d images into RAM with %s channels, %d x %d each", len(images), ",".join(channels), w, h)
    sections = [image.load().astype(np.float32, copy=False) for image in images]
    log.info("Loaded %d images into RAM", len(sections))
    stack4d = np.stack(sections)

    # Average bias
    log.info("Calculating Stack average")
    master_aver = np.mean(stack4d, axis=0)
    path = output_path(output_dir, args.output_prefix, metadata, roi, 'aver')
    log.info("Saving master bias in %s", path)
    hdu = fits.PrimaryHDU(master_aver)
    fill_header(hdu.header, metadata, img0, comment=f'Master bias from {len(images)} images stack')
    hdu.writeto(path, overwrite=True)
    # Standard deviation Pixel map
    if args.stdev_map:
        log.info("Calculating stddev map")
        stdev_map = np.std(stack4d, axis=0)
        path = output_path(output_dir, args.output_prefix, metadata, roi, 'stdev')
        log.info("Saving stdev map in %s", path)
        hdu = fits.PrimaryHDU(stdev_map)
        fill_header(hdu.header, metadata, img0, comment=f'Standard dev. bias pixel map from {len(images)} images stack')
        hdu.writeto(path, overwrite=True)
    # Maximun puxel map
    if args.max_map:
        log.info("Calculating max pixel map")
        max_map = np.max(stack4d, axis=0)
        path = output_path(output_dir, args.output_prefix, metadata, roi, 'max')
        log.info("Saving max map in %s", path)
        hdu = fits.PrimaryHDU(max_map)
        fill_header(hdu.header, metadata, img0, comment=f'Maximum pixel map from {len(images)} images stack')
        hdu.writeto(path, overwrite=True)


# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def bias(args):
    command =  args.command
    if  command == 'create':
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


# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=bias, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="Master bias creation"
        )
