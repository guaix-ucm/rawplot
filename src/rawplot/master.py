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
from astropy.io import fits

from lica.cli import execute
from lica.misc import file_paths
from lica.validators import vdir, vfile, vfloat01, valid_channels
from lica.raw.loader import ImageLoaderFactory, FULL_FRAME_NROI, CHANNELS


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


def fill_header(header, metadata, img, history=None):
    header['CHANNELS'] = (metadata['channels'], 'Channel stored order')
    header['SECTION'] = (metadata['roi'], 'NumPy format')
    header['EXPTIME'] = (float(img.exposure()), '[s]')
    header['MAKER'] = (metadata['maker'], 'Manufacturer')
    header['CAMERA'] = (metadata['camera'], 'Sensor')
    header['ISO'] = (metadata['iso'], 'Sensor')
    header['DATE-OBS'] = (metadata['datetime'], 'When taken')
    header['IMAGETYP'] = (metadata['imagetyp'], 'Image type')
    header['COMMENT'] = f'Created with {os.path.basename(sys.argv[0])}'
    if history:
         header['HISTORY'] = history


def output_path(output_dir, prefix, metadata, roi, tag):
    width = metadata['width']
    height = metadata['height']
    channels = metadata['channels'].replace(' ', '_')
    imagetyp = metadata['imagetyp'].lower()
    filename = f"{prefix}_{imagetyp}_{tag}.fit"
    return os.path.join(output_dir, filename)


# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------


def master(args):
    channels = CHANNELS
    n_roi = FULL_FRAME_NROI
    image_type = args.image_type
    log.info("Normalized ROI is %s", n_roi)
    if args.output_dir is None:
        output_dir = os.getcwd()
    file_list = sorted(file_paths(args.input_dir, args.image_filter))
    factory = ImageLoaderFactory()
    # get common metadata from the first image in the list
    image0 = factory.image_from(file_list[0], n_roi, channels)
    roi = image0.roi()
    h, w = image0.shape()
    metadata = image0.metadata()
    metadata['imagetyp'] = image_type
   
    images = [factory.image_from(path, n_roi, channels) for path in file_list]
    log.info("Begin loading %d images into RAM with %s channels, %d x %d each", len(images), ",".join(channels), w, h)
    sections = [image.load().astype(np.float32, copy=False) for image in images]
    log.info("Loaded %d images into RAM", len(sections))
    stack4d = np.stack(sections)

    # Average bias
    log.info("Calculating master %s average", image_type)
    master_aver = np.mean(stack4d, axis=0)
    path = output_path(output_dir, args.output_prefix, metadata, roi, 'aver')
    log.info("Saving master %s file in %s", image_type, path)
    hdu = fits.PrimaryHDU(master_aver)
    fill_header(hdu.header, metadata, image0, history=f' Created master {image_type} from {len(images)} images')
    hdu.writeto(path, overwrite=True)
    # Standard deviation Pixel map
    if args.stdev_map:
        log.info("Calculating master %s stddev map", image_type)
        stdev_map = np.std(stack4d, axis=0)
        path = output_path(output_dir, args.output_prefix, metadata, roi, 'stdev')
        log.info("Saving stdev map in %s", path)
        hdu = fits.PrimaryHDU(stdev_map)
        fill_header(hdu.header, metadata, image0, history=f'Created std. dev. {image_type} pixel map from {len(images)} images')
        hdu.writeto(path, overwrite=True)
    # Maximun puxel map
    if args.max_map:
        log.info("Calculating master %s max pixel map", image_type)
        max_map = np.max(stack4d, axis=0)
        path = output_path(output_dir, args.output_prefix, metadata, roi, 'max')
        log.info("Saving max map in %s", path)
        hdu = fits.PrimaryHDU(max_map)
        fill_header(hdu.header, metadata, image0, history=f'Created max. {image_type} pixel map from {len(images)} images')
        hdu.writeto(path, overwrite=True)


# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):
    parser.add_argument('-i', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser.add_argument('-f', '--image-filter', type=str, required=True, help='Images filter, glob-style (i.e flat*, dark*)')
    parser.add_argument('-p', '--output-prefix', type=str, required=True, help='Output file prefix')
    parser.add_argument('-o', '--output-dir', type=vdir, default=None, help='Output directory defaults to current dir.')
    parser.add_argument('-t', '--image-type',  choices=['BIAS', 'DARK', 'FLAT', 'OBJECT'], default='BIAS', help='Image type. (default: %(default)s)')
    parser.add_argument('--stdev-map',  action='store_true', help='Also create standard deviation map')
    parser.add_argument('--max-map',  action='store_true', help='Also create max. pixel value map')
 
# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=master, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="Create a averaged master 3D FITS cube from a list of not-debayered color images"
    )
