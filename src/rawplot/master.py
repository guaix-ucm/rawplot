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
import itertools

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
from astropy.io import fits

from lica.cli import execute
from lica.misc import file_paths
from lica.validators import vdir, vfile, vfloat, vfloat01, valid_channels
from lica.raw.loader import ImageLoaderFactory, FULL_FRAME_NROI, CHANNELS


# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__

# ----------------
# Module constants
# ----------------

IMAGETYP_KEYWORDS = {
    'bias': 'Bias Frame',
    'dark': 'Dark Frame',
    'flat': 'Flat Frame',
    'light': 'Light Frame',
}
# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# ------------------
# Auxiliary fnctions
# ------------------

try:
    from itertools import batched
except:
    def batched(iterable, n):
        if n < 1:
            raise ValueError('n must be at least one')
        it = iter(iterable)
        while batch := tuple(itertools.islice(it, n)):
            yield batch

def fill_header(header, metadata, img, history=None):
    header['CHANNELS'] = (metadata['channels'], 'Channel stored order')
    header['SECTION'] = (metadata['roi'], 'NumPy format')
    header['EXPTIME'] = (float(img.exptime()), '[s]')
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
    imagetyp = metadata['imagetyp'].lower().replace(' ', '_')
    filename = f"{prefix}_{imagetyp}_{tag}.fit"
    return os.path.join(output_dir, filename)

def basic_reduction(image, bias, dark):
    if bias is None and dark is None:
        return image.load().astype(np.float32, copy=False)
    if bias is None and dark is not None:
        log.info("Substracting dark count: %0.2f", dark * image.exptime())
        return image.load().astype(np.float32, copy=False) - dark * image.exptime()
    elif bias is not None and dark is None:
        log.info("Substracting pedestal value: %0.2f", bias)
        return image.load().astype(np.float32, copy=False) - bias
    elif bias is not None and dark is not None:
        log.info("Substracting pedestal value: %0.2f and dark count: %0.2f", bias)
        return image.load().astype(np.float32, copy=False) - bias - dark * image.exptime()

# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def master(args):
    M = len(CHANNELS)
    image_type = args.image_type
    log.info("Normalized ROI is %s", FULL_FRAME_NROI)
    if args.output_dir is None:
        args.output_dir = os.getcwd()
    file_list = sorted(file_paths(args.input_dir, args.image_filter))
    factory = ImageLoaderFactory()
    # get common metadata from the first image in the list
    image0 = factory.image_from(file_list[0], FULL_FRAME_NROI, CHANNELS)
    roi = image0.roi()
    h, w = image0.shape()
    metadata = image0.metadata()
    metadata['imagetyp'] = IMAGETYP_KEYWORDS.get(image_type, None)
    accum = np.zeros(shape=(M, h, w), dtype=np.float32)
    accum_file = os.path.join(args.output_dir, "accum.npy")
    accum.tofile(accum_file, sep="")
    N = len(file_list)
    # Divide the input file list in smaller batches to avoid
    # excessive RAM usage
    batched_list = list(batched(file_list, args.batch))
    log.info("The process comprises %d batches of %d images max. per batch", len(batched_list), args.batch)
    for i, batch in enumerate(batched_list, start=1):
        images = [factory.image_from(path, FULL_FRAME_NROI, CHANNELS) for path in batch]
        log.info("[%d/%d] Begin loading %d images into RAM with %s channels, %d x %d each", 
            i, len(batched_list), len(batch), " ".join(CHANNELS), w, h)
        pixels = [basic_reduction(image, args.bias, args.dark) for image in images] 
        current = np.sum(np.stack(pixels), axis=0)
        accum = np.fromfile(accum_file, dtype=np.float32).reshape(M, h, w)
        accum += current
        accum.tofile(accum_file, sep="")
    os.remove(accum_file)
    master_aver = accum / N
    path = output_path(args.output_dir, args.prefix, metadata, roi, 'aver')
    log.info("Saving master %s file from %d images in %s", image_type, N, path)
    hdu = fits.PrimaryHDU(master_aver)
    fill_header(hdu.header, metadata, image0, history=f' Created master {image_type} from {N} images')
    hdu.writeto(path, overwrite=True)
    

    
# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):
    parser.add_argument('-i', '--input-dir', type=vdir, required=True, help='Input directory with RAW files')
    parser.add_argument('-f', '--filter', dest='image_filter', type=str, required=True, help='Images filter, glob-style (i.e flat*, dark*)')
    parser.add_argument('-p', '--prefix', type=str, required=True, help='Output file prefix')
    parser.add_argument('-o', '--output-dir', type=vdir,  help='Output directory defaults to current dir.')
    parser.add_argument('-b', '--batch', type=int,  default=25, help='Batch size (default) %(default)s')
    parser.add_argument('-t', '--image-type',  choices=['bias', 'dark', 'flat', 'light'], default='bias', help='Image type. (default: %(default)s)')
    parser.add_argument('-bi', '--bias',  type=vfloat,  help='Substract a pedestal level in each image before averaging. (default: %(default)s)')
    parser.add_argument('-dk', '--dark',  type=vfloat,  help='Dark count rate in DN/sec. Rate x T is substracted from each image before averaging. (default: %(default)s)')
# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=master, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="Create a averaged master 3D FITS cube from a list of not-debayered RAW color images"
    )
