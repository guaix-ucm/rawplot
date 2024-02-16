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
import logging

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
from astropy.io import fits

from lica.cli import execute
from lica.misc import file_paths
from lica.validators import vdir, vfile, vfloat, vflopath
from lica.raw.loader import ImageLoaderFactory,  FULL_FRAME_NROI, CHANNELS

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


# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------
    
def add_result_header(res_header, first_header, second, history):
    for key in first_header.keys():
        if key == 'COMMENT':
            for comment in first_header[key]:
                res_header.add_comment(comment)
        elif key == 'HISTORY':
            for hist in first_header[key]:
                res_header.add_history(hist)
        else:
             res_header[key] = first_header[key]
    if history:
        res_header['HISTORY'] = history[:72]
    else:
        data =  f"Substracted {second:0.2e}" if (type(second) == float) else f"Substracted {os.path.basename(second)[:60]}"
        res_header['HISTORY'] = data

def output_file(args):
    if args.output_file:
        return args.output_file
    folder = os.path.dirname(args.first)
    name, ext = os.path.splitext(os.path.basename(args.first))
    return os.path.join(folder, f"{name}_subs{ext}")

def arith_sub(args):
    res_path = output_file(args)
    with fits.open(args.first) as hdu1:
        header = hdu1[0].header
        if type(args.second) == float:
            pixels =   hdu1[0].data - args.second
        else:
            with fits.open(args.second) as hdu2:
                pixels = hdu1[0].data - hdu2[0].data
    hdu_res = fits.PrimaryHDU(pixels)
    add_result_header(hdu_res.header, header, args.second, args.history)
    hdu_res.writeto(res_path, overwrite=True)
    log.info("Created result image on: %s", res_path)


def arith_rgb(args):
    res_path = output_file(args)
    factory = ImageLoaderFactory()
    red_image = factory.image_from(args.red, FULL_FRAME_NROI, CHANNELS)
    blue_image = factory.image_from(args.blue, FULL_FRAME_NROI, CHANNELS)
    green_image = factory.image_from(args.green, FULL_FRAME_NROI, CHANNELS)
    red_pixels = red_image.load()
    green_pixels = green_image.load()
    blue_pixels = blue_image.load()
    metadata = green_image.metadata()
    # We compose the new composite image
    # from the different color planes
    # in each image
    R = CHANNELS.index('R')
    B = CHANNELS.index('B')
    Gr = CHANNELS.index('Gr')
    Gb =  CHANNELS.index('Gb')
    result_seq = [None, None, None, None]
    result_seq[R] = red_pixels[R]
    result_seq[B] = blue_pixels[B]
    result_seq[Gr] = green_pixels[Gr]
    result_seq[Gb] = green_pixels[Gb]
    result_array = np.stack(result_seq)
    # Copying metadata from the Green image
    hdu_res = fits.PrimaryHDU(result_array)
    for key, item in metadata.items():
        hdu_res.header[key] = str(item)
    hdu_res.header['exptime'] = float(metadata['exposure'])
    hdu_res.writeto(res_path, overwrite=True)
    log.info("Created result image on: %s", res_path)

# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

COMMAND_TABLE = {
    'sub': arith_sub,
    'rgb': arith_rgb,
}

def arith(args):
    func = COMMAND_TABLE[args.command]
    func(args)

def add_args(parser):
    
    subparser = parser.add_subparsers(dest='command')

    parser_sub = subparser.add_parser('sub', help='Substracts an image or a value (second argument) from a given image (first argument)')
    parser_sub.add_argument('first', type=vfile, help='Image to be substracted')
    parser_sub.add_argument('second', type=vflopath, help='Scalar value or 3D FITS Cube image to be substracted')
    parser_sub.add_argument('-o', '--output-file', type=str, help='Optional output file name for the resulting 3D FITS cube image')
    parser_sub.add_argument('-hi', '--history', type=str, help='Optional HISTORY FITS card to add to resulting image')

    parser_rgb = subparser.add_parser('rgb', help='Combine R Gr, Gb and B channels from 3 different images into the same 3D FITS cube')
    parser_rgb.add_argument('-r','--red', type=vfile, required=True, help='Red channel image')
    parser_rgb.add_argument('-g','--green', type=vfile, required=True, help='Green channels image')
    parser_rgb.add_argument('-b','--blue', type=vfile, required=True, help='Blue channel image')
    parser_rgb.add_argument('-o', '--output-file', type=str, help='Optional output file name for the resulting 3D FITS cube image')
    

# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=arith, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="Arithmetic operations on one or two 3D-FITS cubes"
    )
