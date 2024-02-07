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
from lica.validators import vdir, vfile, vfloat, vflopath


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
        log.info("Reading %s %s", key, first_header[key])
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

def arith_sub(args):
    folder = os.path.dirname(args.first)
    name, ext = os.path.splitext(os.path.basename(args.first))
    res_path = os.path.join(folder, f"{name}_subs{ext}")
    with fits.open(args.first) as hdu1:
        header = hdu1[0].header
        if type(args.second) == float:
            pixels =   hdu1[0].data - args.second
        else:
            with fits.open(args.first) as hdu2:
                pixels = hdu1[0].data - hdu2[0].data
    hdu_res = fits.PrimaryHDU(pixels)
    add_result_header(hdu_res.header, header, args.second, args.history)
    hdu_res.writeto(res_path, overwrite=True)

# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

COMMAND_TABLE = {
    'sub': arith_sub,
}

def arith(args):
    func = COMMAND_TABLE[args.command]
    func(args)

def add_args(parser):
    
    subparser = parser.add_subparsers(dest='command')

    parser_sub = subparser.add_parser('sub', help='Substracts an image or a value (second argument) from a given image (first argument)')
    parser_sub.add_argument('first', type=vfile, help='Image to be substracted')
    parser_sub.add_argument('second', type=vflopath, help='Image to be substracted')
    parser_sub.add_argument('-hi', '--history', type=str, help='Optional HISTORY FITS card to add to resulting image')

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
