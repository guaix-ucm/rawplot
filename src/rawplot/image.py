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
import fractions

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt



from lica.cli import execute
from lica.validators import vfile, vfloat01, valid_channels
from lica.raw import ImageLoaderFactory, SimulatedDarkImage, NormRoi, CHANNELS
from lica.misc import file_paths

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.mpl.plot import plot_layout, plot_cmap, plot_edge_color, plot_image, plot_histo, axes_reshape

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

def image_common(args):   
    channels = valid_channels(args.channels)
    n_roi = NormRoi(args.x0, args.y0, args.width, args.height)
    if args.sim_dark is not None:
        image = SimulatedDarkImage(args.input_file, n_roi, channels, dk_current=args.sim_dark)
    else:
        factory =  ImageLoaderFactory()
        image = factory.image_from(args.input_file, n_roi=None, channels=channels)
    stack = image.load()
    image_section = factory.image_from(args.input_file, n_roi=n_roi, channels=channels)
    section = image_section.load()
    roi = image_section.roi()
    aver = np.mean(section,  axis=(1,2))
    mdn = np.median(section,  axis=(1,2))
    std = np.std(section, axis=(1,2))
    metadata = image.metadata()
    log.info("section %s average is %s", roi, aver)
    log.info("section %s stddev is %s", roi, std)
    log.info("section %s median is %s", roi, mdn)
    return  roi, channels, metadata, stack, aver, mdn, std


def image_histo(args):
    roi, channels, metadata, stack, aver, mdn, std= image_common(args)
    decimate = args.every
    dcm = fractions.Fraction(1, decimate)
    title = f"Image: {metadata['name']}\n" \
            f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}, Exposure: {metadata['exposure']} [s]\n" \
            f"Color Plane Size: {metadata['height']} rows x {metadata['width']} cols (decimated {dcm})\n" \
            f"Stats Section: {roi} {roi.height()} rows x {roi.width()} cols"
    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(12, 9), layout='tight')
    fig.suptitle(title)
    axes = axes_reshape(axes, channels)
    for row in range(0,display_rows):
        for col in range(0,display_cols):
            i = 2*row+col
            if len(channels) == 3 and row == 1 and col == 1: # Skip the empty slot in 2x2 layout with 3 items
                axes[row][col].set_axis_off()
                break
            plot_histo(axes[row][col], stack[i], channels[i], decimate, aver[i], mdn[i], std[i])
    plt.show()


def image_pixels(args):
    roi, channels, metadata, stack, aver, mdn, std = image_common(args)
    title = f"Image: {metadata['name']}\n" \
            f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}, Exposure: {metadata['exposure']} [s]\n" \
            f"Color Plane Size: {metadata['height']} rows x {metadata['width']} cols\n" \
            f"Stats Section: {roi} {roi.height()} rows x {roi.width()} cols"
    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(12, 9), layout='tight')
    fig.suptitle(title)
    axes = axes_reshape(axes, channels)
    for row in range(0,display_rows):
        for col in range(0,display_cols):
            i = 2*row+col
            if len(channels) == 3 and row == 1 and col == 1: # Skip the empty slot in 2x2 layout with 3 items
                axes[row][col].set_axis_off()
                break
            cmap = plot_cmap(channels)
            edge_color = plot_edge_color(channels)
            plot_image(fig, axes[row][col], stack[i], roi, channels[i], aver[i], mdn[i], std[i], cmap[i], edge_color[i])
    plt.show()


def image(args):
    command =  args.command
    if  command == 'pixels':
        image_pixels(args)
    else:
        image_histo(args)


# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):

    subparser = parser.add_subparsers(dest='command')

    parser_pixels = subparser.add_parser('pixels', help='Display image pixels')
    parser_histo  = subparser.add_parser('histo', help='Display image histogram')

    parser_pixels.add_argument('-i', '--input-file', type=vfile, required=True, help='Input RAW file')
    parser_pixels.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser_pixels.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser_pixels.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser_pixels.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser_pixels.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser_pixels.add_argument('--sim-dark', type=float, default=None, help='Simulate dark frame with given dark current')

    parser_histo.add_argument('-i', '--input-file', type=vfile, required=True, help='Input RAW file')
    parser_histo.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser_histo.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser_histo.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser_histo.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser_histo.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser_histo.add_argument('--every', type=int, metavar='<N>', default=100, help='Decimation factor for histogram plot')
    parser_histo.add_argument('--sim-dark', type=float, default=None, help='Simulate dark frame with given dark current')

   


# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=image, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="Display RAW image or histogram in channels"
        )
