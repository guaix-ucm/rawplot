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


def bias_display(args):
    channels = valid_channels(args.channels)
    log.info("Loading %s channels in this order", " ".join(channels))
    image = FITSImage(args.input_file)
    roi = image.roi(args.x0, args.y0, args.width, args.height)
    stack = image.load_cube(channels=channels)
    section = image.load_cube(roi, channels)
    log.info("Stack shape is %s, dtype is %s", stack.shape, stack.dtype)
    aver = np.mean(section,  axis=(1,2))
    mdn = np.median(section,  axis=(1,2))
    std = np.std(section, axis=(1,2))
    metadata = image.metadata()
    log.info("Average is %s", aver)
    log.info("StdDev is %s", std)
    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(12, 9), layout='tight')
    title = f"Image: {image.name()}\n" \
        f"{metadata['maker']} {metadata['camera']}, Exposure: {metadata['exposure']} [s]\n" \
        f"Color Plane Size: {image.shape()[0]} rows x {image.shape()[1]} cols\n" \
        f"Stats Section: {roi} {roi.height()} rows x {roi.width()} cols"
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


def bias_histogram(args):
    channels = valid_channels(args.channels)
    log.info("Loading %s channels in this order", " ".join(channels))
    decimate = args.every
    dcm = fractions.Fraction(1, decimate)
    image = FITSImage(args.input_file)
    roi = image.roi(args.x0, args.y0, args.width, args.height)
    stack = image.load_cube(channels=channels)
    section = image.load_cube(roi, channels)
    log.info("Stack shape is %s, dtype is %s", stack.shape, stack.dtype)
    aver = np.mean(section,  axis=(1,2))
    mdn = np.median(section,  axis=(1,2))
    std = np.std(section, axis=(1,2))
    metadata = image.metadata()
    log.info("Average is %s", aver)
    log.info("StdDev is %s", std)
    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(12, 9), layout='tight')
    title = f"Image: {image.name()}\n" \
        f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}, Exposure: {metadata['exposure']} [s]\n" \
        f"Color Plane Size: {image.shape()[0]} rows x {image.shape()[1]} cols (decimated {dcm})\n" \
        f"Stats Section: {roi} {roi.height()} rows x {roi.width()} cols"
    

    fig.suptitle(title)
    axes = axes_reshape(axes, channels)
    for row in range(0,display_rows):
        for col in range(0,display_cols):
            i = 2*row+col
            if len(channels) == 3 and row == 1 and col == 1: # Skip the empty slot in 2x2 layout with 3 items
                axes[row][col].set_axis_off()
                break
            if args.histogram:
                plot_histo(axes[row][col], stack[i], channels[i], decimate, aver[i], mdn[i], std[i])
    plt.show()







    title = f"Image: {image.name()}\n" \
        f"{metadata['maker']} {metadata['camera']}, Exposure: {metadata['exposure']} [s]\n" \
        f"Color Plane Size: {image.shape()[0]} rows x {image.shape()[1]} cols\n" \
        f"Stats Section: {roi} {roi.height()} rows x {roi.width()} cols"
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

   
# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def bias(args):
    command =  args.command
    if  command == 'create':
        bias_create(args)
    elif command == 'pixels':
        bias_display(args)



# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):

    subparser = parser.add_subparsers(dest='command')

    parser_create = subparser.add_parser('create', help='Create a 3D FITS cube master bias')
    parser_pixels = subparser.add_parser('pixels', help='Display images from a 3D FITS cube master bias')
    parser_histo  = subparser.add_parser('histo', help='Display histogram from a 3D FITS cube master bias')
   
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


    parser_pixels.add_argument('-i', '--input-file', type=vfile, required=True, help='Input FITS file')
    parser_pixels.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser_pixels.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser_pixels.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser_pixels.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser_pixels.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')

    parser_histo.add_argument('-i', '--input-file', type=vfile, required=True, help='Input FITS file')
    parser_histo.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser_histo.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser_histo.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser_histo.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser_histo.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser.add_argument('--every', type=int, metavar='<N>', default=10, help='Decimation factor for histogram plot')
   

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
