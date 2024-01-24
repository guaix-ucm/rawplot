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
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lica.cli import execute
from lica.validators import vfile, vfloat01, valid_channels
from lica.rawimage import RawImage, SimulatedDarkImage
from lica.mpl import plot_layout, plot_cmap, plot_edge_color, axes_reshape

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# ------------------
# Auxiliary fnctions
# ------------------

def plot_histo(axes, color_plane, title, decimate, average, median, stddev):
    axes.set_title(fr'channel {title}: $median={median}, \mu={average},\;\sigma={stddev}$')
    data = color_plane.reshape(-1)[::decimate]
    bins=list(range(data.min(), data.max()+1))
    axes.hist(data, bins=bins, rwidth=0.9, align='left', label='hist')
    axes.set_xlabel('Pixel value [DN]')
    axes.set_ylabel('Pixel count')
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.axvline(x=average, linestyle='--', color='r', label="mean")
    axes.axvline(x=median, linestyle='--', color='k', label="median")
    axes.legend()


def plot_image(fig, axes, color_plane, roi, title, average, median, stddev, colormap, edgecolor):
    axes.set_title(fr'{title}: $median={median}, \mu={average},\;\sigma={stddev}$')
    im = axes.imshow(color_plane, cmap=colormap)
    # Create a Rectangle patch
    rect = patches.Rectangle(roi.xy(), roi.width(), roi.height(), 
                    linewidth=1, linestyle='--', edgecolor=edgecolor, facecolor='none')
    axes.add_patch(rect)
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.10)
    fig.colorbar(im, cax=cax, orientation='vertical')



# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def image(args):
    decimate = args.every
    channels = valid_channels(args.channels)
    dcm = fractions.Fraction(1, decimate)
    if args.sim_dark is not None:
        image = SimulatedDarkImage(args.input_file, dk_current=args.sim_dark)
    else:
        image = RawImage(args.input_file)
    roi = image.roi(args.x0, args.y0, args.width, args.height)
    stack = image.debayered(channels=channels)
    section = image.debayered(roi, channels)
    log.info("Stack shape is %s, dtype is %s", stack.shape, stack.dtype)
    aver = np.ndarray.round( np.mean(section,  axis=(1,2)), 1)
    mdn = np.ndarray.round( np.median(section,  axis=(1,2)), 1)
    std = np.ndarray.round( np.std(section, axis=(1,2)), 2)
    metadata = image.exif()
    log.info("average shape is %s", aver.shape)
    log.info("stddev shape is %s", std.shape)

    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(12, 9), layout='tight')
    if args.histogram:
         title = f"Image: {image.name()}\n" \
            f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}, Exposure: {metadata['exposure']} [s]\n" \
            f"Color Plane Size: {image.shape()[0]} rows x {image.shape()[1]} cols (decimated {dcm})\n" \
            f"Stats Section: {roi} {roi.height()} rows x {roi.width()} cols"
    else:
        title = f"Image: {image.name()}\n" \
            f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}, Exposure: {metadata['exposure']} [s]\n" \
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
            if args.histogram:
                plot_histo(axes[row][col], stack[i], channels[i], decimate, aver[i], mdn[i], std[i])
            elif args.image:
                cmap = plot_cmap(channels)
                edge_color = plot_edge_color(channels)
                plot_image(fig, axes[row][col], stack[i], roi, channels[i], aver[i], mdn[i], std[i], cmap[i], edge_color[i])
    plt.show()

# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):
    parser.add_argument('-i', '--input-file', type=vfile, required=True, help='Input RAW file')
    parser.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1]')
    parser.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1]')
    parser.add_argument('-c','--channels', default=['R', 'G1', 'G2','B'], nargs='+',
                    choices=['R', 'G1', 'G2', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--image',  action='store_true', help='display image and section statistics')
    group1.add_argument('--histogram', action='store_true', help='display image histogram plot')
    parser.add_argument('--every', type=int, metavar='<N>', default=10, help='Decimation factor for histogram plot')
    parser.add_argument('--sim-dark', type=float, default=None, help='Simulate dark frame with given dark current')

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
