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
from lica.validators import vfile, vfloat, vfloat01, valid_channels
from lica.raw.loader import ImageLoaderFactory, SimulatedDarkImage, NormRoi, FULL_FRAME_NROI
from lica.raw.analyzer.image import ImageStatistics

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.mpl.plot import mpl_main_image_loop, mpl_main_plot_loop, plot_layout, plot_cmap, plot_edge_color, axes_reshape
from .util.common import common_info, bias_from, make_plot_title_from

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# ------------------
# Auxiliary fnctions
# ------------------


def plot_histo(axes, i, channel, xlabel, ylabel, x, *args, **kwargs):
    median = kwargs['median'][i]
    mean = kwargs['mean'][i]
    stddev = kwargs['stddev'][i]
    decimate = kwargs.get('decimate', 10)
    ylog = kwargs.get('ylog', False)
    title = fr'{channel}: median={median:.2f}, $\mu={mean:.2f}, \sigma={stddev:.2f}$'
    axes.set_title(title)
    data = x.reshape(-1)[::decimate]
    if data.dtype  in (np.uint16, np.uint32,):
        bins=list(range(data.min(), data.max()+1))
    else:
        bins='auto'
    if ylog:
        axes.set_yscale('log', base=10)
    axes.hist(data, bins=bins, rwidth=0.9, align='left', label='hist')
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.legend()
  


def plot_image(axes, i, channel, roi, colormap, edgecolor, pixels, *args, **kwargs):
    median = kwargs['median'][i]
    mean = kwargs['mean'][i]
    stddev = kwargs['stddev'][i]
    title = fr'{channel}: median={median:.2f}, $\mu={mean:.2f}, \sigma={stddev:.2f}$'
    axes.set_title(title)
    im = axes.imshow(pixels, cmap=colormap)
    # Create a Rectangle patch
    rect = patches.Rectangle(roi.xy(), roi.width(), roi.height(), 
                    linewidth=1, linestyle='--', edgecolor=edgecolor, facecolor='none')
    axes.add_patch(rect)
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.10)
    axes.get_figure().colorbar(im, cax=cax, orientation='vertical')


# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def image_histo(args):
    file_path, roi, n_roi, channels, metadata = common_info(args)
    decimate = args.every
    dcm = fractions.Fraction(1, decimate)
    bias = bias_from(args)
    analyzer = ImageStatistics(file_path, n_roi, channels, bias=bias)
    analyzer.run()
    aver, mdn, std = analyzer.mean() , analyzer.median(), analyzer.std()
    log.info("section %s average is %s", roi, aver)
    log.info("section %s stddev is %s", roi, std)
    log.info("section %s median is %s", roi, mdn)
    pixels = analyzer.pixels()
    title = make_plot_title_from(f"{metadata['name']} (decimated {dcm})", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
        figsize  = (12, 9),
        channels = channels,
        plot_func = plot_histo,
        xlabel = "Pixel value [DN]",
        ylabel = "Pixel Count",
        x     = pixels,
        # Extra arguments
        decimate = decimate,
        mean = aver,
        median = mdn,
        stddev = std,
        ylog = args.y_log
    )


def image_pixels(args):
    file_path, roi, n_roi, channels, metadata = common_info(args)
    bias = bias_from(args)
    pixels = ImageLoaderFactory().image_from(file_path, FULL_FRAME_NROI, channels).load()
    analyzer = ImageStatistics(file_path, n_roi, channels, bias=bias)
    analyzer.run()
    aver, mdn, std = analyzer.mean() , analyzer.median(), analyzer.std()
    log.info("section %s average is %s", roi, aver)
    log.info("section %s stddev is %s", roi, std)
    log.info("section %s median is %s", roi, mdn)
    title = make_plot_title_from(f"{metadata['name']}", metadata, roi)
    mpl_main_image_loop(
        title    = title,
        figsize  = (12, 9),
        channels = channels,
        roi = roi,
        plot_func = plot_image,
        pixels    = pixels,
        # Extra arguments
        mean = aver,
        median = mdn,
        stddev = std
    )


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

    # -----------------------
    # Pixels command parsing
    # ----------------------
    parser_pixels.add_argument('-i', '--input-file', type=vfile, required=True, help='Input RAW file')
    parser_pixels.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser_pixels.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser_pixels.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1] (default: %(default)s)')
    parser_pixels.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1] (default: %(default)s)')
    parser_pixels.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    group0 = parser_pixels.add_mutually_exclusive_group(required=False)
    group0.add_argument('-bl', '--bias-level',  type=vfloat, default=None, help='Bias level, common for all channels (default: %(default)s)')
    group0.add_argument('-bf', '--bias-file',  type=vfile, default=None, help='Bias image (3D FITS cube) (default: %(default)s)')

    parser_pixels.add_argument('--sim-dark', type=float, default=None, help='Simulate dark frame with given dark current')

    # -------------------------
    # Histogram command parsing
    # -------------------------
    parser_histo.add_argument('-i', '--input-file', type=vfile, required=True, help='Input RAW file')
    parser_histo.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser_histo.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser_histo.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1] (default: %(default)s)')
    parser_histo.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1] (default: %(default)s) ')
    parser_histo.add_argument('-c','--channels', default=('R', 'Gr', 'Gb','B'), nargs='+',
                    choices=('R', 'Gr', 'Gb', 'G', 'B'),
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser_histo.add_argument('--every', type=int, metavar='<N>', default=100, help='Decimation factor for histogram plot (default: %(default)s) ')
    group0 = parser_histo.add_mutually_exclusive_group(required=False)
    group0.add_argument('-bl', '--bias-level',  type=vfloat, default=None, help='Bias level, common for all channels (default: %(default)s)')
    group0.add_argument('-bf', '--bias-file',  type=vfile, default=None, help='Bias image (3D FITS cube) (default: %(default)s)')
    parser_histo.add_argument('--y-log',  action='store_true', help='Logaritmic scale for pixel counts')
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
