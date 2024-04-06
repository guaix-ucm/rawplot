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
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lica.cli import execute
from lica.validators import vfile, vfloat, vfloat01, vflopath
from lica.raw.loader import ImageLoaderFactory,  FULL_FRAME_NROI
from lica.raw.analyzer.image import ImageStatistics

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.mpl.plot import mpl_main_image_loop, mpl_main_plot_loop
from .util.common import common_info, make_plot_title_from

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# ------------------
# Auxiliary fnctions
# ------------------

def plot_histo(axes, i, x, y, xtitle, ytitle, ylabel, channel, **kwargs):
    median = kwargs['median'][i]
    mean = kwargs['mean'][i]
    stddev = kwargs['stddev'][i]
    decimate = kwargs.get('decimate', 10)
    ylog = kwargs.get('ylog', False)
    title = fr'{channel[i]}: median={median:.2f}, $\mu={mean:.2f}, \sigma={stddev:.2f}$'
    axes.set_title(title)
    data = x[i].reshape(-1)[::decimate]
    if data.dtype  in (np.uint16, np.uint32,):
        bins=list(range(data.min(), data.max()+1))
    else:
        bins='auto'
    if ylog:
        axes.set_yscale('log', base=10)
    axes.hist(data, bins=bins, rwidth=0.9, align='left', label=ylabel)
    axes.set_xlabel(xtitle)
    axes.set_ylabel(ytitle)
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
  

def plot_image(axes, i, pixels, channel, roi, colormap, edgecolor, **kwargs):
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
    analyzer = ImageStatistics(file_path, n_roi, channels, args.bias, args.dark)
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
        plot_func = plot_histo,
        xtitle = "Pixel value [DN]",
        ytitle = "Pixel Count",
        x     = pixels,
        y     = None, # Y is the pixel count, no explicit array is needed
        channels = channels,
        # Extra arguments
        decimate = decimate,
        mean = aver,
        median = mdn,
        stddev = std,
        ylog = args.y_log
    )


def image_pixels(args):
    file_path, roi, n_roi, channels, metadata = common_info(args)
    simulated = args.sim_dark is not None
    pixels = ImageLoaderFactory().image_from(file_path, FULL_FRAME_NROI, channels, simulated=simulated, dark_current=args.sim_dark).load()
    analyzer = ImageStatistics(file_path, n_roi, channels, bias=args.bias, dark=args.dark)
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

COMMAND_TABLE = {
    'pixels': image_pixels,
    'histo': image_histo, 
}

def image(args):
    command =  args.command
    func = COMMAND_TABLE[command]
    func(args)


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
    parser_pixels.add_argument('-x', '--x0', type=vfloat01,  help='Normalized ROI start point, x0 coordinate [0..1]')
    parser_pixels.add_argument('-y', '--y0', type=vfloat01,  help='Normalized ROI start point, y0 coordinate [0..1]')
    parser_pixels.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1] (default: %(default)s)')
    parser_pixels.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1] (default: %(default)s)')
    parser_pixels.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser_pixels.add_argument('-bi', '--bias',  type=vflopath,  help='Bias, either a single value for all channels or else a 3D FITS cube file (default: %(default)s)')
    parser_pixels.add_argument('-dk', '--dark',  type=vfloat,  help='Dark count rate in DN/sec. (default: %(default)s)')
    parser_pixels.add_argument('--sim-dark', type=float,  help='Simulate dark frame with given dark current')
    parser_pixels.add_argument('--sim-dark', type=float,  help='Generate synthetic dark frame with given dark count rate [DN/sec]')
    parser_pixels.add_argument('--sim-read-noise', type=float,  help='Generate synthetic dark frame with given readout noise [DN]')


    # -------------------------
    # Histogram command parsing
    # -------------------------
    parser_histo.add_argument('-i', '--input-file', type=vfile, required=True, help='Input RAW file')
    parser_histo.add_argument('-x', '--x0', type=vfloat01,  help='Normalized ROI start point, x0 coordinate [0..1]')
    parser_histo.add_argument('-y', '--y0', type=vfloat01,  help='Normalized ROI start point, y0 coordinate [0..1]')
    parser_histo.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1] (default: %(default)s)')
    parser_histo.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1] (default: %(default)s) ')
    parser_histo.add_argument('-c','--channels', default=('R', 'Gr', 'Gb','B'), nargs='+',
                    choices=('R', 'Gr', 'Gb', 'G', 'B'),
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser_histo.add_argument('--every', type=int, metavar='<N>', default=100, help='Decimation factor for histogram plot (default: %(default)s) ')
    parser_histo.add_argument('-bi', '--bias',  type=vflopath,  help='Bias, either a single value for all channels or else a 3D FITS cube file (default: %(default)s)')
    parser_histo.add_argument('-dk', '--dark',  type=vfloat,  help='Dark count rate in DN/sec. (default: %(default)s)')
    parser_histo.add_argument('--y-log',  action='store_true', help='Logaritmic scale for pixel counts')
    parser_histo.add_argument('--sim-dark', type=float,  help='Generate synthetic dark frame with given dark count rate [DN/sec]')
    parser_histo.add_argument('--sim-read-noise', type=float,  help='Generate synthetic dark frame with given readout noise [DN]')

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
