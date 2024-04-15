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
from lica.validators import vfile, vfloat, vfloat01, vflopath
from lica.raw.loader import ImageLoaderFactory, FULL_FRAME_NROI
from lica.raw.analyzer.image import ImageStatistics

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.mpl.plot import mpl_main_image_loop, mpl_main_plot_loop, mpl_main_pairs_plot_loop
from .util.common import common_info_with_sim, make_plot_title_from, make_plot_no_roi_title_from, extended_roi


# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# -----------------
# Matplotlib styles
# -----------------

# Load global style sheets
plt.style.use("rawplot.resources.global")

# ------------------
# Auxiliary fnctions
# ------------------


def plot_radial(axes, i, is_H, x, y, xtitle, ytitle, ylabel, channels, **kwargs):
    centroid = kwargs['centroid']
    geom_center = kwargs['geom_center']
    x_roi = kwargs['roi_x']
    y_roi = kwargs['roi_y']
    axes.set_xlabel(xtitle)
    axes.set_ylabel(ytitle)
    if is_H:
        title = f'{channels[i]}. Aggregate of {x_roi.height()} central columns along {x_roi.width()} rows'
        axes.plot(x[0][i], y[0][i], label="H")
        axes.axvline(centroid[0][i], linestyle='--', label="opti.")
        axes.axvline(geom_center[0][i], linestyle=':', color='tab:orange', label="geom.")
    else:
        title = f'{channels[i]}. Aggregate of {y_roi.width()} central rows along {y_roi.height()} columns'
        axes.plot(x[1][i], y[1][i], label = "V")
        axes.axvline(centroid[1][i], linestyle='--', label="opti.")
        axes.axvline(geom_center[1][i], linestyle=':', color='tab:orange', label="geom.")
    axes.set_title(title)
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.legend()

  

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
    # Create the Rectangle patch for the standard ROI   
    rect = patches.Rectangle(roi.xy(), roi.width(), roi.height(), 
                    linewidth=1, linestyle='--', edgecolor=edgecolor, facecolor='none')
    axes.add_patch(rect)
    # Create more Rectangle patches for optional extended ROIs
    for key in ('extended_roi_x', 'extended_roi_y'):
        extended_roi = kwargs.get(key)
        if extended_roi:
            rect = patches.Rectangle(extended_roi.xy(), extended_roi.width(), extended_roi.height(), 
                        linewidth=1, linestyle=':', edgecolor=edgecolor, facecolor='none')
            axes.add_patch(rect)
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.10)
    axes.get_figure().colorbar(im, cax=cax, orientation='vertical')


# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def image_histo(args):
    file_path, roi, n_roi, channels, metadata, simulated, image0 = common_info_with_sim(args)
    decimate = args.every
    dcm = fractions.Fraction(1, decimate)
    analyzer = ImageStatistics.attach(image0, bias=args.bias, dark=args.dark)
    analyzer.run()
    aver, mdn, std = analyzer.mean() , analyzer.median(), analyzer.std()
    log.info("section %s average is %s", roi, aver)
    log.info("section %s stddev is %s", roi, std)
    log.info("section %s median is %s", roi, mdn)
    pixels = analyzer.pixels()
    title = make_plot_title_from(f"{metadata['name']} (decimated {dcm})", metadata, roi)
    mpl_main_plot_loop(
        title    = title,
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
    file_path, roi, n_roi, channels, metadata, simulated, image0 = common_info_with_sim(args)
    simulated = args.sim_dark is not None or args.sim_read_noise is not None
    # The pixels we need to display are those of the whole image, not the ROI
    pixels = ImageLoaderFactory().image_from(file_path, FULL_FRAME_NROI, channels, 
        simulated=simulated, 
        read_noise=args.sim_read_noise,
        dark_current=args.sim_dark
    ).load()
    if args.extended_roi:
        height, width = image0.shape()
        extended_roi_x, extended_roi_y = extended_roi(roi, width, height)
    else:
         extended_roi_x, extended_roi_y = (None, None)
    analyzer = ImageStatistics.attach(image0, bias=args.bias, dark=args.dark)
    analyzer.run()
    aver, mdn, std = analyzer.mean() , analyzer.median(), analyzer.std()
    log.info("section %s average is %s", roi, aver)
    log.info("section %s stddev is %s", roi, std)
    log.info("section %s median is %s", roi, mdn)
    metadata = image0.metadata()
    roi = image0.roi()
    title = make_plot_title_from(f"{metadata['name']}", metadata, roi)
    mpl_main_image_loop(
        title    = title,
        channels = channels,
        roi = roi,
        plot_func = plot_image,
        pixels    = pixels,
        # Extra arguments
        mean = aver,
        median = mdn,
        stddev = std,
        extended_roi_x = extended_roi_x,
        extended_roi_y = extended_roi_y,
    )


def image_optical(args):
    file_path, roi, n_roi, channels, metadata, simulated, image0 = common_info_with_sim(args)
    simulated = args.sim_dark is not None or args.sim_read_noise is not None
    # The pixels we need to display are those of the whole image, not the ROI
    pixels = ImageLoaderFactory().image_from(file_path, FULL_FRAME_NROI, channels, 
        simulated=simulated, 
        read_noise=args.sim_read_noise,
        dark_current=args.sim_dark
    ).load()

    height, width = image0.shape()
    roi_x, roi_y = extended_roi(roi, width, height)
   
    # produce the Horizontal aggregate from 0...ncols
    pixels_x = pixels[:,roi_x.y0:roi_x.y1, roi_x.x0:roi_x.x1]
    H = np.mean(pixels_x, axis=1)  
    Z, N = H.shape
    X = np.tile(np.arange(0, N),(Z,1))
    
    # produce the Vertical aggregate from 0...rows
    pixels_y = pixels[:,roi_y.y0:roi_y.y1, roi_y.x0:roi_y.x1]
    V = np.mean(pixels_y, axis=2)
    Z, M = V.shape
    Y = np.tile(np.arange(0, M),(Z,1)) 
    
    # Calculate the center fo gravity 
    # of these marginal distrubutions H & V
    xc = np.sum(X*H, axis=1)/np.sum(H, axis=1)
    yc = np.sum(Y*V, axis=1)/np.sum(V, axis=1)
    log.info("Centroid Xc = %s",xc)
    log.info("Centroid Yc = %s",yc)
    centroid = (xc, yc)
    # Caluclate the geometrical center for all channels
    ocx = np.tile(np.array([width / 2]),(Z,1))
    ocy = np.tile(np.array([height / 2]),(Z,1))
    
    title = make_plot_no_roi_title_from(f"{metadata['name']}", metadata)

    mpl_main_pairs_plot_loop(
        title    = title,
        plot_func = plot_radial,
        xtitle = "Pixel coordinates",
        ytitle = "Mean PV",
        x     = (X, Y),
        y     = (H, V),
        ylabel = "good",
        channels = channels,
        # Extra arguments
        centroid = (xc, yc),
        geom_center = (ocx, ocy),
        roi_x = roi_x,
        roi_y = roi_y,
    )
 


COMMAND_TABLE = {
    'pixels': image_pixels,
    'histo': image_histo, 
    'optical': image_optical,
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
    parser_optical = subparser.add_parser('optical', help='Display image aggregate X & Y axes for optical misalighment analysis')

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
    parser_pixels.add_argument('--extended-roi', action='store_true', help='Plot X & Y extended ROIs')
    parser_pixels.add_argument('-bi', '--bias',  type=vflopath,  help='Bias, either a single value for all channels or else a 3D FITS cube file (default: %(default)s)')
    parser_pixels.add_argument('-dk', '--dark',  type=vfloat,  help='Dark count rate in DN/sec. (default: %(default)s)')
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

    # -----------------------
    # Optical command parsing
    # -----------------------
    parser_optical.add_argument('-i', '--input-file', type=vfile, required=True, help='Input RAW file')
    parser_optical.add_argument('-x', '--x0', type=vfloat01,  help='Normalized ROI start point, x0 coordinate [0..1]')
    parser_optical.add_argument('-y', '--y0', type=vfloat01,  help='Normalized ROI start point, y0 coordinate [0..1]')
    parser_optical.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1] (default: %(default)s)')
    parser_optical.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1] (default: %(default)s)')
    parser_optical.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser_optical.add_argument('-bi', '--bias',  type=vflopath,  help='Bias, either a single value for all channels or else a 3D FITS cube file (default: %(default)s)')
    parser_optical.add_argument('-dk', '--dark',  type=vfloat,  help='Dark count rate in DN/sec. (default: %(default)s)')
    parser_optical.add_argument('--sim-dark', type=float,  help='Generate synthetic dark frame with given dark count rate [DN/sec]')
    parser_optical.add_argument('--sim-read-noise', type=float,  help='Generate synthetic dark frame with given readout noise [DN]')



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
