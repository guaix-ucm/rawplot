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
import functools

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

from mpl_toolkits.axes_grid1 import make_axes_locatable


from lica.cli import execute
from lica.validators import vfile, vfloat, vfloat01, vflopath, voddint
from lica.raw.loader import ImageLoaderFactory, FULL_FRAME_NROI
from lica.raw.analyzer.image import ImageStatistics

from astropy.convolution import convolve, Box2DKernel

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.mpl.plot import mpl_main_image_loop, mpl_main_plot_loop, plot_contour_cmap, plot_image_cmap, plot_edge_color
from .util.common import common_info, common_info_with_sim, make_plot_title_from, make_plot_no_roi_title_from
from .util.common import extended_roi, geom_center, marginal_distributions, centroid

# ----------------
# Module constants
# ----------------

MIN_CONTOUR_LEVEL = 0.05
MAX_CONTOUR_LEVEL = 0.80
PREDEFINED_CONTOUR_LEVELS = (0.05, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)

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
# Auxiliary functions
# ------------------

voddint_3_11 = functools.partial(voddint, 3, 11)



def plot_radial(axes, i, x, y, xtitle, ytitle, ylabel, channels, **kwargs):
    xc, yc = kwargs['centroid']
    gcx, gcy = kwargs['geom_center']
    width, height = kwargs['bounds']
    axes.set_xlabel(xtitle)
    axes.set_ylabel(ytitle)
    axes.plot(x[0][i], y[0][i], label="H")
    axes.plot(x[1][i], y[1][i], label = "V")
    axes.axvline(xc[i], linestyle='--', label="H opti.")
    axes.axvline(yc[i], linestyle='--', label="V opti.")
    axes.axvline(gcx[i], linestyle=':', color='tab:orange', label="H geom.")
    axes.axvline(gcy[i], linestyle=':', color='tab:orange', label="V geom.")
    title = f'{channels[i]}: Aggregate of central {width} rows (H), {height} cols (V)'
    axes.set_title(title)
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.legend()
  

def plot_histo(axes, i, x, y, xtitle, ytitle, ylabel, channels, **kwargs):
    median = kwargs['median'][i]
    mean = kwargs['mean'][i]
    stddev = kwargs['stddev'][i]
    decimate = kwargs.get('decimate', 10)
    ylog = kwargs.get('ylog', False)
    title = fr'{channels[i]}: median={median:.2f}, $\mu={mean:.2f}, \sigma={stddev:.2f}$'
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
  
def plot_image(axes, i, pixels, channels, roi, **kwargs):
    median = kwargs['median'][i]
    mean = kwargs['mean'][i]
    stddev = kwargs['stddev'][i]
    gcx, gcy = kwargs['geom_center']
    img_cmap = plot_image_cmap(channels)[i]
    edgecolor = plot_edge_color(channels)[i]
    title = fr'{channels[i]}: median={median:.2f}, $\mu={mean:.2f}, \sigma={stddev:.2f}$'
    axes.set_title(title)
    im = axes.imshow(pixels, cmap=img_cmap)
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
    axes.plot(gcx[i],gcy[i], marker='x', label='Center')
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.10)
    axes.get_figure().colorbar(im, cax=cax, orientation='vertical')

def plot_contour(axes, i, pixels, channels, roi, **kwargs):
    levels =  kwargs['levels']
    labels =  kwargs['labels']
    gcx, gcy = kwargs['geom_center']
    img_cmap = plot_image_cmap(channels)[i]
    ctr_cmap = plot_contour_cmap(channels)[i]
    edgecolor = plot_edge_color(channels)[i]
    title = fr'{channels[i]}: Normalized contour levels'
    axes.set_title(title)
    im = axes.imshow(pixels, cmap=img_cmap)
    # Create the contour
    CS = axes.contour(pixels, levels=levels, norm='linear', cmap=ctr_cmap)
    if labels:
        axes.clabel(CS, inline=True, fontsize=16)
    # Create more Rectangle patches for optional extended ROIs
    for key in ('extended_roi_x', 'extended_roi_y'):
        extended_roi = kwargs.get(key)
        if extended_roi:
            rect = patches.Rectangle(extended_roi.xy(), extended_roi.width(), extended_roi.height(), 
                        linewidth=1, linestyle=':', edgecolor=edgecolor, facecolor='none')
            axes.add_patch(rect)
    axes.plot(gcx[i],gcy[i], marker='x', label='Center')
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.10)
    fig = axes.get_figure()
    fig.colorbar(im, cax=cax, orientation='vertical')
   


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
    Z, M, N = pixels.shape
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
    # Calculate the geometrical center for all channels
    gcx, gcy = geom_center(pixels, channels)
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
        geom_center = (gcx, gcy),
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
    Z, M, N = pixels.shape
    offset_x = abs(N-M)
    (X, H), (Y, V) = marginal_distributions(pixels, roi, image0.black_levels())
    Y = Y + offset_x # For plotting purposes
    xc, yc = centroid((X,H), (Y,V))
    gcx, gcy = geom_center(pixels, channels)
    gcy += offset_x # For plotting purposes
    title = make_plot_no_roi_title_from(f"{metadata['name']}", metadata)
    mpl_main_plot_loop(
        title    = title,
        plot_func = plot_radial,
        xtitle = "Pixel coord.",
        ytitle = "Normalized PV",
        x     = (X, Y),
        y     = (H, V),
        ylabel = "good",
        channels = channels,
        # Extra arguments
        centroid = (xc, yc),
        geom_center = (gcx, gcy),
        bounds = (roi.width(), roi.height())
    )
 

def image_contour(args):
    file_path, roi, n_roi, channels, metadata, _, image0 = common_info(args)
    # The pixels we need to display are those of the whole image, not the ROI
    pixels = ImageLoaderFactory().image_from(file_path, FULL_FRAME_NROI, channels, 
        simulated=False,
    ).load()
    # calculate contour levels
    if args.levels is None:
         levels = PREDEFINED_CONTOUR_LEVELS
    else:
        levels = np.round(np.linspace(MIN_CONTOUR_LEVEL, MAX_CONTOUR_LEVEL, num=args.levels, endpoint=True), decimals=2)
    log.info(levels)
    # ROI statistics over the original values
    Z, M, N = pixels.shape
    bias = np.array(image0.black_levels()).reshape(Z,-1)
    analyzer = ImageStatistics.attach(image0)
    analyzer.run()
    aver, mdn, std = analyzer.mean() , analyzer.median(), analyzer.std()
    log.info("section %s average is %s", roi, aver)
    log.info("section %s stddev is %s", roi, std)
    log.info("section %s median is %s", roi, mdn)
    metadata = image0.metadata()
    # Optionally smooths input image with a 2D kernel
    if args.filter:
        size = args.filter
        kernel = Box2DKernel(size, mode='center')
        log.info('Smoothing image with a %dx%d box 2D kernel',size,size)
        filtered = [convolve(pixels[i], kernel, boundary='extend') for i in range(0,Z)]
        pixels = np.stack(filtered, axis=0)
    # Normalize PV
    pixels = (pixels - bias) / np.max(pixels, axis=(1,2)).reshape(Z,1,1)
    gcx, gcy = geom_center(pixels, channels)
    title = make_plot_no_roi_title_from(f"{metadata['name']}", metadata)
    mpl_main_image_loop(
        title     = title,
        channels  = channels,
        plot_func = plot_contour,
        roi = roi,
        pixels    = pixels,
        # Extra arguments
        levels    = levels,
        labels    = args.labels,
        geom_center = (gcx, gcy),
    )

COMMAND_TABLE = {
    'pixels': image_pixels,
    'histo': image_histo, 
    'optical': image_optical,
    'contour': image_contour,
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
    parser_optical = subparser.add_parser('optical', help='Display X & Y axes for optical misaligment analysis')
    parser_contour = subparser.add_parser('contour', help='Display contour graph for optical analysis')

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

    # -----------------------
    # Contour command parsing
    # -----------------------
    parser_contour.add_argument('-i', '--input-file', type=vfile, required=True, help='Input RAW file')
    parser_contour.add_argument('-l', '--levels', metavar='<N>', type=int,  help='Contour levels to apply')
    parser_contour.add_argument('-x', '--x0', type=vfloat01,  help='Normalized ROI start point, x0 coordinate [0..1]')
    parser_contour.add_argument('-y', '--y0', type=vfloat01,  help='Normalized ROI start point, y0 coordinate [0..1]')
    parser_contour.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1] (default: %(default)s)')
    parser_contour.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1] (default: %(default)s)')
    parser_contour.add_argument('-c','--channels', default=['R', 'Gr', 'Gb','B'], nargs='+',
                    choices=['R', 'Gr', 'Gb', 'G', 'B'],
                    help='color plane to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser_contour.add_argument('--filter', metavar='<N>', type=voddint_3_11,  help='Apply a 2D Box filter to contour image of size N')
    parser_contour.add_argument('--labels',  action='store_true', help='Show Contour labels')
 

   

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
