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

import math
import logging

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from lica.cli import execute
from lica.validators import vfile, vfloat01, valid_channels
from lica.raw.loader import ImageLoaderFactory, SimulatedDarkImage, NormRoi


# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.mpl.plot import plot_layout, axes_reshape

from .util.mpl.plot import mpl_main_plot_loop
from .util.common import common_info, bias_from, make_plot_title_from

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# ------------------
# Auxiliary fnctions
# ------------------

def plot_hv(axes, xh, xv, H, V, title, log2):
    base = 2 if log2 else 10
    axes.set_title(f'Channel {title}')
    axes.set_yscale('log', base=base)
    axes.plot(xh, H, label='Horizontal')
    axes.plot(xv, V, label='Vertical')
    axes.set_xlabel('Cycles per pixel pitch [c/p]')
    axes.set_ylabel('Average Energy Spectrum')
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.legend()

def averaged_energy_spectrum(file_path, roi, n_roi, channels, metadata, start):
    pixels = ImageLoaderFactory().image_from(file_path, n_roi, channels).load()
    Z, ROWS, COLS = pixels.shape
    # To remove the DC component it is more effective
    # to take the mean from the image itself rather than
    # using the embedded EXIF black levels
    aver_img =  np.mean(pixels, axis=(1,2))
    log.info("Pixels mean values are %s", aver_img)
    pixels = pixels - aver_img.reshape(Z, 1, 1) # Reduce the (0,0) DC peak in the FFT2
    fft2 = np.fft.fft2(pixels)
    log.info("FFT2 Stack shape is %s", fft2.shape)
    power_spectrum = np.power(np.abs(fft2), 2)
    log.info("Power Spectrum Stack shape is %s", power_spectrum.shape)
    aver_pe =  np.mean(power_spectrum,  axis=(1,2))
    log.info("Before normalization, average Power Spectrum is %s", aver_pe)
    power_spectrum = power_spectrum / aver_pe.reshape(Z,1,1)
    # For all the color planes in the stack
    # To calculate the average of each column, use axis=1
    # Then slice to the proprer range
    H = np.mean(power_spectrum, axis=1)[:,start:COLS//2+1]    
    # To calculate the average of each row, use axis=2.
    V = np.mean(power_spectrum, axis=2)[:,start:ROWS//2+1]
    xh = np.arange(start, COLS//2+1)/COLS # Normalized x for H plot
    xv = np.arange(start, ROWS//2+1)/ROWS # Normalized x for V plot
    return xh, xv, H, V

# ------------------------
# AUXILIARY MAIN FUNCTIONS
# ------------------------

def hv(args):
    log2 = args.log2
    file_path, roi, n_roi, channels, metadata = common_info(args)
    xh, xv, H, V = averaged_energy_spectrum(file_path, roi, n_roi, channels, metadata, args.start)
    title = make_plot_title_from(f"Image: {metadata['name']}", metadata, roi)
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
            plot_hv(axes[row][col], xh, xv, H[i], V[i], channels[i], log2)
    plt.show()


# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):
    parser.add_argument('-i', '--input-file', type=vfile, required=True, help='Input RAW file')
    parser.add_argument('-x', '--x0', type=vfloat01, default=None, help='Normalized ROI start point, x0 coordinate [0..1]')
    parser.add_argument('-y', '--y0', type=vfloat01, default=None, help='Normalized ROI start point, y0 coordinate [0..1]')
    parser.add_argument('-wi', '--width',  type=vfloat01, default=1.0, help='Normalized ROI width [0..1] (default: %(default)s)')
    parser.add_argument('-he', '--height', type=vfloat01, default=1.0, help='Normalized ROI height [0..1] (default: %(default)s)')
    parser.add_argument('-c','--channels', choices=('R', 'Gr', 'Gb', 'G', 'B'), default=('R','Gr','Gb','B'), nargs='+', 
                    help='color plane(s) to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser.add_argument('-s', '--start', type=int, default=0, help='Index to trim power spectrum DC component (recommended value between 2..4) (default: %(default)s)')
    parser.add_argument('--log2',  action='store_true', help='Display plot using log2 instead of log10 scale')
    parser.add_argument('--sim-dark', type=float, default=None, help='Simulate dark frame with given dark current')
    
# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=hv, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="HV Spectrogram plot"
        )