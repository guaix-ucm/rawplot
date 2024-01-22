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

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.validators import vfile, vfloat01, valid_channels
from .util.cli import execute
from .util.rawimage import RawImage, SimulatedDarkImage
from .util.mpl import plot_layout, axes_reshape

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# ------------------
# Auxiliary fnctions
# ------------------

def plot_hv(axes, xh, xv, H, V, title):
    def stops(x, pos): return f'{math.log2(x):.0f}'
    axes.set_title(f'Channel {title}')
    axes.set_yscale('log', base=2)
    #axes.yaxis.set_major_formatter(ticker.FuncFormatter(stops))
    axes.plot(xh, H, label='Horizontal')
    axes.plot(xv, V, label='Vertical')
    axes.set_xlabel('Cycles per pixel pitch [c/p]')
    axes.set_ylabel('Average Energy Spectrum [stops]')
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.legend()


# ------------------------
# AUXILIARY MAIN FUNCTIONS
# ------------------------

def hv(args):
    channels = valid_channels(args.channels)
    if args.sim_dark is not None:
        image = SimulatedDarkImage(args.input_file, dk_current=args.sim_dark)
    else:
        image = RawImage(args.input_file)
    roi = image.roi(args.x0, args.y0, args.width, args.height)
    stack = image.debayered(roi, channels)
    log.info("Stack shape is %s", stack.shape)
    Z, ROWS, COLS = stack.shape
    # Taking the mean from the image itself is more effective removing the DC component
    # than using the black levels
    aver_img =  np.mean(stack, axis=(1,2))
    log.info("Stack mean values are %s", aver_img)
    stack = stack - aver_img.reshape(Z, 1, 1) # Take out the avergae value to reduce the (0,0) DC peak in the FFT2
    fft2 = np.fft.fft2(stack)
    log.info("FFT2 Stack shape is %s", fft2.shape)
    power_spectrum = np.power(np.abs(fft2), 2)
    log.info("Power Spectrum Stack shape is %s", power_spectrum.shape)
    aver_pe =  np.mean(power_spectrum,  axis=(1,2))
    log.info("Before normalization, average Power Spectrum shape is %s", aver_pe)
    power_spectrum = power_spectrum / aver_pe.reshape(Z,1,1)
    start=args.start
    # For all the color planes in the stack
    # To calculate the average of each column, use axis=1
    # Then slice to the proprer range
    H = np.mean(power_spectrum, axis=1)[:,start:COLS//2+1]    
    # To calculate the average of each row, use axis=2.
    V = np.mean(power_spectrum, axis=2)[:,start:ROWS//2+1]
    xh = np.arange(start, COLS//2+1)/COLS # Normalized abscissa for H plot
    xv = np.arange(start, ROWS//2+1)/ROWS # Normalized abscissa for V plot
    log.info("H shape is %s", H.shape)
    log.info("V shape is %s", V.shape)
    log.info("xv shape is %s", xv.shape)
    log.info("xh shape is %s", xh.shape)

    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(12, 9), layout='tight')
    metadata = image.exif()
    fig.suptitle(f"Image: {image.name()}\n"
            f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}, Exposure: {metadata['exposure']} [s]\n"
            f"Color Plane Size: {image.shape()[0]} rows x {image.shape()[1]} cols\n" 
            f"ROI Section: {roi} {roi.height()} rows x {roi.width()} cols")
    axes = axes_reshape(axes, channels)
    for row in range(0,display_rows):
        for col in range(0,display_cols):
            i = 2*row+col
            if len(channels) == 3 and row == 1 and col == 1: # Skip the empty slot in 2x2 layout with 3 items
                axes[row][col].set_axis_off()
                break
            plot_hv(axes[row][col], xh, xv, H[i], V[i], channels[i])
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
    parser.add_argument('-c','--channels', choices=['R', 'G1', 'G2', 'G', 'B'], default=['R','G1','G2','B'], nargs='+', 
                    help='color plane(s) to plot. G is the average of G1 & G2. (default: %(default)s)')
    parser.add_argument('-s', '--start', type=int, default=0, help='(Optional) Index to trim power spectrum DC component (recommended value between 2..4')
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