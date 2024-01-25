# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2021
#
# See the LICENSE file for details
# see the AUTHORS file for authors
# ----------------------------------------------------------------------

# -------------------
# System wide imports
# -------------------

import math
import numpy as np

# --------------------------
# Matplotlib related imports
# --------------------------

import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------
# Constants
# ---------

CMAP = { 'R': "hot", 'G': "summer", 'Gr': "summer", 'Gb': "summer", 'B': "winter"}
EDGE_COLOR = { 'R': "y", 'G': "b", 'Gr': "b", 'Gb': "b", 'B': "r"}
LAYOUT = { 1: (1,1), 2: (1,2), 3: (2,2), 4: (2,2)}

# ------------------------
# Module utility functions
# ------------------------

def plot_linear_equation(axes, xdata, ydata, slope, intercept, xlabel='x', ylabel='y'):
    angle = math.atan(slope)*(180/math.pi)
    x0 = np.min(xdata); x1 = np.max(xdata)
    y0 = np.min(ydata); y1 = np.max(ydata)
    x = x0 + 0.35*(x1-x0)
    y = y0 + 0.45*(y1-y0)
    text = f"${ylabel} = {slope:.2f}{xlabel}{intercept:+.2f}$"
    axes.text(x, y, text,
        rotation_mode='anchor',
        rotation=angle,
        transform_rotates_text=True,
        ha='left', va='top')

def plot_cmap(channels):
    '''Plot color map of channels to display'''
    return [CMAP[ch] for ch in channels]

def plot_edge_color(channels):
    '''Plot color map of channels to display'''
    return [EDGE_COLOR[ch] for ch in channels]

def plot_layout(channels):
    '''Plot layout dimensions  as a fuction of channels to display'''
    # returns (nrows, ncols)
    l = len(channels)
    return LAYOUT[l]

def plot_image(fig, axes, color_plane, roi, title, average, median, stddev, colormap, edgecolor):
    axes.set_title(fr'{title}: $median={median:.2f}, \mu={average:.2f},\;\sigma={stddev:.2f}$')
    im = axes.imshow(color_plane, cmap=colormap)
    # Create a Rectangle patch
    rect = patches.Rectangle(roi.xy(), roi.width(), roi.height(), 
                    linewidth=1, linestyle='--', edgecolor=edgecolor, facecolor='none')
    axes.add_patch(rect)
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.10)
    fig.colorbar(im, cax=cax, orientation='vertical')


def plot_histo(axes, color_plane, title, decimate, average, median, stddev):
    axes.set_title(fr'channel {title}: $median={median:.2f}, \mu={average:.2f},\;\sigma={stddev:.2f}$')
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
  
def axes_reshape(axes, channels):
    '''Reshape Axes to be 2D arrays for 1x1 and 1x2 layout situations'''
    if len(channels) == 1:
        return [[axes]]
    if len(channels) == 2:
        return axes.reshape(1,2)
    return axes