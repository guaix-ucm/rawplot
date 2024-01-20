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

# ---------
# Constants
# ---------

CMAP = { 'R': "hot", 'G': "summer", 'G1': "summer", 'G2': "summer", 'B': "winter"}
EDGE_COLOR = { 'R': "y", 'G': "b", 'G1': "b", 'G2': "b", 'B': "r"}
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
  
def axes_reshape(axes, channels):
    '''Reshape Axes to be 2D arrays for 1x1 and 1x2 layout situations'''
    if len(channels) == 1:
        return [[axes]]
    if len(channels) == 2:
        return axes.reshape(1,2)
    return axes