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

import functools

# --------------------------
# Matplotlib related imports
# --------------------------

import numpy as np
import matplotlib.pyplot as plt

# ---------
# Constants
# ---------
IMAGE_CMAP = { 'R': "hot", 'G': "summer", 'Gr': "summer", 'Gb': "summer", 'B': "winter"}
CONTOUR_CMAP = { 'R': "prism", 'G': "prism", 'Gr': "prism", 'Gb': "flag", 'B': "flag"}
EDGE_COLOR = { 'R': "y", 'G': "b", 'Gr': "b", 'Gb': "b", 'B': "r"}
LAYOUT = { 1: (1,1), 2: (1,2), 3: (2,2), 4: (2,2)}

import logging
log = logging.getLogger(__name__)

# ------------------------
# Module utility functions
# ------------------------

def plot_contour_cmap(channels):
    '''Plot image color map of channels to display'''
    return [CONTOUR_CMAP[ch] for ch in channels]

def plot_image_cmap(channels):
    '''Plot image color map of channels to display'''
    return [IMAGE_CMAP[ch] for ch in channels]

def plot_edge_color(channels):
    '''Plot color map of channels to display'''
    return [EDGE_COLOR[ch] for ch in channels]

def plot_layout(channels):
    '''Plot layout dimensions  as a function of channels to display'''
    # returns (nrows, ncols)
    return LAYOUT[len(channels)]

def axes_reshape(axes, channels):
    '''Reshape Axes to be 2D arrays for 1x1 and 1x2 layout situations'''
    if len(channels) == 1:
        return np.array([axes]).reshape(-1,1)
    if len(channels) == 2:
        return axes.reshape(-1,2)
    return axes

def mpl_main_image_loop(title, pixels, plot_func, channels, roi, **kwargs):
    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols)
    fig.suptitle(title)
    axes = axes_reshape(axes, channels)
    for row in range(0,display_rows):
        for col in range(0,display_cols):
            i = 2*row+col
            if len(channels) == 3 and row == 1 and col == 1: # Skip the empty slot in 2x2 layout with 3 items
                axes[row][col].set_axis_off()
                break
            cmap = plot_image_cmap(channels)
            edge_color = plot_edge_color(channels)
            plot_func(axes[row][col], i,  pixels[i], channels, roi, **kwargs)
    plt.show()

def mpl_main_plot_loop(title, x, y, xtitle, ytitle, plot_func, channels, ylabel=None, **kwargs):
    display_rows, display_cols = plot_layout(channels)
    fig, axes = plt.subplots(nrows=display_rows, ncols=display_cols)
    fig.suptitle(title)
    axes = axes_reshape(axes, channels)
    for row in range(0,display_rows):
        for col in range(0,display_cols):
            i = 2*row+col
            if len(channels) == 3 and row == 1 and col == 1: # Skip the empty slot in 2x2 layout with 3 items
                axes[row][col].set_axis_off()
                break
            plot_func(axes[row][col], i, x, y, xtitle, ytitle, ylabel, channels, **kwargs)
    plt.show()
