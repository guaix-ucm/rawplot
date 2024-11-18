# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2021
#
# See the LICENSE file for details
# see the AUTHORS file for authors
# ----------------------------------------------------------------------

# --------------------
# System wide imports
# -------------------

import csv
import datetime

import logging
from typing import Tuple, Any
from argparse import Namespace

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from lica.cli import execute
from lica.validators import vfile, vsexa

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__

# ----------------
# Module constants
# ----------------

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# -----------------
# Matplotlib styles
# -----------------

# Load global style sheets
plt.style.use("rawplot.resources.global")

# -------------------
# Auxiliary functions
# -------------------


def mycolormap() -> mcolors.LinearSegmentedColormap:
    """Combine two color maps to produce a new one"""
    colors2 = plt.cm.viridis_r(np.linspace(0.0, 1, 192))
    colors1 = plt.cm.YlOrRd_r(np.linspace(0, 1, 64))
    colors = np.vstack((colors1, colors2))
    return mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)


def mpl_tas_plot_loop(
    azimuth: np.ndarray,
    zenith: np.ndarray,
    magnitude: np.ndarray,
    title: str,
    min_mag: float,
    max_mag: float,
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=1, subplot_kw={"projection": "polar"})
    # fig.suptitle("Corrected Spectral Response plot")

    axes.set_theta_zero_location("N")  # Set the north to the north
    axes.set_theta_direction(-1)
    axes.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontdict={"fontsize": 16})
    axes.tick_params(pad=1.2)
    axes.set_ylim(0, max(zenith) + 0.0)
    axes.set_title(title)
    cmap = mycolormap()
    cax = axes.scatter(
        np.radians(azimuth), zenith, c=magnitude, cmap=cmap, vmin=min_mag, vmax=max_mag
    )

    # Colorbar
    cb = fig.colorbar(cax, orientation="horizontal", fraction=0.048, ticks=lev_f_ticks, pad=0.08)
    cb.set_label("Sky Brightness [mag/arcsec$^2$]", fontsize=17)
    cb.ax.tick_params(labelsize=12)
    plt.show()


def map_fields(line: list[str]) -> dict[str, Any]:
    """Each line of TAS CSV is converted to a dictionary with proper data types"""
    result = dict()
    result["sequence"] = int(line[0])  # scanned point sequence number
    result["timestamp"] = datetime.datetime.strptime(line[1] + " " + line[2], "%Y-%m-%d %H:%M:%S")
    result["temp_sky"] = float(line[3])  # degrees Celsius
    result["temp_box"] = float(line[4])  # degrees Celsius
    result["magnitude"] = float(line[5])  # mag/arcsec^2
    result["frequency"] = float(line[6])  # Hz
    result["altitude"] = float(line[7])  # decimal degrees
    result["azimuth"] = float(line[8])  # decimal degrees
    result["latitude"] = vsexa(line[9])  # decimal degrees
    result["longitude"] = vsexa(line[10])  # decimal degrees
    result["height"] = float(line[11])  # meters above sea level
    return result


def read_tas_file(path: str) -> Tuple[dict[str, Any]]:
    with open(path, "r") as csvfile:
        lines = [line for line in csv.reader(csvfile, delimiter="\t")][1:]
    temp_data = tuple(map(map_fields, lines))
    data = dict()
    metadata = dict()
    data["sequence"] = np.array([item["sequence"] for item in temp_data])
    data["zenith"] = np.array([90 - item["altitude"] for item in temp_data])
    data["azimuth"] = np.array([item["azimuth"] for item in temp_data])
    data["frequency"] = np.array([item["frequency"] for item in temp_data])
    data["magnitude"] = np.array([item["magnitude"] for item in temp_data])
    data["temp_sky"] = np.array([item["temp_sky"] for item in temp_data])
    metadata["timestamp"] = temp_data[len(temp_data)//2] # Timestamp at mid exposure.
    metadata["height"] = np.mean(np.array([item["height"] for item in temp_data]))
    metadata["longitude"] = np.mean(np.array([item["longitude"] for item in temp_data]))
    metadata["latitude"] = np.mean(np.array([item["latitude"] for item in temp_data]))
    metadata["temp_box"] = np.mean(np.array([item["temp_box"] for item in temp_data]))
    return data, metadata


# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------


def tas(args: Namespace):
    data, metadata = read_tas_file(args.input_file)
    log.info(data)


# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================


def add_args(parser):
    parser.add_argument(
        "-i",
        "--input-file",
        type=vfile,
        metavar="<CSV FILE>",
        required=True,
        help="CSV file with TAS readings",
    )


# ================
# MAIN ENTRY POINT
# ================


def main():
    execute(
        main_func=tas,
        add_args_func=add_args,
        name=__name__,
        version=__version__,
        description="TESS-W sensor spectral response",
    )
