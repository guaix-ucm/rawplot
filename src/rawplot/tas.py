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
import pathlib

import logging
import functools
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

import astropy.time
import astropy.coordinates
import astropy.timeseries
import astropy.units as u
from astropy.timeseries import TimeSeries

import pytz
from timezonefinder import TimezoneFinder

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


def get_timezone(line: list[str]) -> str:
    """Returns the time zone string from a line of TAS readings"""
    latitude = vsexa(line[9])  # decimal degrees
    longitude = vsexa(line[10])  # decimal degrees
    return TimezoneFinder().timezone_at(lng=longitude, lat=latitude)


def map_fields2(timezone: datetime.tzinfo, line: list[str]) -> dict[str, Any]:
    """Each line of TAS CSV is converted to a dictionary with proper basic Python data types"""
    result = dict()
    result["sequence"] = int(line[0])  # scanned point sequence number
    result["time"] = datetime.datetime.strptime(
        line[1] + "T" + line[2], "%Y-%m-%dT%H:%M:%S"
    ).replace(tzinfo=timezone)
    result["temp_sky"] = float(line[3])  # degrees Celsius
    result["temp_box"] = float(line[4])  # degrees Celsius
    result["magnitude"] = float(line[5])  # mag/arcsec^2
    result["frequency"] = float(line[6])  # Hz
    result["zenith"] = 90 - float(line[7])  # decimal degrees
    result["azimuth"] = float(line[8])  # decimal degrees
    result["latitude"] = vsexa(line[9])  # decimal degrees
    result["longitude"] = vsexa(line[10])  # decimal degrees
    result["height"] = float(line[11])  # meters above sea level
    return result


def tas_metadata(rows: Tuple[dict[str, Any]]) -> dict[str, Any]:
    """Calculate common metadata for the whole dataset"""
    metadata = dict()
    N = len(rows)
    metadata["mid_timestamp"] = rows[N // 2]["time"]  # Timestamp at mid exposure.
    metadata["mean_height"] = np.mean(np.array([item["height"] for item in rows]))
    metadata["mean_longitude"] = np.mean(np.array([item["longitude"] for item in rows]))
    metadata["mean_latitude"] = np.mean(np.array([item["latitude"] for item in rows]))
    metadata["mean_temp_box"] = np.mean(np.array([item["temp_box"] for item in rows]))
    metadata["mean_temp_sky"] = np.mean(np.array([item["temp_sky"] for item in rows]))
    return metadata


def to_astropy(row):
    """Use specialized Astropy Types for some columns"""
    row["longitude"] = astropy.coordinates.Longitude(row["longitude"] * u.deg)
    row["latitude"] = astropy.coordinates.Latitude(row["latitude"] * u.deg)
    row["azimuth"] = astropy.coordinates.Angle(row["azimuth"] * u.deg)
    row["zenith"] = astropy.coordinates.Angle(row["zenith"] * u.deg)
    row["time"] = astropy.time.Time(row["time"])
    return row


def read_tas_file(path: str) -> Tuple[TimeSeries, dict[str, Any]]:
    with open(path, "r") as csvfile:
        lines = [line for line in csv.reader(csvfile, delimiter="\t")][1:]
    timezone = get_timezone(lines[0])  # get the timezone from the first line
    tzinfo = pytz.timezone(timezone)
    map_fields = functools.partial(map_fields2, tzinfo)
    # Low level decoding to Python standard datatups
    rows = tuple(map(map_fields, lines))
    metadata = tas_metadata(rows)
    metadata["timezone"] = timezone
    metadata["file"] = str(pathlib.Path(path).resolve())
    # Convert to Astropy Types to get extra benefits
    rows = tuple(map(to_astropy, rows))
    table = astropy.timeseries.TimeSeries(
        rows=rows,
        names=(
            "time",
            "sequence",
            "zenith",
            "azimuth",
            "frequency",
            "magnitude",
            "temp_sky",
            "temp_box",
            "longitude",
            "latitude",
            "height",
        ),
        units=(
            None,
            None,
            u.deg,
            u.deg,
            u.Hz,
            u.mag() / u.arcsec**2,
            u.deg_C,
            u.deg_C,
            u.deg,  # Not really needed, Longitude already carries the unit
            u.deg,  # Not really needed, Latitude already carries the unit
            u.m,
        ),
    )
    return table, metadata


# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------


def tas(args: Namespace):
    data, metadata = read_tas_file(args.input_file)
    # zenith_grid = np.arange(0, np.max(data["zenith"]) + 1)  # grid values in zenith direction
    # azimuth_grid = np.arange(0, 360 + 1)  # grid values in azimuth direction

    log.info(data.info)
    log.info(metadata)


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
