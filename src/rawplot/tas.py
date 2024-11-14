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

import math
import logging
from typing import Tuple, Any
from itertools import groupby
from collections.abc import Iterable
from argparse import Namespace

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt

from lica.cli import execute
from lica.validators import vfile, vsexa

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__

# ----------------
# Module constants
# ----------------

# Photodiode readings header columns
WAVELENGTH_CSV_HEADER = "wavelength (nm)"
CURRENT_CSV_HEADER = "current (A)"
READ_NOISE_CSV_HEADER = "read noise (A)"
FREQUENCY_CSV_HEADER = "frequency (Hz)"
QE_CSV_HEADER = "quantum efficiency"
RESP_CSV_HEADER = "responsivity"

CUTOFF_FILTERS = [
    {"label": r"$BG38 \Rightarrow OG570$", "wave": 570, "style": "--"},
    {"label": r"$OG570\Rightarrow RG830$", "wave": 860, "style": "-."},
]

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


def mpl_tas_plot_loop(
    wavelength: np.ndarray,
    signal: np.ndarray,
    model: str,
    sensor: str,
    responsivity: bool,
    normalized: bool,
    filters: Iterable[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=1)
    # fig.suptitle("Corrected Spectral Response plot")
    axes.set_xlabel("Wavelength [nm]")
    axes_title = "Responsivity" if responsivity else "Quantum Efficiency"
    axes.set_title(axes_title)
    units = "(normalized)" if normalized else ""
    axes.set_ylabel(f"Signal {units}")
    axes.plot(wavelength, signal, marker="+", linewidth=1, color="blue", label=f"TESS-W ({sensor})")
    for filt in filters:
        axes.axvline(filt["wave"], linestyle=filt["style"], label=filt["label"])
    axes.grid(True, which="major", color="silver", linestyle="solid")
    axes.grid(True, which="minor", color="silver", linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.legend()
    plt.show()



def map_fields(line: list[str]) -> dict[str, Any]:
    result = dict()
    result["tstamp"] = datetime.datetime.strptime(line[0] + " " + line[1], "%Y-%m-%d %H:%M:%S")
    result["temp_sky"] = float(line[2])
    result["temp_box"] = float(line[3])
    result["magnitude"] = float(line[4])
    result["frequency"] = float(line[5])
    result["altitude"] = float(line[6])
    result["azimuth"] = float(line[7])
    result["latitude"] = vsexa(line[8])
    result["longitude"] = vsexa(line[9])
    result["msnm"] = float(line[10])
    return result


def read_tas_file(path: str) -> Any:
    with open(path, "r") as csvfile:
        lines = [line[1:] for line in csv.reader(csvfile, delimiter="\t")][1:]
    return list(map(map_fields, lines))


# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------


def tas(args: Namespace):
    result = read_tas_file(args.input_file)
    log.info(result)


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
