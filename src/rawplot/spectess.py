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


import math
import logging
from typing import Tuple
from itertools import groupby

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt

from lica.cli import execute
from lica.validators import vfile
from lica.csv import read_csv

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.common import common_list_info, make_plot_title_from, export_spectra_to_csv
from .photodiode import photodiode_load, OSI_PHOTODIODE, HAMAMATSU_PHOTODIODE

# ----------------
# Module constants
# ----------------

# Photodiode readings header columns
WAVELENGTH_CSV_HEADER = "wavelength (nm)"
CURRENT_CSV_HEADER = "current (A)"
READ_NOISE_CSV_HEADER = "read noise (A)"

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

def mpl_spectra_plot_loop(title, x, y, xtitle, ytitle, plot_func, ylabel, **kwargs):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(title)
    axes.set_xlabel(xtitle)
    axes.set_ylabel(ytitle)
    filters = kwargs.get("filters", None)
    plot_func(axes, x, y,  **kwargs)
    if filters is not None:
        for filt in filters:
            axes.axvline(filt["wave"], linestyle=filt["style"], label=filt["label"])
    axes.grid(True, which="major", color="silver", linestyle="solid")
    axes.grid(True, which="minor", color="silver", linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.legend()
    plt.show()


def plot_raw_spectral(axes, x, y, **kwargs):
    wavelength = x
    signal = y
    color = "blue"
    marker = "o"
    axes.plot(wavelength, signal, marker=marker, color=color, linewidth=1, label="TESS-W")


def tess_readings_to_arrays(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    response = read_csv(csv_path)
    wavelength = list()
    freq_mean = list()
    freq_std = list()
    for key, grp in groupby(response, key=lambda x: x['wavelength']):
        freqs = [float(x['frequency']) for x in grp]
        freq_mean.append(np.mean(np.array(freqs)))
        freq_std.append(np.std(np.array(freqs), ddof=1))
        wavelength.append(float(key))
    log.info("Got %d TESS-W averaged frequency readings", len(wavelength))
    return np.array(wavelength), np.array(freq_mean), np.array(freq_std) 


def photodiode_readings_to_arrays(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    response = read_csv(csv_path)
    wavelength = np.array([int(entry[WAVELENGTH_CSV_HEADER]) for entry in response])
    current = np.array([math.fabs(float(entry[CURRENT_CSV_HEADER])) for entry in response])
    read_noise = np.array([float(entry[READ_NOISE_CSV_HEADER]) for entry in response])
    log.info("Got %d photodiode readings", wavelength.shape[0])
    return wavelength, current, read_noise


# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------


def raw_spectrum(args):
    log.info(" === DRAFT SPECTRAL RESPONSE PLOT === ")
    title = "Raw Spectral Response plot"
    wavelength, frequency, freq_std = tess_readings_to_arrays(args.input_file)
  
    mpl_spectra_plot_loop(
        title=title,
        plot_func=plot_raw_spectral,
        xtitle="Wavelength [nm]",
        ytitle="Signal [DN]",
        ylabel="good",
        x=wavelength,
        y=frequency,
        # Optional arguments to be handled by the plotting function
        filters=[
            {"label": r"$BG38 \Rightarrow OG570$", "wave": 570, "style": "--"},
            {"label": r"$OG570\Rightarrow RG830$", "wave": 860, "style": "-."},
        ],  # where filters were changesd
    )


def corrected_spectrum(args):
    log.info(" === COMPLETE SPECTRAL RESPONSE PLOT === ")
    responsivity, qe = photodiode_load(args.model, args.resolution)
    log.info(
        "Read %s reference responsivity values at %d nm resolution from %s",
        len(responsivity),
        args.resolution,
        args.model,
    )
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    wavelength, current, read_noise = photodiode_readings_to_arrays(args.csv_file)
    qe = np.array(
        [qe[w] for w in wavelength]
    )  # Only use those wavelenghts actually used in the CSV sequence
    current = current / np.max(current)  # Normalize photodiode current
    title = make_plot_title_from("Corrected Spectral Response plot", metadata, roi)
    wavelength = np.tile(wavelength, len(channels)).reshape(len(channels), -1)
    exptime, signal = signal_from(file_list, n_roi, channels, args.bias, args.dark, args.every)
    signal = qe * signal / current
    signal = signal / np.max(signal)  # Normalize signal to its absolute maxímun for all channels
    if args.export:
        log.info("exporting to CSV file(s)")
        export_spectra_to_csv(
            labels=channels,
            wavelength=wavelength[0],
            signal=signal,
            mode=args.export,
            units=args.units,
            wave_last=args.wavelength_last,
        )
    mpl_spectra_plot_loop(
        title=title,
        channels=channels,
        plot_func=plot_raw_spectral,
        xtitle="Wavelength [nm]",
        ytitle="Signal (normalized)",
        ylabel="good",
        x=wavelength,
        y=signal,
        # Optional arguments to be handled by the plotting function
    )


COMMAND_TABLE = {
    "raw": raw_spectrum,
    "corrected": corrected_spectrum,
}


def spectess(args):
    command = args.command
    func = COMMAND_TABLE[command]
    func(args)


# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================


def add_args(parser):
    subparser = parser.add_subparsers(dest="command")
    parser_raw = subparser.add_parser("raw", help="Raw spectrum")
    parser_corr = subparser.add_parser("corrected", help="Correced spectrum")
    # ---------------------------------------------------------------------------------------------------------------
    parser_raw.add_argument(
        "-i",
        "--input-file",
        type=vfile,
        required=True,
        help="Input frequencies CSV file",
    )
    # ---------------------------------------------------------------------------------------------------------------
    parser_corr.add_argument(
        "-i",
        "--input-file",
        type=vfile,
        required=True,
        help="Input frequencies CSV file",
    )
    parser_corr.add_argument(
        "-p",
        "--photodiode-file",
        type=vfile,
        required=True,
        help="CSV file with photdiode readings",
    )
    parser_corr.add_argument(
        "-m",
        "--model",
        default=OSI_PHOTODIODE,
        choices=(HAMAMATSU_PHOTODIODE, OSI_PHOTODIODE),
        help="Photodiode model. (default: %(default)s)",
    )
    parser_corr.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=5,
        choices=(1, 5),
        help="Wavelength resolution (nm). (default: %(default)s nm)",
    )
    parser_corr.add_argument(
        "--export",
        action="store_true",
        help="Export to CSV file",
    )
    parser_corr.add_argument(
        "-u",
        "--units",
        type=str,
        choices=("nm", "angs"),
        default="nm",
        help="Exported wavelength units. (default: %(default)s)",
    )
    parser_corr.add_argument(
        "-wl",
        "--wavelength-last",
        action="store_true",
        help="Wavelength is last column in exported file",
    )
    # ---------------------------------------------------------------------------------------------------------------
    
# ================
# MAIN ENTRY POINT
# ================


def main():
    execute(
        main_func=spectess,
        add_args_func=add_args,
        name=__name__,
        version=__version__,
        description="TESS-W sensor spectral response",
    )
