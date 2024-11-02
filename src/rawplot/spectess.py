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

def export_spectra_to_csv(path, wavelength, signal, units, wave_last=False):
    wave_exported = wavelength * 10 if units == "angs" else wavelength
   
    if not wave_last:
        header = [
                f"Wavelength [{units}]",
            ] + "signal"
    else:
        header = "signal" + [f"Wavelength [{units}]"]
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(header)
        for row in range(wave_exported.shape[0]):
            data = [signal[lab][row] for lab in range(COLS)]
            if not wave_last:
                data = [wave_exported[row]] + data
            else:
                data = data + [wave_exported[row]]
            writer.writerow(data)
   

def mpl_raw_spectra_plot_loop(
    wavelength, frequency, freq_stddev, photodiode, read_noise, model, filters
):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.suptitle("Raw Spectral Response plot")
    for row in range(0, 2):
        ax = axes[row]
        ax.set_xlabel("Wavelength [nm]")
        if row == 0:
            ax.set_title("Photometer readings")
            ax.set_ylabel("Signal [Hz]")
            # ax.plot(wavelength, frequency, marker='+', color="blue", linewidth=1, label="TESS-W")
            ax.errorbar(
                wavelength,
                frequency,
                yerr=freq_stddev,
                uplims=True,
                lolims=True,
                marker="+",
                color="blue",
                linewidth=1,
                label="TESS-W",
            )
        else:
            ax.set_title("Photodiode readings")
            ax.set_ylabel("Signal [A]")
            ax.errorbar(
                wavelength,
                photodiode,
                yerr=read_noise,
                uplims=True,
                lolims=True,
                marker="+",
                color="green",
                linewidth=1,
                label=model,
            )

        for filt in filters:
            ax.axvline(filt["wave"], linestyle=filt["style"], label=filt["label"])
        ax.grid(True, which="major", color="silver", linestyle="solid")
        ax.grid(True, which="minor", color="silver", linestyle=(0, (1, 10)))
        ax.minorticks_on()
        ax.legend()
    plt.show()


def mpl_corrected_spectra_plot_loop(
    wavelength, signal, model, filters
):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.suptitle("Corrected Spectral Response plot") 
    axes.set_xlabel("Wavelength [nm]")
    axes.set_title("Normalized Spectral response")
    axes.set_ylabel("Signal")
    axes.plot(wavelength, signal, marker='+', color="blue", linewidth=1, label="TESS-W")
    for filt in filters:
        axes.axvline(filt["wave"], linestyle=filt["style"], label=filt["label"])
    axes.grid(True, which="major", color="silver", linestyle="solid")
    axes.grid(True, which="minor", color="silver", linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.legend()
    plt.show()


def tess_readings_to_arrays(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    response = read_csv(csv_path)
    wavelength = list()
    freq_mean = list()
    freq_std = list()
    for key, grp in groupby(response, key=lambda x: x["wavelength"]):
        freqs = [float(x["frequency"]) for x in grp]
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
    log.info(current)
    return wavelength, current, read_noise


# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def raw_spectrum(args):
    log.info(" === RAW SPECTRAL RESPONSE PLOT === ")
    wavelength, frequency, freq_std = tess_readings_to_arrays(args.input_file)
    _, current, read_noise = photodiode_readings_to_arrays(args.photodiode_file)
    mpl_raw_spectra_plot_loop(
        wavelength=wavelength,
        frequency=frequency,
        freq_stddev=freq_std,
        photodiode=current,
        read_noise=read_noise,
        model=args.model,
        # Optional arguments to be handled by the plotting function
        filters=[
            {"label": r"$BG38 \Rightarrow OG570$", "wave": 570, "style": "--"},
            {"label": r"$OG570\Rightarrow RG830$", "wave": 860, "style": "-."},
        ],  # where filters were changesd
    )


def corrected_spectrum(args):
    log.info(" === CORRECTED SPECTRAL RESPONSE PLOT === ")
    responsivity, qe = photodiode_load(args.model, args.resolution)
    log.info(
        "Read %s reference responsivity values at %d nm resolution from %s",
        len(responsivity),
        args.resolution,
        args.model,
    )
    wavelength, frequency, freq_std = tess_readings_to_arrays(args.input_file)
    _, current, read_noise = photodiode_readings_to_arrays(args.photodiode_file)
    qe = np.array(
        [qe[w] for w in wavelength]
    )  # Only use those wavelenghts actually used in the CSV sequence
    signal = qe * frequency / current
    signal = signal / np.max(signal)  # Normalize signal to its absolute maxímun for all channels
    if args.export:
        log.info("exporting to CSV file(s)")
        export_spectra_to_csv(
            labels=['Integral'],
            wavelength=wavelength,
            signal=signal,
            mode=args.export,
            units=args.units,
            wave_last=args.wavelength_last,
        )
    mpl_corrected_spectra_plot_loop(
        wavelength=wavelength,
        signal=signal,
        model=args.model,
        filters=[
            {"label": r"$BG38 \Rightarrow OG570$", "wave": 570, "style": "--"},
            {"label": r"$OG570\Rightarrow RG830$", "wave": 860, "style": "-."},
        ],  # where filters were changesd
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
    parser_raw.add_argument(
        "-p",
        "--photodiode-file",
        type=vfile,
        required=True,
        help="CSV file with photdiode readings",
    )
    parser_raw.add_argument(
        "-m",
        "--model",
        default=OSI_PHOTODIODE,
        choices=(HAMAMATSU_PHOTODIODE, OSI_PHOTODIODE),
        help="Photodiode model. (default: %(default)s)",
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
