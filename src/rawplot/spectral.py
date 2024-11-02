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

import re
import math
import logging

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt

from lica.cli import execute
from lica.validators import vdir, vfile, vfloat, vfloat01, vflopath
from lica.raw.analyzer.image import ImageStatistics
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

WAVELENGTH_REG_EXP = re.compile(r"(\w+)_(\d+)nm_g(\d+)_(\d+)_(\d+)_(\w+).jpg")

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


def mpl_photodiode_plot_loop(title, x, y, xtitle, ytitle, **kwargs):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(title)
    axes.set_xlabel(xtitle)
    axes.set_ylabel(f"{ytitle}")
    filters = kwargs.get("filters", None)
    if filters is not None:
        for filt in filters:
            axes.axvline(filt["wave"], linestyle=filt["style"], label=filt["label"])
    ylogscale = kwargs.get("ylogscale", False)
    if ylogscale:
        axes.set_yscale("log", base=10)
    axes.grid(True, which="major", color="silver", linestyle="solid")
    axes.grid(True, which="minor", color="silver", linestyle=(0, (1, 10)))
    axes.plot(x, y, marker="o", linewidth=1, label="readings")
    qe = kwargs.get("qe", None)
    photodiode = kwargs.get("photodiode", None)
    if qe is not None:
        axes.plot(x, qe, marker="o", linewidth=0, label=f"{photodiode} QE")
    axes.minorticks_on()
    axes.legend()
    plt.show()


def mpl_spectra_plot_loop(title, x, y, xtitle, ytitle, plot_func, channels, ylabel, **kwargs):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(title)
    axes.set_xlabel(xtitle)
    axes.set_ylabel(ytitle)
    filters = kwargs.get("filters", None)
    for i in range(len(channels)):
        plot_func(axes, i, x, y, channels, **kwargs)
    if filters is not None:
        for filt in filters:
            axes.axvline(filt["wave"], linestyle=filt["style"], label=filt["label"])
    axes.grid(True, which="major", color="silver", linestyle="solid")
    axes.grid(True, which="minor", color="silver", linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.legend()
    plt.show()


def plot_raw_spectral(axes, i, x, y, channels, **kwargs):
    wavelength = x[i]
    signal = y[i]
    if channels[i] == "R":
        color = "red"
        marker = "o"
    elif channels[i] == "B":
        color = "blue"
        marker = "o"
    elif channels[i] == "Gr":
        color = (0, 0.5, 0)
        marker = "1"
    elif channels[i] == "Gb":
        color = (0, 0.25, 0)
        marker = "2"
    else:
        color = "green"
    axes.plot(wavelength, signal, marker=marker, color=color, linewidth=1, label=channels[i])


def signal_from(file_list, n_roi, channels, bias, dark, every=2):
    file_list = file_list[::every]
    N = len(file_list)
    signal_list = list()
    exptime_list = list()
    for i, path in enumerate(file_list, start=1):
        analyzer = ImageStatistics.from_path(path, n_roi, channels, bias, dark)
        analyzer.run()
        signal = analyzer.mean()
        signal_list.append(signal)
        exptime = np.full_like(signal, analyzer.loader().exptime())
        exptime_list.append(exptime)
        log.info("[%d/%d] \u03bc signal for image %s = %s", i, N, analyzer.name(), signal)
    return np.stack(exptime_list, axis=-1), np.stack(signal_list, axis=-1)


def get_used_wavelengths(file_list, channels):
    M = len(channels)
    data = list()
    for file in file_list:
        matchobj = WAVELENGTH_REG_EXP.search(file)
        if matchobj:
            item = {
                key: matchobj.group(i)
                for i, key in enumerate(
                    ("tag", "wave", "gain", "seq", "exptime", "filter"), start=1
                )
            }
            item["wave"] = int(item["wave"])
            item["gain"] = int(item["gain"])
            item["seq"] = int(item["seq"])
            item["exptime"] = int(item["exptime"])
            data.append(item)
    log.info("Matched %d files", len(data))
    result = np.array([item["wave"] for item in data])
    result = np.tile(result, M).reshape(M, len(data))
    log.info("Wavelengthss array shape is %s", result.shape)
    return result


def photodiode_readings_to_arrays(csv_path):
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
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    title = make_plot_title_from("Draft Spectral Response plot", metadata, roi)
    wavelength = get_used_wavelengths(file_list, channels)
    exptime, signal = signal_from(file_list, n_roi, channels, args.bias, args.dark, args.every)
    mpl_spectra_plot_loop(
        title=title,
        channels=channels,
        plot_func=plot_raw_spectral,
        xtitle="Wavelength [nm]",
        ytitle="Signal [DN]",
        ylabel="good",
        x=wavelength,
        y=signal,
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


def photodiode_spectrum(args):
    log.info(" === PHOTODIODE SPECTRAL RESPONSET PLOT === ")
    wavelength, current, read_noise = photodiode_readings_to_arrays(args.csv_file)
    responsivity, qe = photodiode_load(args.model, args.resolution)
    qe = np.array(
        [qe[w] for w in wavelength]
    )  # Only use those wavelenghts actually used in the CSV sequence
    if args.raw_readings:
        title = "Raw Photodiode Signal vs Wavelength"
        y = current
        ytitle = "Current [A]"
        ylogscale = False
    elif args.normalized:
        title = "Raw Photodiode Signal vs Wavelength"
        y = current / np.max(current)
        ytitle = "Current (normalized)"
        ylogscale = False
    else:
        title = "Photodiode SNR vs Wavelength"
        y = current / read_noise
        ytitle = "SNR"
        ylogscale = True
    mpl_photodiode_plot_loop(
        title=title,
        xtitle="Wavelength [nm]",
        ytitle=ytitle,
        x=wavelength,
        y=y,
        # Optional arguments to be handled by the plotting function
        ylogscale=ylogscale,
        filters=[
            {"label": r"$BG38 \Rightarrow OG570$", "wave": 570, "style": "--"},
            {"label": r"$OG570\Rightarrow RG830$", "wave": 860, "style": "-."},
        ],  # where filters were changesd
        qe=qe if args.normalized else None,
        photodiode=args.model if args.normalized else None,
    )


COMMAND_TABLE = {
    "raw": raw_spectrum,
    "corrected": corrected_spectrum,
    "photodiode": photodiode_spectrum,
}


def spectral(args):
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
    parser_diode = subparser.add_parser("photodiode", help="Photodiode readings")
    # ---------------------------------------------------------------------------------------------------------------
    parser_raw.add_argument(
        "-i",
        "--input-dir",
        type=vdir,
        required=True,
        help="Input directory with RAW files",
    )
    parser_raw.add_argument(
        "-f",
        "--image-filter",
        type=str,
        required=True,
        help="Images filter, glob-style (i.e. flat*, dark*)",
    )
    parser_raw.add_argument(
        "-x",
        "--x0",
        type=vfloat01,
        help="Normalized ROI start point, x0 coordinate [0..1]",
    )
    parser_raw.add_argument(
        "-y",
        "--y0",
        type=vfloat01,
        help="Normalized ROI start point, y0 coordinate [0..1]",
    )
    parser_raw.add_argument(
        "-wi",
        "--width",
        type=vfloat01,
        default=1.0,
        help="Normalized ROI width [0..1] (default: %(default)s)",
    )
    parser_raw.add_argument(
        "-he",
        "--height",
        type=vfloat01,
        default=1.0,
        help="Normalized ROI height [0..1] (default: %(default)s) ",
    )
    parser_raw.add_argument(
        "-c",
        "--channels",
        default=("R", "Gr", "Gb", "B"),
        nargs="+",
        choices=("R", "Gr", "Gb", "G", "B"),
        help="color plane to plot. G is the average of G1 & G2. (default: %(default)s)",
    )
    parser_raw.add_argument(
        "--every",
        type=int,
        metavar="<N>",
        default=1,
        help="pick every n `file after sorting",
    )
    parser_raw.add_argument(
        "-bi",
        "--bias",
        type=vflopath,
        help="Bias, either a single value for all channels or else a 3D FITS cube file (default: %(default)s)",
    )
    parser_raw.add_argument(
        "-dk",
        "--dark",
        type=vfloat,
        help="Dark count rate in DN/sec. (default: %(default)s)",
    )
    # ---------------------------------------------------------------------------------------------------------------
    parser_corr.add_argument(
        "-i",
        "--input-dir",
        type=vdir,
        required=True,
        help="Input directory with RAW files",
    )
    parser_corr.add_argument(
        "-f",
        "--image-filter",
        type=str,
        required=True,
        help="Images filter, glob-style (i.e. flat*, dark*)",
    )
    parser_corr.add_argument(
        "-x",
        "--x0",
        type=vfloat01,
        help="Normalized ROI start point, x0 coordinate [0..1]",
    )
    parser_corr.add_argument(
        "-y",
        "--y0",
        type=vfloat01,
        help="Normalized ROI start point, y0 coordinate [0..1]",
    )
    parser_corr.add_argument(
        "-wi",
        "--width",
        type=vfloat01,
        default=1.0,
        help="Normalized ROI width [0..1] (default: %(default)s)",
    )
    parser_corr.add_argument(
        "-he",
        "--height",
        type=vfloat01,
        default=1.0,
        help="Normalized ROI height [0..1] (default: %(default)s) ",
    )
    parser_corr.add_argument(
        "-c",
        "--channels",
        default=("R", "Gr", "Gb", "B"),
        nargs="+",
        choices=("R", "Gr", "Gb", "G", "B"),
        help="color plane to plot. G is the average of G1 & G2. (default: %(default)s)",
    )
    parser_corr.add_argument(
        "--every",
        type=int,
        metavar="<N>",
        default=1,
        help="pick every n `file after sorting",
    )
    parser_corr.add_argument(
        "-bi",
        "--bias",
        type=vflopath,
        help="Bias, either a single value for all channels or else a 3D FITS cube file (default: %(default)s)",
    )
    parser_corr.add_argument(
        "-dk",
        "--dark",
        type=vfloat,
        help="Dark count rate in DN/sec. (default: %(default)s)",
    )
    parser_corr.add_argument(
        "-cv",
        "--csv-file",
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
        type=str,
        choices=("combined", "individual"),
        help="Export to CSV file(s)",
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
    parser_diode.add_argument(
        "-cv",
        "--csv-file",
        type=vfile,
        required=True,
        help="CSV file with photdiode readings",
    )
    dioex1 = parser_diode.add_mutually_exclusive_group(required=True)
    dioex1.add_argument(
        "-w",
        "--raw-readings",
        action="store_true",
        help="Plot Photodiode raw readings in A",
    )
    dioex1.add_argument(
        "-n",
        "--normalized",
        action="store_true",
        help="Plot Photodiode normalized readings & QE",
    )
    dioex1.add_argument(
        "-s",
        "--signal-to-noise",
        action="store_true",
        help="Plot Raw Signal to Noise Ratio",
    )
    parser_diode.add_argument(
        "-m",
        "--model",
        default=OSI_PHOTODIODE,
        choices=(HAMAMATSU_PHOTODIODE, OSI_PHOTODIODE),
        help="Photodiode model. (default: %(default)s)",
    )
    parser_diode.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=5,
        choices=(1, 5),
        help="Wavelength resolution (nm). (default: %(default)s nm)",
    )


# ================
# MAIN ENTRY POINT
# ================


def main():
    execute(
        main_func=spectral,
        add_args_func=add_args,
        name=__name__,
        version=__version__,
        description="Draft plot of sensor spectral response",
    )
