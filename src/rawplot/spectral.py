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
import logging
from argparse import ArgumentParser
from enum import IntEnum
from typing import Union, Dict, Any, Sequence, Iterable

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt

import astropy.io.ascii
import astropy.units as u
from astropy.table import Table

from lica import StrEnum
from lica.cli import execute
from lica.validators import vdir, vfile, vfloat, vfloat01, vflopath
from lica.raw.loader.roi import Roi, NormRoi
from lica.raw.analyzer.image import ImageStatistics

import lica.photodiode
from lica.photodiode import PhotodiodeModel, COL, BENCH

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.common import common_list_info, make_plot_title_from, export_spectra_to_csv

# -----------------
# Additiona classes
# -----------------

BiasType = Union[float, str]
DarkType = Union[float, str]


class TBCOL(StrEnum):
    """Additiona columns names for data produced by Scan.exe or TestBench"""

    INDEX = "Index"  # Index number 1, 2, etc produced in the CSV file
    CURRENT = "Electrical Current"  #
    READ_NOISE = "Read Noise"


class PhDOption(IntEnum):
    """Photodiode plot options"""

    RAW = 1
    NORM = 2
    SNR = 3


# ----------------
# Module constants
# ----------------

WAVELENGTH_REG_EXP = re.compile(r"(\w+)_(\d+)nm_g(\d+)_(\d+)_(\d+)_(\w+).jpg")

MONOCROMATOR_FILTERS_LABELS = (
    {"label": r"$BG38 \Rightarrow OG570$", "wave": 570, "style": "--"},
    {"label": r"$OG570\Rightarrow RG830$", "wave": 860, "style": "-."},
)

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


def read_manual_csv(path: str) -> Table:
    """Load CSV files produced by manually copying LICA TestBench.exe into a CSV file"""
    table = astropy.io.ascii.read(
        path,
        delimiter=";",
        data_start=1,
        names=(COL.WAVE, TBCOL.CURRENT, TBCOL.READ_NOISE),
        converters={COL.WAVE: np.float64, TBCOL.CURRENT: np.float64, TBCOL.READ_NOISE: np.float64},
    )
    table[COL.WAVE] = np.round(table[COL.WAVE], decimals=0) * u.nm
    table[TBCOL.CURRENT] = np.abs(table[TBCOL.CURRENT]) * u.A
    table[TBCOL.READ_NOISE] = table[TBCOL.READ_NOISE] * u.A
    return table


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

def get_used_wavelengths(file_list: Iterable[str], channels: Sequence[str]):
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
    log.info("Wavelength array shape is %s", result.shape)
    return result

# ---------------------
# Exported, non-CLI API
# ---------------------

def raw_spectrum(
    file_list: Iterable[str],
    roi: Roi,
    n_roi: NormRoi,
    channels = Sequence[str],
    metadata = Dict[str, Any],
    every: int = 1,
    bias: BiasType = None,
    dark: DarkType = None,
):
    title = make_plot_title_from("Draft Spectral Response plot", metadata, roi)
    wavelength = get_used_wavelengths(file_list, channels)
    exptime, signal = signal_from(file_list, n_roi, channels, bias, dark, every)
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
        filters=MONOCROMATOR_FILTERS_LABELS,  # where filters were changesd
    )


def photodiode_spectrum(path: str, model: str, option: PhDOption, resolution: int) -> None:
    readings = read_manual_csv(path)
    reference = lica.photodiode.load(model=model, resolution=resolution)
    if option == PhDOption.RAW:
        title = "Raw Photodiode Signal vs Wavelength"
        y = readings[TBCOL.CURRENT]
        ytitle = "Current [A]"
        ylogscale = False
    elif option == PhDOption.NORM:
        title = "Raw Photodiode Signal vs Wavelength"
        y = readings[TBCOL.CURRENT] / np.max(readings[TBCOL.CURRENT])
        ytitle = "Current (normalized)"
        ylogscale = False
    else:
        title = "Photodiode SNR vs Wavelength"
        y = readings[TBCOL.CURRENT] / readings[TBCOL.READ_NOISE]
        ytitle = "SNR"
        ylogscale = True
    mpl_photodiode_plot_loop(
        title=title,
        xtitle="Wavelength [nm]",
        ytitle=ytitle,
        x=readings[COL.WAVE],
        y=y,
        # Optional arguments to be handled by the plotting function
        ylogscale=ylogscale,
        filters=MONOCROMATOR_FILTERS_LABELS,  # where filters were changesd
        qe=reference[COL.QE] if option == PhDOption.NORM else None,
        photodiode=model if option == PhDOption.NORM else None,
    )


# ===================================
# Command Line Interface Entry Points
# ===================================


def cli_raw_spectrum(args):
    log.info(" === DRAFT SPECTRAL RESPONSE PLOT === ")
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    raw_spectrum(file_list, roi, n_roi, channels, metadata, args.every, args.bias, args.dark)


def cli_corrected_spectrum(args):
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


def cli_photodiode_spectrum(args):
    log.info(" === PHOTODIODE SPECTRAL RESPONSET PLOT === ")
    if args.raw_readings:
        option = PhDOption.RAW
    elif args.normalized:
        option = PhDOption.NORM
    else:
        option = PhDOption.SNR
    photodiode_spectrum(args.photodiode_file, args.model, option, args.resolution)


def cli_spectral(args):
    args.func(args)


def prs_aux_files() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "-bi",
        "--bias",
        type=vflopath,
        help="Bias, either a single value for all channels or else a 3D FITS cube file (default: %(default)s)",
    )
    parser.add_argument(
        "-dk",
        "--dark",
        type=vfloat,
        help="Dark count rate in DN/sec. (default: %(default)s)",
    )
    return parser


def prs_images() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "-i",
        "--input-dir",
        type=vdir,
        required=True,
        help="Input directory with RAW files",
    )
    parser.add_argument(
        "-f",
        "--image-filter",
        type=str,
        required=True,
        help="Images filter, glob-style (i.e. flat*, dark*)",
    )
    parser.add_argument(
        "-x",
        "--x0",
        type=vfloat01,
        help="Normalized ROI start point, x0 coordinate [0..1]",
    )
    parser.add_argument(
        "-y",
        "--y0",
        type=vfloat01,
        help="Normalized ROI start point, y0 coordinate [0..1]",
    )
    parser.add_argument(
        "-wi",
        "--width",
        type=vfloat01,
        default=1.0,
        help="Normalized ROI width [0..1] (default: %(default)s)",
    )
    parser.add_argument(
        "-he",
        "--height",
        type=vfloat01,
        default=1.0,
        help="Normalized ROI height [0..1] (default: %(default)s) ",
    )
    parser.add_argument(
        "-c",
        "--channels",
        default=("R", "Gr", "Gb", "B"),
        nargs="+",
        choices=("R", "Gr", "Gb", "G", "B"),
        help="color plane to plot. G is the average of G1 & G2. (default: %(default)s)",
    )
    parser.add_argument(
        "--every",
        type=int,
        metavar="<N>",
        default=1,
        help="pick every n `file after sorting",
    )
    return parser


def prs_photod() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "-ph",
        "--photodiode-file",
        type=vfile,
        required=True,
        help="CSV file with photdiode readings",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=PhotodiodeModel.OSI,
        choices=[p for p in PhotodiodeModel],
        help="Photodiode model. (default: %(default)s)",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=5,
        choices=tuple(range(1, 11)),
        help="Wavelength resolution (nm). (default: %(default)s nm)",
    )
    return parser


def add_args(parser):
    subparser = parser.add_subparsers(dest="command")

    parser_raw = subparser.add_parser(
        "raw", parents=[prs_images(), prs_aux_files()], help="Raw spectrum"
    )
    parser_raw.set_defaults(func=cli_raw_spectrum)
    parser_corr = subparser.add_parser(
        "corrected",
        parents=[prs_images(), prs_aux_files(), prs_photod()],
        help="Corrected spectrum",
    )
    parser_corr.set_defaults(func=cli_corrected_spectrum)
    parser_diode = subparser.add_parser(
        "photodiode", parents=[prs_photod()], help="Photodiode readings"
    )
    parser_diode.set_defaults(func=cli_photodiode_spectrum)

    # ---------------------------------------------------------------------------------------------------------------

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
        "--snr",
        action="store_true",
        help="Plot Raw Signal to Noise Ratio",
    )


# ================
# MAIN ENTRY POINT
# ================


def main():
    execute(
        main_func=cli_spectral,
        add_args_func=add_args,
        name=__name__,
        version=__version__,
        description="Draft plot of sensor spectral response",
    )
