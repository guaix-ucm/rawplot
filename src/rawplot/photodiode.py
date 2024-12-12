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

from argparse import Namespace, ArgumentParser
import logging
from typing import Tuple

# ---------------------
# Thrid-party libraries
# ---------------------

import matplotlib.pyplot as plt

from astropy.table import Table, Column
import astropy.units as u

import lica
from lica.cli import execute
from lica.photodiode import PhotodiodeModel, COL, BENCH


# ------------------------
# Own modules and packages
# ------------------------

from . import __version__

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

# ------------------
# Auxiliary fnctions
# ------------------

def vbench(x: str) -> float:
    x = float(x)
    if not (BENCH.WAVE_START <= x <= BENCH.WAVE_END):
        raise ValueError(f"{x} outside LICA Optical Test Bench range")
    return x

def get_labels(x: Column, y: Column) -> Tuple[str, str]:
    """Get the labels for a table column, using units if necessary"""
    xunit = x.unit
    yunit = y.unit
    xlabel = x.name + f" [{xunit}]" if xunit != u.dimensionless_unscaled else x.name
    ylabel = y.name + f" [{yunit}]" if yunit != u.dimensionless_unscaled else y.name
    return xlabel, ylabel


def plot_photodiode(table: Table, title: str, marker: str):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(title)
    xlabel, ylabel = get_labels(table[COL.WAVE], table[COL.RESP])
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel + " & " + COL.QE)
    axes.grid(True, which="major", color="silver", linestyle="solid")
    axes.grid(True, which="minor", color="silver", linestyle=(0, (1, 10)))
    axes.plot(table[COL.WAVE], table[COL.RESP], marker=marker, linewidth=0, label=COL.RESP)
    axes.plot(table[COL.WAVE], table[COL.QE], marker=marker, linewidth=0, label=COL.QE)
    axes.minorticks_on()
    axes.legend()
    plt.show()


# ---------------------------------
# Exported non-CLI (i.e to Jupyter)
# ---------------------------------


def plot(
    model: str,
    resolution: int,
    wave_start: int = BENCH.WAVE_START,
    wave_end: int = BENCH.WAVE_START,
    marker: str = ".",
) -> None:
    log.info(" === PHOTODIODE RESPONSIVITY & QE PLOT === ")
    table = lica.photodiode.load(model, resolution, wave_start, wave_end, cross_calibrated=True)
    log.info("Table info is\n%s", table.info)
    plot_photodiode(
        title=f"{model} characteristics @ {resolution} nm",
        table=table,
        marker=marker,
    )

def export(
    path: str,
    model: str,
    resolution: int,
    wave_start: int = BENCH.WAVE_START,
    wave_end: int = BENCH.WAVE_START,
) -> None:
    lica.photodiode.export(path, model, resolution, wave_start, wave_end)


# =============
# CLI INTERFACE
# =============


def cli_export(args: Namespace) -> None:
    log.info(" === PHOTODIODE RESPONSIVITY & QE EXPORT === ")
    export(args.ecsv_file, args.model, args.resolution, args.wave_start, args.wave_end)


def cli_plot(args: Namespace) -> None:
    log.info(" === PHOTODIODE RESPONSIVITY & QE PLOT === ")
    plot(args.model, args.resolution, args.wave_start, args.wave_end)


def cli_photodiode(args: Namespace) -> None:
    args.func(args)


# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================


def common_parser() -> ArgumentParser:
    """Common Options for subparsers"""
    parser = ArgumentParser(add_help=False)
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
    parser.add_argument(
        "-w1",
        "--wave-start",
        type=vbench,
        metavar="<W1 nm>",
        default=BENCH.WAVE_START,
        help="Start wavelength in nm (defaults to %(default)d)",
    )
    parser.add_argument(
        "-w2",
        "--wave-end",
        type=vbench,
        metavar="<W2 nm>",
        default=BENCH.WAVE_END,
        help="End wavelength in nm (defaults to %(default)d)",
    )

    return parser


def add_args(parser) -> None:
    subparser = parser.add_subparsers(dest="command")
    parser_plot = subparser.add_parser(
        "plot", parents=[common_parser()], help="Plot Responsivity & Quantum Efficiency"
    )
    parser_plot.set_defaults(func=cli_plot)
    parser_expo = subparser.add_parser(
        "export",
        parents=[common_parser()],
        help="Export Responsivity & Quantum Efficiency to CSV file",
    )
    parser_expo.set_defaults(func=cli_export)

    # ------------------------------------------------------------------------------------
    parser_plot.add_argument(
        "--marker",
        type=str,
        choices=[".", "o", "+", "*"],
        default=".",
        help="Plot Marker",
    )
    # ------------------------------------------------------------------------------------
    parser_expo.add_argument(
        "-f", "--ecsv-file", type=str, required=True, help="ECSV file name to export"
    )


# ================
# MAIN ENTRY POINT
# ================


def main():
    execute(
        main_func=cli_photodiode,
        add_args_func=add_args,
        name=__name__,
        version=__version__,
        description="LICA reference photodiodes characteristics",
    )


__all__ = ["plot", "export"]
