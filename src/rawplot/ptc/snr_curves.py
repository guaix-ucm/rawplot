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

import logging

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
from lica.validators import vdir, vfloat, vfloat01, vflopath

# ------------------------
# Own modules and packages
# ------------------------

from ..util.mpl.plot import mpl_main_plot_loop
from ..util.common import common_list_info, make_plot_title_from, assert_physical
from .common import signal_and_noise_variances

# ----------------
# Module constants
# ----------------


# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)


def snr_parser_arguments(parser):
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
        "-wi", "--width", type=vfloat01, default=1.0, help="Normalized ROI width [0..1]"
    )
    parser.add_argument(
        "-he",
        "--height",
        type=vfloat01,
        default=1.0,
        help="Normalized ROI height [0..1]",
    )
    parser.add_argument(
        "-c",
        "--channels",
        default=["R", "Gr", "Gb", "B"],
        nargs="+",
        choices=["R", "Gr", "Gb", "G", "B"],
        help="color plane to plot. G is the average of G1 & G2. (default: %(default)s)",
    )
    parser.add_argument(
        "--every",
        type=int,
        metavar="<N>",
        default=1,
        help="pick every n `file after sorting",
    )
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
    parser.add_argument(
        "--p-fpn",
        type=vfloat01,
        metavar="<p>",
        help="Fixed Pattern Noise Percentage factor: [0..1] (default: %(default)s)",
    )
    parser.add_argument(
        "-rd",
        "--read-noise",
        type=vfloat,
        metavar="<\u03c3>",
        help="Read noise [DN] (default: %(default)s)",
    )
    parser.add_argument(
        "-gn",
        "--gain",
        type=vfloat,
        metavar="<g>",
        help="Gain [e-/DN] (default: %(default)s)",
    )
    parser.add_argument(
        "-ph",
        "--physical-units",
        action="store_true",
        help="Display in [e-] physical units instead of [DN]. Requires --gain",
    )
    parser.add_argument(
        "--log2",
        action="store_true",
        help="Display plot using log2 instead of log10 scale",
    )


def nsr_parser_arguments(parser):
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
        "-wi", "--width", type=vfloat01, default=1.0, help="Normalized ROI width [0..1]"
    )
    parser.add_argument(
        "-he",
        "--height",
        type=vfloat01,
        default=1.0,
        help="Normalized ROI height [0..1]",
    )
    parser.add_argument(
        "-c",
        "--channels",
        default=["R", "Gr", "Gb", "B"],
        nargs="+",
        choices=["R", "Gr", "Gb", "G", "B"],
        help="color plane to plot. G is the average of G1 & G2. (default: %(default)s)",
    )
    parser.add_argument(
        "--every",
        type=int,
        metavar="<N>",
        default=1,
        help="pick every n `file after sorting",
    )
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
    parser.add_argument(
        "--log2",
        action="store_true",
        help="Display plot using log2 instead of log10 scale",
    )
    parser.add_argument(
        "-th",
        "--threshold",
        type=vfloat,
        default=0.5,
        metavar="<TH>",
        help="Threshold [DN] (default: %(default)s)",
    )


def model_snr(signal, read_noise, p_fpn, gain=1.0):
    """All units in DN, can be used with e- if gain is 1.0"""
    return signal / (np.sqrt(read_noise**2 + (signal / gain) + (p_fpn * signal) ** 2))


def check_model(args):
    return (
        True
        if args.read_noise is not None and args.p_fpn is not None and args.gain is not None
        else False
    )


def plot_fitted_box(axes, fitted):
    """All graphical elements for a fitting line"""
    mean = fitted["mean"]
    std = fitted["std"]
    label = fitted["label"]
    fitted_y = fitted["y"]
    fitted_x = fitted["x"]
    axes.plot(fitted_x, fitted_y, marker="o", linewidth=1, label=f"{label} (selected)")
    text = "\n".join((f"{label}", rf"$\mu = {mean:0.2e}$", rf"$\sigma = {std:0.2e}$"))
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    axes.text(0.4, 0.95, text, transform=axes.transAxes, va="top", bbox=props)


def plot_snr_vs_signal(axes, i, x, y, xtitle, ytitle, ylabel, channels, **kwargs):
    """For Curves 1 to 8"""
    phys = kwargs.get("phys", False)  # rae we dealing with physical values?
    # Main data plot goes here
    axes.plot(x[i], y[i], marker="o", linewidth=0, label=ylabel)
    # Additional data plots go here
    model = kwargs.get("model", None)
    if model is not None:
        g = kwargs["gain"]
        rd = kwargs["read"]
        p = kwargs["p_fpn"]
        axes.plot(x[i], model[i], marker="o", linewidth=0, label="model")
        axes.axvline(
            g * rd**2,
            linestyle="--",
            linewidth=1,
            color="k",
            label=r"$\sigma_{READ}$ limit",
        )
        axes.axvline(
            1 / (g * p**2),
            linestyle="-.",
            linewidth=1,
            color="k",
            label=r"$\sigma_{SHOT}$ limit",
        )
        # plot_fitted_box(axes, fitted[i])
    # Titles, scales and grids
    axes.set_title(f"channel {channels[i]}")
    base = 2 if kwargs.get("log2", False) else 10
    axes.set_xscale("log", base=base)
    axes.set_yscale("log", base=base)
    axes.grid(True, which="major", color="silver", linestyle="solid")
    axes.grid(True, which="minor", color="silver", linestyle=(0, (1, 10)))
    axes.minorticks_on()
    units = r"$[e^{-}]$" if phys else "[DN]"
    axes.set_xlabel(f"{xtitle} {units}")
    axes.set_ylabel(f"{ytitle}")
    if ylabel:
        axes.legend()


# ------------------------
# AUXILIARY MAIN FUNCTIONS
# ------------------------


def snr_curve1(args):
    log.info(" === SNR CHART 1: Total SNR vs. Signal === ")
    assert_physical(args)
    file_list, roi, n_roi, channels, metadata = common_list_info(args)
    read_noise = args.read_noise or 0.0
    signal, total_var, shot_read_var, fpn_var, shot_var = signal_and_noise_variances(
        file_list=file_list,
        n_roi=n_roi,
        channels=channels,
        bias=args.bias,
        dark=args.dark,
        read_noise=read_noise,
    )
    total_noise = np.sqrt(total_var)
    shot_noise = np.sqrt(shot_var)
    fpn_noise = np.sqrt(fpn_var)
    if args.gain and args.physical_units:
        total_noise *= args.gain
        shot_noise *= args.gain
        fpn_noise *= args.gain
        read_noise *= args.gain
        signal *= args.gain
        model = model_snr(signal, read_noise, args.p_fpn) if check_model(args) else None
    else:
        model = (
            model_snr(signal, args.read_noise, args.p_fpn, args.gain) if check_model(args) else None
        )
    snr = signal / total_noise
    title = make_plot_title_from("SNR (Total Noise) vs. Signal", metadata, roi)
    mpl_main_plot_loop(
        title=title,
        plot_func=plot_snr_vs_signal,  # 2D (channel, data) Numpy array
        xtitle="Signal",
        ytitle="SNR",
        x=signal,
        y=snr,
        ylabel=r"$SNR_{TOTAL}$",
        channels=channels,
        # Optional arguments
        model=model,
        phys=args.physical_units,
        log2=args.log2,
        gain=args.gain,
        read=args.read_noise,
        p_fpn=args.p_fpn,
    )
