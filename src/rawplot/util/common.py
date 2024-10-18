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
import logging

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np

from lica.validators import valid_channels
from lica.raw.loader import ImageLoaderFactory, NormRoi, Roi
from lica.misc import file_paths

# ------------------------
# Own modules and packages
# ------------------------

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)


def assert_physical(args):
    if args.gain is None and args.physical_units:
        raise ValueError("Can'use physical units [-e] if --gain is not set")


def assert_range(args):
    if args.from_value is not None and args.to_value is None:
        raise ValueError("Missing --to value")
    if args.from_value is None and args.to_value is not None:
        raise ValueError("Missing --from value")
    if args.from_value is None and args.to_value is None:
        raise ValueError("Missing --from and --to values")
    if args.from_value > args.to_value:
        temp = args.from_value
        args.from_value = args.to_value
        args.to_value = temp


def common_list_info(args):
    channels = valid_channels(args.channels)
    log.info("Working with %d channels: %s", len(channels), channels)
    n_roi = NormRoi(args.x0, args.y0, args.width, args.height)
    log.info("Normalized ROI is %s", n_roi)
    file_list = sorted(file_paths(args.input_dir, args.image_filter))[:: args.every]
    log.info("Processing %d files, selected every %d files", len(file_list), args.every)
    factory = ImageLoaderFactory()
    image0 = factory.image_from(file_list[0], n_roi, channels)
    roi = image0.roi()
    metadata = image0.metadata()
    log.info("Common ROI %s and metadata taken from %s", metadata["roi"], metadata["name"])
    return file_list, roi, n_roi, channels, metadata


def common_info(args):
    channels = valid_channels(args.channels)
    log.info("Working with %d channels: %s", len(channels), channels)
    n_roi = NormRoi(args.x0, args.y0, args.width, args.height)
    log.info("Normalized ROI is %s", n_roi)
    factory = ImageLoaderFactory()
    file_path = args.input_file
    simulated = False
    log.info("Simulated Dark Frame image is %s", simulated)
    image0 = factory.image_from(file_path, n_roi, channels, simulated=simulated)
    roi = image0.roi()
    metadata = image0.metadata()
    log.info("ROI %s and metadata taken from %s", metadata["roi"], metadata["name"])
    return file_path, roi, n_roi, channels, metadata, False, image0


def common_info_with_sim(args):
    channels = valid_channels(args.channels)
    log.info("Working with %d channels: %s", len(channels), channels)
    n_roi = NormRoi(args.x0, args.y0, args.width, args.height)
    log.info("Normalized ROI is %s", n_roi)
    factory = ImageLoaderFactory()
    file_path = args.input_file
    simulated = args.sim_dark is not None or args.sim_read_noise is not None
    log.info("Simulated Dark Frame image is %s", simulated)
    image0 = factory.image_from(
        file_path,
        n_roi,
        channels,
        simulated=simulated,
        dark_current=args.sim_dark,
        read_noise=args.sim_read_noise,
    )
    roi = image0.roi()
    metadata = image0.metadata()
    log.info("ROI %s and metadata taken from %s", metadata["roi"], metadata["name"])
    return file_path, roi, n_roi, channels, metadata, simulated, image0


def make_plot_title_from(title, metadata, roi):
    title = (
        f"{title}\n"
        f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}\n"
        f"Color Plane Size: {metadata['width']} cols x {metadata['height']} rows\n"
        f"ROI: {roi} {roi.width()} cols x {roi.height()} rows"
    )
    return title


def make_plot_no_roi_title_from(title, metadata):
    title = (
        f"{title}\n"
        f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}, Date: {metadata['datetime']}\n"
        f"Color Plane Size: {metadata['width']} cols x {metadata['height']} rows\n"
    )
    return title


def make_image_title_from(title, metadata, roi):
    title = (
        f"{title}\n"
        f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}, Date: {metadata['datetime']}\n"
        f"Color Plane Size: {metadata['width']} cols x {metadata['height']} rows\n"
        f"ROI: {roi} {roi.width()} cols x {roi.height()} rows"
    )
    return title


def extended_roi(roi, width, height):
    log.info("ROI = %s. Centre is %s", roi, roi.centre())
    roi_ext_x = Roi.extend_X(roi, width)
    log.info("Extended X ROI = %s. Centre is %s", roi_ext_x, roi_ext_x.centre())
    roi_ext_y = Roi.extend_Y(roi, height)
    log.info("Extended Y ROI = %s. Centre is %s", roi_ext_y, roi_ext_y.centre())
    return roi_ext_x, roi_ext_y


def geom_center(pixels, channels):
    # We don't take into account the real Bayer grid offset yet ....
    Z, M, N = pixels.shape
    gcx = np.tile(np.array(N / 2), (Z, 1))
    gcy = np.tile(np.array(M / 2), (Z, 1))
    return gcx, gcy


def marginal_distributions(pixels, roi, black_levels=None):
    Z, M, N = pixels.shape
    roi_x, roi_y = extended_roi(roi, N, M)
    if black_levels:
        bias = np.array(black_levels).reshape(Z, -1)
    else:
        bias = np.zeros(Z).reshape(Z, -1)
    # produce the Horizontal aggregate from 0...N ncols
    pixels_x = pixels[:, roi_x.y0 : roi_x.y1, roi_x.x0 : roi_x.x1]
    H = np.mean(pixels_x, axis=1) - bias
    log.info("H shape = %s", H.shape)
    H = H / np.max(H, axis=1).reshape(Z, -1)
    # H abscissae in pixels
    X = np.tile(np.arange(0, N), (Z, 1))
    # produce the Vertical aggregate from 0...M rows
    pixels_y = pixels[:, roi_y.y0 : roi_y.y1, roi_y.x0 : roi_y.x1]
    V = np.mean(pixels_y, axis=2) - bias
    log.info("V shape = %s", V.shape)
    V = V / np.max(V, axis=1).reshape(Z, -1)
    # V abscissae in pixels
    Y = np.tile(np.arange(0, M), (Z, 1))
    return (X, H), (Y, V)


def centroid(marginal_X, marginal_Y):
    X, H = marginal_X
    Y, V = marginal_Y
    xc = np.sum(X * H, axis=1) / np.sum(H, axis=1)
    yc = np.sum(Y * V, axis=1) / np.sum(V, axis=1)
    log.info("Centroid Xc = %s", xc)
    log.info("Centroid Yc = %s", yc)
    return xc, yc


def export_spectra_to_csv(labels, wavelength, signal, mode, units, wave_last=False):
    wave_exported = wavelength * 10 if units == "angs" else wavelength
    COLS = len(labels)
    if mode == "combined":
        if not wave_last:
            header = [
                f"Wavelength [{units}]",
            ] + labels
        else:
            header = labels + [f"Wavelength [{units}]"]
        path = "combined_" + "_".join(labels) + ".csv"
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
    else:
        if not wave_last:
            header = (f"Wavelength [{units}]", "Relative value")
        else:
            header = ("Relative value", f"Wavelength [{units}]")
        for i, label in enumerate(labels):
            path = label + ".csv"
            selected_signal = signal[i]
            with open(path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                writer.writerow(header)
                for j in range(wave_exported.shape[0]):
                    if not wave_last:
                        data = (wave_exported[j], selected_signal[j])
                    else:
                        data = (selected_signal[j], wave_exported[j])
                    writer.writerow(data)
