# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2021
#
# See the LICENSE file for details
# see the AUTHORS file for authors
# ----------------------------------------------------------------------

#--------------------
# System wide imports
# -------------------

import logging

# ---------------------
# Thrid-party libraries
# ---------------------

from lica.validators import valid_channels
from lica.raw.loader import ImageLoaderFactory, NormRoi
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
    file_list = sorted(file_paths(args.input_dir, args.image_filter))[::args.every]
    log.info("Processing %d files, selected every %d files", len(file_list), args.every)
    factory =  ImageLoaderFactory()
    image0 = factory.image_from(file_list[0], n_roi, channels)
    roi = image0.roi()
    metadata = image0.metadata()
    log.info("Common ROI %s and metadata taken from %s", metadata['roi'], metadata['name'])
    return file_list, roi, n_roi, channels, metadata

def common_info(args):
    channels = valid_channels(args.channels)
    log.info("Working with %d channels: %s", len(channels), channels)
    n_roi = NormRoi(args.x0, args.y0, args.width, args.height)
    log.info("Normalized ROI is %s", n_roi)
    factory =  ImageLoaderFactory()
    file_path = args.input_file
    image0 = factory.image_from(file_path, n_roi, channels)
    roi = image0.roi()
    metadata = image0.metadata()
    log.info("ROI %s and metadata taken from %s", metadata['roi'], metadata['name'])
    return file_path, roi, n_roi, channels, metadata

def make_plot_title_from(title, metadata, roi):
    title = f"{title}\n" \
            f"{metadata['maker']} {metadata['camera']}, ISO: {metadata['iso']}\n" \
            f"Color Plane Size: {metadata['width']} cols x {metadata['height']} rows\n" \
            f"ROI: {roi} {roi.width()} cols x {roi.height()} rows"
    return title