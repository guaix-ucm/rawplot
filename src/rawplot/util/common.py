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


def bias_from(args):
    if args.bias_file is not None:
        bias = args.bias_file
    elif args.bias_level is not None:
        bias = args.bias_level
    else:
        bias = None
    return bias