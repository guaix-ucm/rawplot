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

import os
import math
import logging

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
from lica.raw.loader import ImageLoaderFactory,  NormRoi

# ------------------------
# Own modules and packages
# ------------------------


# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

def image_common(args):   
    channels = valid_channels(args.channels)
    log.info("Working with %d channels: %s", len(channels), channels)
    n_roi = NormRoi(args.x0, args.y0, args.width, args.height)
    log.info("Normalized ROI is %s", n_roi)
    if args.sim_dark is not None:
        image = SimulatedDarkImage(args.input_file, n_roi, channels, dk_current=args.sim_dark)
    else:
        factory =  ImageLoaderFactory()
        image = factory.image_from(args.input_file, n_roi=None, channels=channels)
    metadata = image.metadata()
    stack = image.load()
    image_section = factory.image_from(args.input_file, n_roi=n_roi, channels=channels)
    section = image_section.load()
    roi = image_section.roi()
    aver = np.mean(section,  axis=(1,2))
    mdn = np.median(section,  axis=(1,2))
    std = np.std(section, axis=(1,2))
    log.info("section %s average is %s", roi, aver)
    log.info("section %s stddev is %s", roi, std)
    log.info("section %s median is %s", roi, mdn)
    return  roi, channels, metadata, stack, aver, mdn, std
