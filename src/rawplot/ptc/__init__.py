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

from lica.cli import execute

# ------------------------
# Own modules and packages
# ------------------------

from .._version import __version__
from .table import ptc_curves
from .noise_curves import noise_parser_arguments, noise_curve1, noise_curve2, noise_curve3, noise_curve4
from .variance_curves import variance_parser_arguments, variance_curve1
from .snr_curves import snr_parser_arguments, snr_curve1

# ----------------
# Module constants
# ----------------


# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# ------------------
# Auxiliary fnctions
# ------------------


CURVES_TABLE = {
    'curves': ptc_curves,
    'curve1': noise_curve1,
    'curve2': noise_curve2,
    'curve3': noise_curve3,
    'curve4': noise_curve4,
    'curve5': variance_curve1,
    'curve6': snr_curve1,
}


# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def ptc(args):
    command =  args.command
    function = CURVES_TABLE[command]
    function(args)


def add_args(parser):
    subparser = parser.add_subparsers(dest='command')
    parser_charts = subparser.add_parser('curves', help='Plot available PTC curves in matplotlib')

    parser_curve1 = subparser.add_parser('curve1', help='Plot read, shot, FPN & total noise vs. signal, [DN] or [e-]')
    noise_parser_arguments(parser_curve1)
   
    parser_curve2 = subparser.add_parser('curve2', help='read+shot noise vs. signal, [DN] or [e-]')
    noise_parser_arguments(parser_curve2)
    
    parser_curve3 = subparser.add_parser('curve3', help='shot noise vs. signal, [DN] or [e-]')
    noise_parser_arguments(parser_curve3)
   
    parser_curve4 = subparser.add_parser('curve4', help='FPN vs. signal, [DN] or [e-]')
    noise_parser_arguments(parser_curve4)
    
    parser_curve5 = subparser.add_parser('curve5', help='read+shot variance vs signal, [DN]')
    variance_parser_arguments(parser_curve5)

    parser_curve6 = subparser.add_parser('curve6', help='SNR vs signal, [DN] or [e-]')
    snr_parser_arguments(parser_curve6)

# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=ptc, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="Plot PTC curves from a set of RAW images"
    )
