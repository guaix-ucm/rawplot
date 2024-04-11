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

import re
import csv
import math
import logging

from importlib.resources import files

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt

from lica.cli import execute

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.mpl.plot import mpl_main_plot_loop
from .util.common import common_list_info, make_plot_title_from, assert_physical

# ----------------
# Module constants
# ----------------

OSI_PHOTODIODE       = 'OSI-11-01-004-10D'
HAMAMATSU_PHOTODIODE = 'Ham-S2281-04'

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

def mpl_photodiode_plot_loop(title, wavelength, responsivity, qe, xtitle, ytitle):
    fig, axes = plt.subplots(nrows=1, ncols=1, layout='tight')
    fig.suptitle(title)
    axes.set_xlabel(xtitle)
    axes.set_ylabel(f"{ytitle}")
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.plot(wavelength, responsivity,  marker='o', linewidth=0, label='Responsivity [A/W]')
    axes.plot(wavelength, qe,  marker='o', linewidth=0, label='Quantum Efficiency')
    axes.minorticks_on()
    axes.legend()
    plt.show()

def photodiode_export(model, resolution, path):
    log.info("Exporting model %s, resolution %d nm to file %s", model, resolution, path)
    f = files('rawplot.resources').joinpath(model + '.csv')
    with f.open('r') as csvfile:
        lines = csvfile.readlines()
    with open(path,'w') as exportfile:
        exportfile.writelines(lines[0:1])
        exportfile.writelines(lines[1::resolution])

def photodiode_load(model, resolution):
    '''Return dictionaries whose keys are the wavelengths'''
    f = files('rawplot.resources').joinpath(model + '.csv')
    with f.open('r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        responsivity = dict()
        qe = dict()
        for row in reader:
            responsivity[int(row['Wavelength (nm)'])] = float(row['Responsivity (A/W)'])
            qe[int(row['Wavelength (nm)'])] = float(row['Quantum Efficiency'])
    # resample dictionaries if necessary for 5 nm resolution
    responsivity = { key: val for key, val in responsivity.items() if key % resolution == 0}
    qe = { key: val for key, val in qe.items() if key % resolution == 0}
    return responsivity, qe
    
# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def export(args):
    log.info(" === PHOTODIODE RESPONSIVITY & QE EXPORT === ")
    photodiode_export(args.model, args.resolution, args.csv_file)

def plot(args):
    log.info(" === PHOTODIODE RESPONSIVITY & QE PLOT === ")
    responsivity, qe = photodiode_load(args.model, args.resolution)
    wavelength = np.array([key for key,value in responsivity.items()])
    responsivity = np.array([value for key, value in responsivity.items()])
    qe = np.array([value for key, value in qe.items()])
    mpl_photodiode_plot_loop(
        title = f"{args.model} characteristics", 
        wavelength = wavelength, 
        responsivity = responsivity, 
        qe = qe, 
        xtitle = "Wavelengnth [nm]", 
        ytitle = "Responsivity [A/W] & Quantum Efficiency"
    )


COMMAND_TABLE = {
    'plot': plot,
    'export': export,
}

def photodiode(args):
    command =  args.command
    func = COMMAND_TABLE[command]
    func(args)

# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):

    subparser = parser.add_subparsers(dest='command')

    parser_plot = subparser.add_parser('plot', help='Plot Responsivity & Quantum Efficiency')
    parser_expo = subparser.add_parser('export', help='Export Responsivity & Quantum Efficiency to CSV file')

    parser_plot.add_argument('-m','--model', default=OSI_PHOTODIODE, choices=(HAMAMATSU_PHOTODIODE, OSI_PHOTODIODE),
                    help='Photodiode model. (default: %(default)s)')
    parser_plot.add_argument('-r','--resolution', type=int, default=5, choices=(1,5), 
                    help='Wavelength resolution (nm). (default: %(default)s nm)')
  
    parser_expo.add_argument('-m','--model', default=OSI_PHOTODIODE, choices=(HAMAMATSU_PHOTODIODE, OSI_PHOTODIODE),
                    help='Photodiode model. (default: %(default)s)')
    parser_expo.add_argument('-r','--resolution', type=int, default=5, choices=(1,5), 
                    help='Wavelength resolution (nm). (default: %(default)s nm)')
    parser_expo.add_argument('-f', '--csv-file', type=str, required=True, help='CSV file name to export')
    

# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=photodiode, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="LICA reference photodiodes characteristics"
    )
