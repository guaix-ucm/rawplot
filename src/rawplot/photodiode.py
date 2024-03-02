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
from lica.misc import file_paths
from lica.validators import vdir, vfile, vfloat, vfloat01, vflopath
from lica.raw.loader import ImageLoaderFactory
from lica.raw.analyzer.image import ImageStatistics
from lica.csv import read_csv

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
HAMAMATSU_PHOTODIODE = 'Ham-S2281-01'

PHOTODIODE_QE_DATA    = files('rawplot.resources').joinpath('OSI-11-01-004_10D.csv')
WAVELENGTH_REG_EXP = re.compile(r'(\w+)_(\d+)nm_g(\d+)_(\d+)_(\d+)_(\w+).jpg')

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# ------------------
# Auxiliary fnctions
# ------------------

def photodiode_qe_data(step_size):
    '''Read the QE data embedded in a resource CSV file'''
    assert step_size == 1 or step_size == 5, "Step size must be 1 or 5"
    log.info("Reading OSI Photodiode Quantum Efficiency data for %d nm", step_size)
    with PHOTODIODE_QE_DATA.open('r') as csvfile:
        reader = csv.DictReader(csvfile)
        #qe_data = { int(row['Wavelength (nm)']): float(row['Quantum Efficiency']) for row in reader}
        qe_data = { int(row['Wavelength (nm)']): float(row['Interpolated Responsivity (A/W)']) for row in reader}
    if step_size == 5:
        # Down sampling to 5 nm
        qe_data = { key: val for key, val in qe_data.items() if key % 5 == 0}
    qe_data = np.array(tuple(qe_data[key] for key in sorted(qe_data.keys())))
    log.info(qe_data.shape)
    return qe_data

def mpl_photodiode_plot_loop(title, figsize, x, y, xtitle, ytitle,  **kwargs):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize, layout='tight')
    fig.suptitle(title)
    axes.set_xlabel(xtitle)
    axes.set_ylabel(f"{ytitle}")
    filters = kwargs.get('filters', None)
    if filters is not None:
        for filt in filters:
            axes.axvline(filt['wave'], linestyle=filt['style'], label=filt['label'])
    ylogscale = kwargs.get('ylogscale', False)
    if ylogscale:
        axes.set_yscale('log', base=10)
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.plot(x, y,  marker='o', linewidth=1)
    axes.minorticks_on()
    axes.legend()
    plt.show()


def plot_raw_spectral(axes, i, x, y, channels, **kwargs):
    wavelength = x[i]
    signal = y[i]
    if channels[i] == 'R':
        color = 'red'
        marker = 'o'
    elif  channels[i] == 'B':
        color = 'blue'
        marker = 'o'
    elif  channels[i] == 'Gr':
        color = (0, 0.5, 0)
        marker = '1'
    elif  channels[i] == 'Gb':
        color = (0, 0.25, 0)
        marker = '2'
    else:
        color = 'green'
    axes.plot(wavelength, signal,  marker=marker, color=color, linewidth=1, label=channels[i])
   


def csv_to_arrays(csv_path):
     response = read_csv(csv_path)
     wavelength = np.array([int(entry[WAVELENGTH_CSV_HEADER]) for entry in response])
     current = np.array([math.fabs(float(entry[CURRENT_CSV_HEADER])) for entry in response])
     read_noise = np.array([float(entry[READ_NOISE_CSV_HEADER]) for entry in response])
     log.info("Got photodiode %d readings", wavelength.shape[0])
     return wavelength, current, read_noise


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
        reader = csv.DictReader(csvfile)
        qe = { int(row['Wavelength (nm)']): float(row['Quantum Efficiency']) for row in reader}
        responsivity = { int(row['Wavelength (nm)']): float(row['Interpolated Responsivity (A/W)']) for row in reader}
    # resample dictionary if necessary
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
    parser_expo  = subparser.add_parser('export', help='Export Responsivity & Quantum Efficiency to CSV file')

    parser_plot.add_argument('-m','--model', default=OSI_PHOTODIODE,
                    choices=(HAMAMATSU_PHOTODIODE, OSI_PHOTODIODE),
                    help='Photodiode model. (default: %(default)s)')
  
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
        description="Draft plot of sensor spectral response"
    )
