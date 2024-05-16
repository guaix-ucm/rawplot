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

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt

from lica.cli import execute
from lica.validators import vdir, vfile, vfloat, vfloat01, vflopath
from lica.raw.analyzer.image import ImageStatistics
from lica.csv import read_csv

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.mpl.plot import mpl_main_plot_loop
from .util.common import common_list_info, make_plot_title_from, assert_physical
from .photodiode import photodiode_load, OSI_PHOTODIODE, HAMAMATSU_PHOTODIODE

# ----------------
# Module constants
# ----------------

WAVELENGTH_REG_EXP = re.compile(r'(\w+)_(\d+)nm_g(\d+)_(\d+)_(\d+)_(\w+).jpg')

# Photodiode readings header columns
WAVELENGTH_CSV_HEADER = 'wavelength (nm)'
CURRENT_CSV_HEADER = 'current (A)'
READ_NOISE_CSV_HEADER = 'read noise (A)'

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

def mpl_filters_plot_loop(title, x, y, xtitle, ytitle, plot_func, ylabels, **kwargs):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(title)
    axes.set_xlabel(xtitle)
    axes.set_ylabel(ytitle)
    filters = kwargs.get('filters')
    diode = kwargs.get('diode')
    model = kwargs.get('model')
    Z, _ = y.shape 
    for i in range(Z):
        plot_func(axes, i, x, y, ylabels, **kwargs)
    if filters is not None:
        for filt in filters:
            axes.axvline(filt['wave'], linestyle=filt['style'], label=filt['label'])
    if diode is not None:
        axes.plot(x, diode,  marker='o', linewidth=0, label=model)
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.legend()
    plt.show()


def plot_raw_spectral(axes, i, x, y, ylabels, **kwargs):
    wavelength = x
    signal = y[i]
    marker = 'o'
    if i == 0:
        color = 'red'
    elif  i == 1:  # Green
        color = 'green'  
    elif  i == 2:
        color = 'blue' 
    else:
        color = 'magenta' # Other filter
    axes.plot(wavelength, signal,  marker=marker, color=color, linewidth=1, label=ylabels[i])
   


def get_used_wavelengths(file_list, channels):
    M = len(channels)
    data = list()
    for file in file_list:
        matchobj = WAVELENGTH_REG_EXP.search(file)
        if matchobj:
            item = { key:  matchobj.group(i) for i, key in enumerate(('tag', 'wave', 'gain', 'seq', 'exptime', 'filter'), start=1)}
            item['wave'] = int(item['wave'])
            item['gain'] = int(item['gain'])
            item['seq'] = int(item['seq'])
            item['exptime'] = int(item['exptime'])
            data.append(item)
    log.info("Matched %d files", len(data))
    result = np.array([item['wave'] for item in data])
    result = np.tile(result, M).reshape(M,len(data))
    log.info("Wavelengthss array shape is %s", result.shape)
    return result


def csv_readings_to_array(csv_path):
    log.info("reading CSV file %s", csv_path)
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        contents = {int(round(float(row[1]),0)): float(row[2]) for row in reader}
    wavelength = np.array(sorted(contents.keys()))
    signal = np.array(tuple(contents[key] for key in sorted(contents.keys())))
    return wavelength, signal

def get_info_from(args):
    accum = list()
    labels = list()
    if args.filter1:
        wavelength, signal = csv_readings_to_array(args.filter1)
        accum.append(signal)
        labels.append(args.label1)
    if args.filter2:
        _, signal = csv_readings_to_array(args.filter2)
        accum.append(signal)
        labels.append(args.label2)
    if args.filter3:
        _, signal = csv_readings_to_array(args.filter3)
        accum.append(signal)
        labels.append(args.label3)
    if args.filter4:
        _, signal = csv_readings_to_array(args.filter4)
        accum.append(signal)
        labels.append(args.label4)
    signal = np.vstack(accum)
    if args.diode:
        _, diode = csv_readings_to_array(args.diode)
        model = args.model
    else:
        diode = None
        model = None
    return wavelength, signal, labels, diode, model

# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def raw_spectrum(args):
    log.info(" === DRAFT SPECTRAL RESPONSE PLOT === ")
    wavelength, signal, labels, diode, model = get_info_from(args)
    mpl_filters_plot_loop(
        title    = f"Raw response for {args.title}",
        plot_func = plot_raw_spectral,
        xtitle = "Wavelength [nm]",
        ytitle = f"Signal [A]",
        ylabels = labels,
        x  = wavelength,
        y  = signal,
        # Optional arguments to be handled by the plotting function
        diode = diode,
        model = model,
        filters=[ 
            {'label':r'$BG38 \Rightarrow OG570$',  'wave': 570, 'style': '--'}, 
            {'label':r'$OG570\Rightarrow RG830$', 'wave': 860, 'style': '-.'},
        ] # where filters were changesd
    )

    
def corrected_spectrum(args):
    log.info(" === COMPLETE SPECTRAL RESPONSE PLOT === ")
    wavelength, signal, labels, diode, model = get_info_from(args)
    responsivity, qe = photodiode_load(args.model, args.resolution)
    log.info("Read %s reference responsivity values at %d nm resolution from %s", len(responsivity), args.resolution, args.model)
    qe = np.array([qe[w] for w in wavelength]) # Only use those wavelenghts actually used in the CSV sequence
    diode = diode / np.max(diode) # Normalize photodiode current
    signal = qe * signal / diode
    signal = signal / np.max(signal) # Normalize signal to its absolute maxímun for all channels
    mpl_filters_plot_loop(
        title    = f"Corrected response for {args.title}",
        plot_func = plot_raw_spectral,
        xtitle = "Wavelength [nm]",
        ytitle = f"Normalized signal level",
        ylabels = labels,
        x  = wavelength,
        y  = signal,
        filters=[ 
            {'label':r'$BG38 \Rightarrow OG570$',  'wave': 570, 'style': '--'}, 
            {'label':r'$OG570\Rightarrow RG830$', 'wave': 860, 'style': '-.'},
        ] # where filters were changesd
    )


COMMAND_TABLE = {
    'raw': raw_spectrum,
    'corrected': corrected_spectrum, 
}

def filters(args):
    command =  args.command
    func = COMMAND_TABLE[command]
    func(args)

# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):

    subparser = parser.add_subparsers(dest='command')

    parser_raw = subparser.add_parser('raw', help='Raw spectrum')
    parser_corr  = subparser.add_parser('corrected', help='Correced spectrum')
    # ---------------------------------------------------------------------------------------------------------------
    parser_raw.add_argument('-t', '--title', type=str, help='Filters set model (ie. "Astronomik L-RGB Type 2c"')
    parser_raw.add_argument('-f1', '--filter1', required=True, type=vfile, help='Filter 1 readings CSV file')
    parser_raw.add_argument('-l1', '--label1', required=True, type=str, help='Filter 1 plot label')
    parser_raw.add_argument('-f2', '--filter2', type=vfile, help='Filter 2 readings CSV file')
    parser_raw.add_argument('-l2', '--label2', type=str, help='Filter 2 plot label')
    parser_raw.add_argument('-f3', '--filter3', type=vfile, help='Filter 3 readings CSV file')
    parser_raw.add_argument('-l3', '--label3', type=str, help='Filter 3 plot label')
    parser_raw.add_argument('-f4', '--filter4', type=vfile, help='Filter 4 readings CSV file')
    parser_raw.add_argument('-l4', '--label4', type=str, help='Filter 4 plot label')
    parser_raw.add_argument('-d', '--diode', type=vfile, help='reference photodiode readings CSV file')
    parser_raw.add_argument('-m','--model', default=OSI_PHOTODIODE, choices=(HAMAMATSU_PHOTODIODE, OSI_PHOTODIODE),
                    help='Reference photodiode model. (default: %(default)s)')
    # ---------------------------------------------------------------------------------------------------------------
    parser_corr.add_argument('-t', '--title', type=str, help='Filters set model (ie. "Astronomik L-RGB Type 2c"')
    parser_corr.add_argument('-f1', '--filter1', required=True, type=vfile, help='Filter 1 readings CSV file')
    parser_corr.add_argument('-l1', '--label1', required=True, type=str, help='Filter 1 plot label')
    parser_corr.add_argument('-f2', '--filter2', type=vfile, help='Filter 2 readings CSV file')
    parser_corr.add_argument('-l2', '--label2', type=str, help='Filter 2 plot label')
    parser_corr.add_argument('-f3', '--filter3', type=vfile, help='Filter 3 readings CSV file')
    parser_corr.add_argument('-l3', '--label3', type=str, help='Filter 3 plot label')
    parser_corr.add_argument('-f4', '--filter4', type=vfile, help='Filter 4 readings CSV file')
    parser_corr.add_argument('-l4', '--label4', type=str, help='Filter 4 plot label')
    parser_corr.add_argument('-d', '--diode', type=vfile, required=True, help='reference photodiode readings CSV file')
    parser_corr.add_argument('-m','--model', default=OSI_PHOTODIODE, choices=(HAMAMATSU_PHOTODIODE, OSI_PHOTODIODE),
                    help='Reference photodiode model. (default: %(default)s)')
    parser_corr.add_argument('-r','--resolution', type=int, default=5, choices=(1,5), 
                    help='Wavelength resolution (nm). (default: %(default)s nm)')
    # ---------------------------------------------------------------------------------------------------------------

# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=filters, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="Filters spectral response"
    )
