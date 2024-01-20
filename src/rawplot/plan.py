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
import sys
import math
import logging

# ---------------------
# Thrid-party libraries
# ---------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ------------------------
# Own modules and packages
# ------------------------

from ._version import __version__
from .util.cli import execute
from .util.validators import vfloat
from .util.misc import write_csv

# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger(__name__)

# ------------------
# Auxiliary fnctions
# ------------------

def plot_histo(axes, color_plane, title, decimate, average, median, stddev):
    axes.set_title(fr'channel {title}: $median={median}, \mu={average},\;\sigma={stddev}$')
    data = color_plane.reshape(-1)[::decimate]
    bins=list(range(data.min(), data.max()+1))
    axes.hist(data, bins=bins, rwidth=0.9, align='left', label='hist')
    axes.set_xlabel('Pixel value [DN]')
    axes.set_ylabel('Pixel count')
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    axes.axvline(x=average, linestyle='--', color='r', label="mean")
    axes.axvline(x=median, linestyle='--', color='k', label="median")
    axes.legend()

def plot_image(fig, axes, color_plane, roi, title, average, median, stddev, colormap, edgecolor):
    axes.set_title(fr'{title}: $median={median}, \mu={average},\;\sigma={stddev}$')
    im = axes.imshow(color_plane, cmap=colormap)
    # Create a Rectangle patch
    rect = patches.Rectangle(roi.xy(), roi.width(), roi.height(), 
                    linewidth=1, linestyle='--', edgecolor=edgecolor, facecolor='none')
    axes.add_patch(rect)
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.10)
    fig.colorbar(im, cax=cax, orientation='vertical')

def export_to_csv(path, sequence):
    header = ("image #","exposure [secs]")
    seq = [ {'image #': f'{i:03d}', 'exposure [secs]': f'{t:.7f}'} for i, t in enumerate(sequence, start=1)]
    write_csv(path, header, seq)

def remove_duplicates(seq, epsilon=1.0e-7):
    new = [seq[i] for i in range(0, len(seq)-1) if math.fabs(seq[i+1]-seq[i]) > epsilon]
    new.append(seq[-1])
    return new

def stops_plan(ti, tf, max_dn, ppl, reverse):
    stops = int(round(math.log2(max_dn),0))
    T = list()
    log.info("STOPS = %d",stops)
    if reverse:
        nti = ti; ntf = ti + (tf-ti)/2
        for level in range(0, stops):
            tseq = np.linspace(nti, ntf, num=ppl)
            nti = ntf
            ntf = nti + (tf - nti)/2
            if level == 0:
                T.extend(tseq.tolist())
            else:
                # Discard the first point, as it was repeated with the previous iteration
                T.extend(tseq.tolist()[1:]) 
    else:
        ntf = tf; nti = tf - (tf-ti)/2
        for level in range(0, stops):
            tseq = np.linspace(nti, ntf, num=ppl)
            ntf = nti
            nti = ntf - (ntf-ti)/2
            if level == 0:
                T.extend(tseq.tolist())
            else:
                # Discard the last point, as it was repeated with the previous iteration
                T.extend(tseq.tolist()[:-1]) 
    log.info("Stops exposure time plan contains %d different exposures", len(T))
    return T

def linear_plan(ti, tf, n, endpoint):
    T = np.linspace(ti, tf, num=n, endpoint=endpoint).tolist()
    log.info("Linear exposure time plan contains %d different exposures", len(T))
    return T

def log_plan(ti, tf, n, reverse, endpoint):
    #T = ((tf-ti)*(1-np.logspace(math.log2(ti), math.log2(tf),  num=n, base=2))).tolist()
    #T = np.geomspace(ti, tf,  num=n).tolist()
    if not reverse:
        T = np.geomspace(ti, tf,  num=n, endpoint=endpoint).tolist()
    else:
        T =  (ti + tf-np.geomspace(ti, tf,  num=n, endpoint=endpoint)).tolist()
        T.reverse()
    log.info("Log exposure time plan contains %d different exposures", len(T))
    return T

def combilog_plan(ti, tf, n):
    #T = ((tf-ti)*(1-np.logspace(math.log2(ti), math.log2(tf),  num=n, base=2))).tolist()
    #T = np.geomspace(ti, tf,  num=n).tolist()
    n = n//2
    T = list()
    T.extend(log_plan(ti, tf, n, reverse=False, endpoint=False))
    T.extend(log_plan(ti, tf, n, reverse=True, endpoint=True))
    return remove_duplicates(sorted(T))

def combistops_plan(ti, tf, max_dn, ppl):
    #T = ((tf-ti)*(1-np.logspace(math.log2(ti), math.log2(tf),  num=n, base=2))).tolist()
    #T = np.geomspace(ti, tf,  num=n).tolist()
    T = list()
    T.extend(stops_plan(ti, tf, max_dn, ppl, reverse=False))
    T.extend(stops_plan(ti, tf, max_dn, ppl, reverse=True))
    return remove_duplicates(sorted(T))

# -----------------------
# AUXILIARY MAIN FUNCTION
# -----------------------

def plan(args):
    if args.command == 'linear':
        T = linear_plan(args.t_initial, args.t_final, args.num_images, endpoint=True)
    elif args.command == 'stops':
        T = stops_plan(args.t_initial, args.t_final, args.max_dn, args.points_per_level, args.reverse)
    elif args.command == 'log':
        T = log_plan(args.t_initial, args.t_final, args.num_images, args.reverse, endpoint=True)
    elif args.command == 'combilog':
        T = combilog_plan(args.t_initial, args.t_final, args.num_images)
    elif args.command == 'combistops':
        T = combistops_plan(args.t_initial, args.t_final, args.max_dn, args.points_per_level)
    else:
        raise ValueError(f"Missing command!. Type {os.path.basename(sys.argv[0])} -h for help")
    log.info("final plan contains %d exposures", len(T))
    log.info("Sequence of exposures: %s", T)
    if args.csv_file:
        export_to_csv(args.csv_file, T)
    num_seq = range(1, len(T)+1)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 9), layout='tight')
    fig.suptitle(f"{args.command} Exposure plan")   
    axes.plot(num_seq, T,  marker='o', linewidth=1, label="data")
    axes.set_xlabel('Image sequence number')
    axes.set_ylabel('Exposure time [s]')
    axes.grid(True,  which='major', color='silver', linestyle='solid')
    axes.grid(True,  which='minor', color='silver', linestyle=(0, (1, 10)))
    axes.minorticks_on()
    plt.show()

# ===================================
# MAIN ENTRY POINT SPECIFIC ARGUMENTS
# ===================================

def add_args(parser):

    subparser = parser.add_subparsers(dest='command')

    parser_linear = subparser.add_parser('linear', help='generate linear exposure plan')
    parser_log  = subparser.add_parser('log', help='generate log exposure plan')
    parser_stops  = subparser.add_parser('stops', help='generate log2 exposure plan based on saturation ADU stops')
    parser_combilog = subparser.add_parser('combilog', help='combineation of log + reverse log exposure plan')
    parser_combistops = subparser.add_parser('combistops', help='combineation of stops + reverse stops exposure plan')

    # ------------------------------
    # Arguments for 'linear' command
    # ------------------------------

    parser_linear.add_argument('-ti', '--t-initial', type=vfloat, required=True, help='Initial exposure time [secs]')
    parser_linear.add_argument('-tf', '--t-final', type=vfloat, required=True, help='Final exposure time [secs]')
    parser_linear.add_argument('-n', '--num-images', type=int, required=True, help='Number of images to take')
    parser_linear.add_argument('-f', '--csv-file', type=str, default=None, help='(Optional) CSV file to export')

    # ------------------------------
    # Arguments for 'log' command
    # ------------------------------

    parser_log.add_argument('-ti', '--t-initial', type=vfloat, required=True, help='Initial exposure time [secs]')
    parser_log.add_argument('-tf', '--t-final', type=vfloat, required=True, help='Final exposure time [secs]')
    parser_log.add_argument('-n', '--num-images', type=int, required=True, help='Number of images to take')
    parser_log.add_argument('-r','--reverse',  action='store_true', help='reverse point distribution')
    parser_log.add_argument('-f', '--csv-file', type=str, default=None, help='(Optional) CSV file to export')

    # ------------------------------
    # Arguments for 'stops' command
    # ------------------------------

    parser_stops.add_argument('-ti', '--t-initial', type=vfloat, required=True, help='Initial exposure time [secs]')
    parser_stops.add_argument('-tf', '--t-final', type=vfloat, required=True, help='Final exposure time [secs]')
    parser_stops.add_argument('-m', '--max-dn',  type=int, required=True, help='Saturation value in image [DN]')
    parser_stops.add_argument('-ppl', '--points-per-level',  type=int, default=3, help='Number of images per level')
    parser_stops.add_argument('-r','--reverse',  action='store_true', help='reverse point distribution')
    parser_stops.add_argument('-f', '--csv-file', type=str, default=None, help='(Optional) CSV file to export')

    # ------------------------------
    # Arguments for 'combilog' command
    # ------------------------------

    parser_combilog.add_argument('-ti', '--t-initial', type=vfloat, required=True, help='Initial exposure time [secs]')
    parser_combilog.add_argument('-tf', '--t-final', type=vfloat, required=True, help='Final exposure time [secs]')
    parser_combilog.add_argument('-n', '--num-images', type=int, metavar='<N>', required=True, help='Number of images to take')
    parser_combilog.add_argument('-f', '--csv-file', type=str, default=None, help='(Optional) CSV file to export')

    # ------------------------------
    # Arguments for 'combistops' command
    # ------------------------------

    parser_combistops.add_argument('-ti', '--t-initial', type=vfloat, required=True, help='Initial exposure time [secs]')
    parser_combistops.add_argument('-tf', '--t-final', type=vfloat, required=True, help='Final exposure time [secs]')
    parser_combistops.add_argument('-m', '--max-dn',  type=int, metavar='<MAX DN>', required=True, help='Saturation value in image [DN]')
    parser_combistops.add_argument('-ppl', '--points-per-level', type=int, default=3, help='Number of images per level')
    parser_combistops.add_argument('-f', '--csv-file', type=str, default=None, help='(Optional) CSV file to export')

# ================
# MAIN ENTRY POINT
# ================

def main():
    execute(main_func=plan, 
        add_args_func=add_args, 
        name=__name__, 
        version=__version__,
        description="Display RAW image or histogram in channels"
    )
