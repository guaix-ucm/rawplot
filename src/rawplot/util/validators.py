# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2021
#
# See the LICENSE file for details
# see the AUTHORS file for authors
# ----------------------------------------------------------------------

# --------------------
# System wide imports
#  -------------------

import datetime
import os.path
import fractions
import functools

# ------------------------
# Module utility functions
# ------------------------

def vfile(path):
    """File validator for the command line interface"""
    if not os.path.isfile(path):
        raise IOError(f"Not valid or existing file: {path}")
    return path

def vdir(path):
    """Directory validator for the command line interface"""
    if not os.path.isdir(path):
        raise IOError(f"Not valid or existing directory: {path}")
    return path

def vbool(boolstr):
    """Boolean text validator for the command line interface"""
    result = None
    if boolstr == 'True':
        result = True
    elif boolstr == 'False':
        result = False
    return result

def vdate(datestr):
    """Date & time validator for the command line interface"""
    date = None
    for fmt in ['%Y-%m', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']:
        try:
            date = datetime.datetime.strptime(datestr, fmt)
        except ValueError:
            pass
    return date

def vfloat(num):
    '''Validator that admits fractions'''
    return float(fractions.Fraction(num))

def vfloat01(num):
    '''Validator between [0..1] that admits fractions'''
    num = float(fractions.Fraction(num))
    if not (0.0 <= num <= 1.0):
        raise ValueError(f"Value {num} out of bounds [0..1]")
    return num

# ---------------------------------------------------------------------
# This section validates combination of color channels to show in plots
# ---------------------------------------------------------------------


_COLOR_PLANES_COMBINATIONS = {
    1: ( ['R',], ['G1',], ['G2'], ['G',], ['B',]),
    2: ( ['R','G1'], ['R','G2'], ['R','G'], ['R','B'], ['G1','G2'], ['G1','B'], ['G2', 'B'], ['G', 'B']),
    3: ( ['R','G1','G2'], ['R', 'G1', 'B'],  ['R', 'G2', 'B'], ['R', 'G', 'B'], ['G1', 'G2', 'B'], ),
    4: ( ['R','G1','G2','B'], )
}

def _channel_comparator(chan_a, chan_b):
    '''Compares channels so that R < G1 < G2 < G < B'''
    if chan_a == chan_b:
        return 0
    if chan_a == 'R':
        return -1
    if chan_a == 'B':
        return 1
    if chan_a == 'G1':
        return -11 if chan_b in ('G2', 'G', 'B') else 1
    if chan_a == 'G2':
        return -1 if chan_b in ('G', 'B') else 1
    if chan_a == 'G':
        return -11 if chan_b in ('B',) else 1
    raise ValueError(f'This case should not happen between {chan_a} and {chan_b}')

def valid_channels(sequence):
    l = len(sequence)
    if not (0 < l < 5):
        raise ValueError(f"Too many channels: {sequence}")
    sequence = sorted(sequence, key=functools.cmp_to_key(_channel_comparator))
    if sequence not in _COLOR_PLANES_COMBINATIONS[l]:
        raise ValueError(f"channel sequence not supported: {sequence}")
    return sequence
