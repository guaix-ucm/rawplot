# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2021
#
# See the LICENSE file for details
# see the AUTHORS file for authors
# ----------------------------------------------------------------------

# -------------------
# System wide imports
# -------------------

import os
import csv
import glob
import logging

# ---------
# Constants
# ---------

log = logging.getLogger(__name__)

# ------------------
# Auxiliar functions
# ------------------

def file_paths(input_dir, files_filter):
    '''Given a directory and a file filter, returns full path list'''
    file_list =  [os.path.join(input_dir, fname) for fname in glob.iglob(files_filter, root_dir=input_dir)]
    if not file_list:
        raise OSError("File list is empty, review the directory path or filter")
    return file_list

def write_csv(path, header, sequence, delimiter=';'):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, delimiter=delimiter)
        writer.writeheader()
        for row in sequence:
            writer.writerow(row)
    log.info("generated CSV file: %s", path)

