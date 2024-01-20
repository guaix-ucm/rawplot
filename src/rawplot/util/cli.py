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

import sys
import datetime
import os.path
import logging
import logging.handlers
import argparse
import traceback

# -------------
# Local imports
# -------------


# -----------------------
# Module global variables
# -----------------------

log = logging.getLogger()

# ------------------------
# Module utility functions
# ------------------------


def configure_logging(args):
    '''Configure the root logger'''
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.ERROR
    else:
        level = logging.INFO

    # get the root logger
   
    log.setLevel(level)

    # Log formatter
    # fmt = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] %(message)s')
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    # create console handler and set level to debug
    if args.console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch.setLevel(logging.DEBUG)
        log.addHandler(ch)
    # Create a file handler suitable for logrotate usage
    if args.log_file:
        # fh = logging.handlers.WatchedFileHandler(args.log_file)
        fh = logging.handlers.TimedRotatingFileHandler(args.log_file, when='midnight', interval=1, backupCount=365)
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)

    if args.modules:
        modules = args.modules.split(',')
        for module in modules:
            logging.getLogger(module).setLevel(logging.DEBUG)


def arg_parser(name, version, description):
    # create the top-level parser
    parser = argparse.ArgumentParser(prog=name, description=description)

    # ----------------------------------------
    # Generic args common to every command
    # ---------------------------------------

    parser.add_argument('--version', action='version', version='{0} {1}'.format(name, version))
    parser.add_argument('--console', action='store_true', help='Log to console.')
    parser.add_argument('--log-file', type=str, metavar="<FILE>", default=None, help='Log to file.')
    group0 = parser.add_mutually_exclusive_group()
    group0.add_argument('--verbose', action='store_true', help='Verbose output.')
    group0.add_argument('--quiet',   action='store_true', help='Quiet output.')
    parser.add_argument('--modules', type=str, default=None, action='store', 
        help='comma separated list of modules to activate debug level upon.')

    return parser



def execute(main_func, add_args_func, name, version, description):
    """
    Utility entry point
    """
    try:
        parser = arg_parser(name, version, description)
        add_args_func(parser) # Adds more arguments
        args = parser.parse_args(sys.argv[1:])
        configure_logging(args)
        log.info(f"============== {name} {version} ==============")
        main_func(args)
    except KeyboardInterrupt:
        log.critical("[%s] Interrupted by user ", name)
    except Exception as e:
        log.critical("[%s] Fatal error => %s", name, str(e))
        traceback.print_exc()
    finally:
        pass