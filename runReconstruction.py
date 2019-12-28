#!/usr/bin/env python

"""
runReconstruction:
Reconstruct birefringence, slow axis, transmission, and degree of polarization from polarization-resolved images.
This script provides a convenient method to workflow process multi-dimensional images acquired with Micro-Manager and OpenPolScope acquisition plugin.

Parameters
----------
    --config: path to configuration file.
Returns
-------
    None: the script writes data to disk.

"""

import argparse

from ReconstructOrder.workflow import reconstruct_batch


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                       help='path to yaml configuration file')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    reconstruct_batch(args.config)



