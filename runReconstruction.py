"""
Reconstruct birefringence, slow axis, transmission, and degree of polarization from polarization-resolved images.
This script provides a convenient method to workflow process multi-dimensional images acquired with Micro-Manager and OpenPolScope acquisition plugin.

Parameters
----------
    --config: path to configuration file.
Returns
-------
    None: the script writes data to disk.

"""

import sys
sys.path.append(".") # Adds current directory to python search path.
from workflow.multiDimProcess import process_background, loopPos, read_metadata, parse_bg_options, compute_flat_field
from utils.ConfigReader import ConfigReader
from utils.imgIO import process_position_list, process_z_slice_list, process_timepoint_list
import os
import argparse


def runReconstruction(configfile):
    """
    Initializes the code execution. Reads the config file and the metadata and parses the user input. The set of
    parameters is passed to ``processImg`` which executes the data reconstruction.

    Parameters
    ----------
    configfile: str
        Path to yaml config file

    """

    config = ConfigReader()
    config.read_config(configfile)

    # read meta data
    img_obj_list, bg_obj_list = read_metadata(config)

    # process user-provided position, z_slice, and timepoint subsets to be analyzed
    img_obj_list = process_position_list(img_obj_list, config)
    img_obj_list = process_z_slice_list(img_obj_list, config)
    img_obj_list = process_timepoint_list(img_obj_list, config)

    # process background options
    img_obj_list = parse_bg_options(img_obj_list, config)

    # run through set of samples in img_obj_list
    for img_obj, bg_obj in zip(img_obj_list, bg_obj_list):
        processImg(img_obj, bg_obj, config)


def processImg(img_obj, bg_obj, config):
    """
    Executes image reconstruction for a given sample.

    Parameters
    ----------
    img_obj
    bg_obj
    config

    """

    print('Processing ' + img_obj.name + ' ....')    
    
    # Write metadata in processed folder
    img_obj.chNamesIn = img_obj.chNamesOut
    img_obj.writeMetaData()
    
    # Write config file in processed folder
    config.write_config(os.path.join(img_obj.ImgOutPath, 'config.yml'))
    
    img_obj, img_reconstructor = process_background(img_obj, bg_obj, config)
    
    flatField = config.processing.flatfield_correction
    if flatField:  # find background fluorescence for flatField correction
        img_obj = compute_flat_field(img_obj, config)
        
    img_obj.loopZ = 'reconstruct'
    img_obj = loopPos(img_obj, config, img_reconstructor) 


def parse_args():
    """
    Parse command line arguments

    In python namespaces are implemented as dictionaries

    Returns
    -------
        Namespace containing the arguments passed.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='path to yaml configuration file')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    runReconstruction(args.config)