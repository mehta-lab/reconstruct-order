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
import sys
sys.path.append(".") # Adds current directory to python search path.
# sys.path.append("..") # Adds parent directory to python search path.
# sys.path.append(os.path.dirname(sys.argv[0]))
from workflow.multiDimProcess import process_background, loopPos, compute_flat_field, read_metadata, parse_bg_options
from utils.ConfigReader import ConfigReader
from utils.imgIO import process_position_list, process_z_slice_list, process_timepoint_list
import os
import argparse



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

def processImg(img_obj, bg_obj, config):
    print('Processing ' + img_obj.name + ' ....')    
    
    # Write metadata in processed folder
    img_obj.chNamesIn = img_obj.chNamesOut
    img_obj.writeMetaData()
    
    # Write config file in processed folder
    config.write_config(os.path.join(img_obj.ImgOutPath, 'config.yml')) # save the config file in the processed folder
    
    img_obj, img_reconstructor = process_background(img_obj, bg_obj, config)
    
    flatField = config.processing.flatfield_correction
    if flatField:  # find background flourescence for flatField corection
        img_obj = compute_flat_field(img_obj, config)
        
    img_obj.loopZ ='sample'
    img_obj = loopPos(img_obj, config, img_reconstructor) 

def runReconstruction(configfile):
    config = ConfigReader()
    config.read_config(configfile)
    
    # read meta data
    img_obj_list, bg_obj_list = read_metadata(config)
    img_obj_list = process_position_list(img_obj_list, config)
    img_obj_list = process_z_slice_list(img_obj_list, config)
    img_obj_list = process_timepoint_list(img_obj_list, config)
    # process background options
    img_obj_list = parse_bg_options(img_obj_list, config)
    

    for img_obj, bg_obj in zip(img_obj_list, bg_obj_list):
        processImg(img_obj, bg_obj, config)

if __name__ == '__main__':
    args = parse_args()
    runReconstruction(args.config)