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
from workflow.multiDimProcess import process_background, loopPos, compute_flat_field, create_metadata_object, parse_bg_options
from utils.ConfigReader import ConfigReader
import os
import argparse
import yaml



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

def write_config(config, config_fname):
    with open(config_fname, 'w') as f:
        yaml.dump(config, f, default_flow_style=True)
        #TODO: the output of output config is not in the correct format.

#TODO: the intent of img_io variable is not clear - does it contain only metadata from Micro-Manager and user?
def processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, PosList, BgDir, config):
    print('Processing ' + SmDir + ' ....')
    flatField = config.processing.flatfield_correction
    img_io, img_io_bg = create_metadata_object(config, RawDataPath, ImgDir, SmDir, BgDir)
    img_io, img_io_bg = parse_bg_options(img_io, img_io_bg, config, RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir)

    # process_background constructs an ImgReconstructor
    #   - calculates and adds the attribute 'stokes_param_bg' to the class
    #   - adds the attribute "img_raw_bg" to the img_io input, which is the raw background data
    img_io, img_reconstructor = process_background(img_io, img_io_bg, config)

    img_io.PosList = PosList
    if flatField:  # find background flourescence for flatField corection
        img_io = compute_flat_field(img_io, config)
    img_io.loopZ ='sample'
    img_io = loopPos(img_io, config, img_reconstructor)
    img_io.chNamesIn = img_io.chNamesOut
    img_io.writeMetaData()
    write_config(config, os.path.join(img_io.ImgOutPath, 'config.yml')) # save the config file in the processed folder

def runReconstruction(configfile):
    config = ConfigReader()
    config.read_config(configfile)

    # RawDataPath is a subfolder of ImgDir
    split_data_dir = os.path.split(config.dataset.data_dir)
    RawDataPath = split_data_dir[0]
    ImgDir = split_data_dir[1]

    ProcessedPath = config.dataset.processed_dir
    SmDirList = config.dataset.samples
    BgDirList = config.dataset.background
    PosList = config.dataset.positions

    for SmDir, BgDir, PosList_ in zip(SmDirList, BgDirList, PosList):
        processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, PosList_, BgDir, config)


if __name__ == '__main__':
    args = parse_args()
    runReconstruction(args.config)