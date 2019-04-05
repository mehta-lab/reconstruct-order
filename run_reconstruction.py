"""
Reconstruct retardance and orientation maps from images taken with different polarized illumination output
by Open compute. This script using the 4- or 5-frame reconstruction algorithm described in Michael Shribak and
Rudolf Oldenbourg, 2003.
 
* outputChann: (list) output channel names
    Current available output channel names:
        'Transmission'
        'Retardance'
        'Orientation' 
        'Retardance+Orientation'
        'Transmission+Retardance+Orientation'
        '405'
        '488'
        '568'
        '640'
        
* circularity: (bool) flip the sign of polarization. Set "True" for Dragonfly and "False" for ASI
* bgCorrect: (str) 
    'Auto' (default) to correct the background using background from the metadata if available, otherwise use input background folder;
    'None' for no background correction; 
    'Input' to always use input background folder   
* flatField: (bool) perform flat-field correction if True
* norm: (bool) scale images individually for optimal dynamic range. Set False for tiled images
    
"""

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import sys
sys.path.append(".") # Adds current directory to python search path.
# sys.path.append("..") # Adds parent directory to python search path.
# sys.path.append(os.path.dirname(sys.argv[0]))
from compute.multiDimProcess import process_background, loopPos, compute_flat_field, creat_metadata_object, parse_bg_options
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
        yaml.dump(config, f, default_flow_style=False)

def processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, PosList, BgDir, config):
    print('Processing ' + SmDir + ' ....')
    flatField = config.processing.flatfield_correction
    img_io, img_io_bg = creat_metadata_object(config, RawDataPath, ImgDir, SmDir, BgDir)
    img_io, img_io_bg = parse_bg_options(img_io, img_io_bg, config, RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir)
    img_io, img_reconstructor = process_background(img_io, img_io_bg, config)
    img_io.PosList = PosList
    if flatField:  # find background flourescence for flatField corection
        img_io = compute_flat_field(img_io, config)
    img_io.loopZ ='sample'
    img_io = loopPos(img_io, config, img_reconstructor)
    img_io.chNamesIn = img_io.chNamesOut
    img_io.writeMetaData()
    write_config(config, os.path.join(img_io.ImgOutPath, 'config.yml')) # save the config file in the processed folder

def run_action(args):
    config = ConfigReader()
    config.read_config(args.config)
    
    split_data_dir = os.path.split(config.dataset.data_dir)
    RawDataPath = split_data_dir[0]
    ProcessedPath = config.dataset.processed_dir
    ImgDir = split_data_dir[1]
    SmDirList = config.dataset.samples
    BgDirList = config.dataset.background
    PosList = config.dataset.positions
    
    for SmDir, BgDir, PosList_ in zip(SmDirList, BgDirList, PosList):
        processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, PosList_, BgDir, config)

if __name__ == '__main__':
    args = parse_args()
    run_action(args)