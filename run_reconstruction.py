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
from compute.multiDimProcess import process_background, loopPos, compute_flat_field, read_metadata, parse_bg_options
from utils.ConfigReader import ConfigReader
from utils.imgIO import process_position_list
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

def run_action(args):
    config = ConfigReader()
    config.read_config(args.config)
    
    # read meta data
    img_obj_list, bg_obj_list = read_metadata(config)
    img_obj_list, config = process_position_list(img_obj_list, config)
    # process background options
    img_obj_list = parse_bg_options(img_obj_list, config)
    

    for img_obj, bg_obj in zip(img_obj_list, bg_obj_list):
        processImg(img_obj, bg_obj, config)

if __name__ == '__main__':
    args = parse_args()
    run_action(args)