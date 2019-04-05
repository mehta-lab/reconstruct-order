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
from compute.multiDimProcess import findBackground, loopPos
from utils.ConfigReader import ConfigReader
from utils.imgIO import readMetaData
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

def processImg(img_io_list, img_io_bg_list, config):
    for img_io, img_io_bg in zip(img_io_list, img_io_bg_list):
        print('Processing ' + SmDir + ' ....')
        img_io.writeMetaData()
        img_io.chNamesIn = img_io.chNamesOut
        write_config(config, os.path.join(img_io.ImgOutPath, 'config.yml')) # save the config file in the processed folder
                
        img_io, img_reconstructor = findBackground(img_io, img_io_bg_, config) # find background tile
        
        img_io.loopZ ='sample'
        plot_config = config['plotting']
        img_io = loopPos(img_io, img_reconstructor, plot_config)

def run_action(args):
    config = ConfigReader()
    config.read_config(args.config)
            
    # Check that all paths to be analyzed exist
    if len(set(config.dataset.background)) <= 1:
        img_io_bg = readMetaData(RawDataPath, ProcessedPath, ImgDir, SmDir_, BgDir_, PosList_, config)
        img_io_bg = [img_io_bg] * len(config.dataset.samples)
    
    img_io =[]; img_io_bg = []
    for SmDir_, BgDir_, PosList_ in zip(SmDir, BgDir, PosList):
        img_io, img_io_bg = readMetaData(RawDataPath, ProcessedPath, ImgDir, SmDir_, BgDir_, PosList_, config)
        img_io.SmDir
        img_io.BgDir
        img_io.PosList
        PosList[i] = img_io.PosList
        checkThatAllDirsExist(SmDir, BgDir, PosList)
                # OutputPath = OutputPath + '_pol'
        img_io.ImgOutPath = img_io_bg.OutputPath
        os.makedirs(OutputPath, exist_ok=True)  # create folder for processed images

    processImg(img_io, img_io_bg, config)

class args:
    config = '/Users/ivan.ivanov/Documents/Benchmark/config_Benchmark.txt'

if __name__ == '__main__':
    args = args()
#    args = parse_args()
    run_action(args)