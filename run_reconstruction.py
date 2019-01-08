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
sys.path.append("..") # Adds parent directory to python search path.
# sys.path.append(os.path.dirname(sys.argv[0]))
from compute.multiDimProcess import findBackground, loopPos
from utils.imgIO import GetSubDirName
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

def read_config(config_fname):
    """Read the config file in yml format

    TODO: validate config!

    :param str config_fname: fname of config yaml with its full path
    :return:
    """

    with open(config_fname, 'r') as f:
        config = yaml.load(f)

    return config

def processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann, BgDir_local=None, flatField=False,
               bgCorrect=True, circularity=False, norm=True):
    print('Processing ' + SmDir + ' ....')
    imgSm, img_reconstructor = findBackground(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann,
                           BgDir_local=BgDir_local, flatField=flatField,bgCorrect=bgCorrect,
                           ff_method='open') # find background tile
    imgSm.loopZ ='sample'
    imgSm = loopPos(imgSm, img_reconstructor, flatField=flatField, bgCorrect=bgCorrect, circularity=circularity, norm=norm)

def run_action(args):
    config = read_config(args.config)
    RawDataPath = config['dataset']['RawDataPath']
    ProcessedPath = config['dataset']['ProcessedPath']
    ImgDir = config['dataset']['ImgDir']
    SmDir = config['dataset']['SmDir']
    BgDir = config['dataset']['BgDir']
    BgDir_local = config['dataset']['BgDir_local']
    outputChann = config['processing']['outputChann']
    circularity= config['processing']['circularity']
    bgCorrect=config['processing']['bgCorrect']
    flatField = config['processing']['flatField']
    batchProc = config['processing']['batchProc']
    norm = config['plotting']['norm']

    if batchProc:
        ImgPath = os.path.join(RawDataPath, ImgDir)
        SmDirList = GetSubDirName(ImgPath)
        for SmDir in SmDirList:
            # if 'SM' in SmDir or 'BG' in SmDir :
            # if 'SM' in SmDir and 'SMS' not in SmDir:
            if 'SM' in SmDir:
                processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann,
                           BgDir_local=BgDir_local, flatField=flatField, bgCorrect=bgCorrect,
                           circularity=circularity, norm=norm)
    else:
        processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann,
                   BgDir_local=BgDir_local, flatField=flatField, bgCorrect=bgCorrect,
                   circularity=circularity, norm=norm)


if __name__ == '__main__':
    args = parse_args()
    run_action(args)