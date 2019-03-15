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

def write_config(config, config_fname):
    with open(config_fname, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann, config, BgDir_local=None, flatField=False,
               bgCorrect=True, circularity=False, norm=True, azimuth_offset=0):
    print('Processing ' + SmDir + ' ....')
    img_io, img_reconstructor = findBackground(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann,
                           BgDir_local=BgDir_local, flatField=flatField,bgCorrect=bgCorrect,
                           ff_method='open', azimuth_offset=azimuth_offset) # find background tile
    img_io.loopZ ='sample'
    plot_config = config['plotting']
    img_io = loopPos(img_io, img_reconstructor, plot_config, flatField=flatField, bgCorrect=bgCorrect,
                     circularity=circularity)
    img_io.chNamesIn = img_io.chNamesOut
    img_io.writeMetaData()
    write_config(config, os.path.join(img_io.ImgOutPath, 'config.yml')) # save the config file in the processed folder

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
    azimuth_offset = config['processing']['azimuth_offset']


    if isinstance(SmDir, list):
        batchProc = True
        SmDirList = SmDir
    else:
        if batchProc:
            ImgPath = os.path.join(RawDataPath, ImgDir)
            SmDirList = GetSubDirName(ImgPath)

    if batchProc:
        if isinstance(BgDir, list):
            BgDirList = BgDir
        else:
            # Make BgDirList same length as SmDirList
            BgDirList = [BgDir] * len(SmDirList)
        assert len(SmDirList) == len(BgDirList), \
            'Length of the background directory list must be one or same as sample directory list'

        for SmDir, BgDir in zip(SmDirList, BgDirList):
            # if 'SM' in SmDir:
            processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann, config,
                       BgDir_local=BgDir_local, flatField=flatField, bgCorrect=bgCorrect,
                       circularity=circularity, azimuth_offset=azimuth_offset)
    else:
        processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann, config,
                   BgDir_local=BgDir_local, flatField=flatField, bgCorrect=bgCorrect,
                   circularity=circularity, azimuth_offset=azimuth_offset)


if __name__ == '__main__':
    args = parse_args()
    run_action(args)