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
from utils.imgIO import GetSubDirName, readMetaData
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
        
    assert 'RawDataPath' in config['dataset'], \
        'Please provde RawDataPath in config file'
    assert 'ProcessedPath' in config['dataset'], \
        'Please provde ProcessedPath in config file'
    assert 'ImgDir' in config['dataset'], \
        'Please provde ImgDir in config file'
    assert 'SmDir' in config['dataset'], \
        'Please provde SmDir in config file'
    config['dataset'].setdefault('BgDir', [])

    config['processing'].setdefault('outputChann', ['Transmission', 'Retardance', 'Orientation', 'Scattering'])
    config['processing'].setdefault('circularity', 'rcp')
    config['processing'].setdefault('bgCorrect', 'None')
    config['processing'].setdefault('flatField', False)
    config['processing'].setdefault('batchProc', False)
    config['processing'].setdefault('azimuth_offset', 0)
    config['processing'].setdefault('PosList', 'all')
    config['processing'].setdefault('separate_pos', True)
    
    config['plotting'].setdefault('norm', True)
    config['plotting'].setdefault('save_fig', False)
    config['plotting'].setdefault('save_stokes_fig', False)

    return config

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
    config = read_config(args.config)
    RawDataPath = config['dataset']['RawDataPath']
    ProcessedPath = config['dataset']['ProcessedPath']
    ImgDir = config['dataset']['ImgDir']
    SmDir = config['dataset']['SmDir']
    PosList = config['dataset']['PosList']
    BgDir = config['dataset']['BgDir']

    if not isinstance(SmDir, list):
        if batchProc:
            ImgPath = os.path.join(RawDataPath, ImgDir)
            SmDir = GetSubDirName(ImgPath)
        else:
            SmDir = [SmDir]

    # if input is e.g. 'all' or 'Pos1', use for all samples
    if not isinstance(PosList, list):
        PosList = [PosList]*len(SmDir)
    # if input is ['Pos1','Pos2','Pos3'], use for all samples
    elif not any(isinstance(i, list) for i in PosList):
        PosList = [PosList]*len(SmDir)
        
    # Make BgDir same length as SmDir
    if not isinstance(BgDir, list):
        BgDir = [BgDir] * len(SmDir)
            
    assert len(SmDir) == len(BgDir) == len(PosList), \
        'Length of the background directory list must be one or same as sample directory list'
            
    # Check that all paths to be analyzed exist
    img_io =[]; img_io_bg = []
    for SmDir_, BgDir_, PosList_ in zip(SmDir, BgDir, PosList):
        img_io, img_io_bg = readMetaData(RawDataPath, ProcessedPath, ImgDir, SmDir_, BgDir_, PosList_, config)
        img_io.SmDir = 
        img_io.BgDir
        img_io.PosList
        PosList[i] = img_io.PosList
        checkThatAllDirsExist(SmDir, BgDir, PosList):
                # OutputPath = OutputPath + '_pol'
        img_io.ImgOutPath = img_io_bg.OutputPath
        os.makedirs(OutputPath, exist_ok=True)  # create folder for processed images

    processImg(img_io, img_io_bg, config)

class args:
    config = '/Users//ivan.ivanov//Documents/Benchmark/config_Benchmark.txt'

if __name__ == '__main__':
    args = args()
#    args = parse_args()
    run_action(args)