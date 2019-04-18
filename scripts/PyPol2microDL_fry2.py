import sys
sys.path.append(".") # Adds current directory to python search path.
sys.path.append("..") # Adds parent directory to python search path.
import os
from utils.mManagerIO import mManagerReader
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

def run_action(args):
    config = read_config(args.config)
    RawDataPath = config['dataset']['RawDataPath']
    ProcessedPath = config['dataset']['ProcessedPath']
    ImgDir = config['dataset']['ImgDir']
    SmDir = config['dataset']['SmDir']
    outputChann = config['processing']['output_chan']
    ImgSmPath = os.path.join(RawDataPath, ImgDir, SmDir)  # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'

    # OutputPath = os.path.join(ImgSmPath,'split_images')
    # imgSm = mManagerReader(ImgSmPath,OutputPath, output_chan)
    # imgSm.save_microDL_format_old()

    OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir, 'microDL_format')
    img_io = mManagerReader(ImgSmPath, OutputPath, outputChann)
    img_io.save_microDL_format_new()

if __name__ == '__main__':
    args = parse_args()
    run_action(args)



