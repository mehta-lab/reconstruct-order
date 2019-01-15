import sys
sys.path.append("..") # Adds parent directory to python search path.
import os
from utils.mManagerIO import mManagerReader
import argparse
import yaml
from scripts.channel_registration_3D import translate_3D
import json
from utils.imgProcessing import imBitConvert

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
    registration_params_path = config['processing']['registration_params']
    outputChann = config['processing']['outputChann']
    ImgSmPath = os.path.join(RawDataPath, ImgDir, SmDir)  # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'

    OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir+'_registered')
    img_io = mManagerReader(ImgSmPath, OutputPath, outputChann)
    with open(registration_params_path, 'r') as f:
        registration_params = json.load(f)

    if not os.path.exists(img_io.ImgSmPath):
        raise FileNotFoundError(
            "image file doesn't exist at:", img_io.ImgSmPath
        )
    os.makedirs(img_io.ImgOutPath, exist_ok=True)
    for t_idx in range(img_io.nTime):
        img_io.tIdx = t_idx
        for pos_idx in range(img_io.nPos):  # nXY
            img_io.posIdx = pos_idx
            print('Processing position %03d, time %03d ...' % (posIdx, tIdx))
            images = img_io.read_multi_chan_img_stack()
            images_registered = translate_3D(images, outputChann, registration_params)
            for images, channel in zip(images_registered, outputChann):
                for z_idx in range(img_io.nZ):
                    image = imBitConvert(images[z_idx], bit=16, norm=False)
                    img_io.write_img(image, channel, pos_idx, z_idx, t_idx)

if __name__ == '__main__':
    args = parse_args()
    run_action(args)