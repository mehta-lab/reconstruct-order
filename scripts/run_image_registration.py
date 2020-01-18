import sys
sys.path.append("..") # Adds parent directory to python search path.
import os
from ReconstructOrder.metadata.mManagerIO import mManagerReader
import argparse
import yaml
from scripts.channel_registration_3D import translate_3D
import json
from ReconstructOrder.utils.imgProcessing import im_bit_convert

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
    :param str config_fname: fname of config yaml with its full path
    :return: dict dictionary of config parameters
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
    input_chan = output_chan = config['processing']['output_chan']
    binning = config['processing']['binning']

    img_sm_path = os.path.join(RawDataPath, ImgDir, SmDir)

    OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir+'_registered')
    img_io = mManagerReader(img_sm_path, OutputPath, input_chans=input_chan, output_chans=output_chan)
    size_z_um = img_io.size_z_um
    z_ids = list(range(0, img_io.n_z))
    if 'z_ids' in config['processing']:
        z_ids = config['processing']['z_ids']
    with open(registration_params_path, 'r') as f:
        registration_params = json.load(f)

    if not os.path.exists(img_io.img_sm_path):
        raise FileNotFoundError(
            "image file doesn't exist at:", img_io.img_sm_path
        )
    os.makedirs(img_io.img_output_path, exist_ok=True)
    for t_idx in range(img_io.n_time):
        img_io.t_idx = t_idx
        for pos_idx in range(img_io.n_pos):  # nXY
            img_io.pos_idx = pos_idx
            print('Processing position %03d, time %03d ...' % (pos_idx, t_idx))
            images = img_io.read_multi_chan_img_stack(z_ids=z_ids)
            images_registered = translate_3D(images,
                                             output_chan,
                                             registration_params,
                                             size_z_um,
                                             binning)

            for chan_idx, images in enumerate(images_registered):
                img_io.chan_idx = chan_idx
                for idx, z_idx in enumerate(z_ids):
                    img_io.z_idx = z_idx
                    image = im_bit_convert(images[idx], bit=16, norm=False)
                    img_io.write_img(image)

if __name__ == '__main__':
    args = parse_args()
    run_action(args)