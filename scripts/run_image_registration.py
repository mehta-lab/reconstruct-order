import sys
sys.path.append("..") # Adds parent directory to python search path.
import os
from ReconstructOrder.utils.mManagerIO import mManagerReader
import argparse
import yaml
import numpy as np
import pandas as pd
from scripts.channel_registration_3D import translate_3D
import json
from ReconstructOrder.utils.imgProcessing import im_bit_convert
from waveorder2reconorder import read_img, write_img
from dynamorph_seg_map import get_sms_im_name
from ReconstructOrder.utils.imgIO import get_sub_dirs

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
    input_dir = config['dataset']['RawDataPath']
    output_dir = config['dataset']['ProcessedPath']
    registration_params_path = config['processing']['registration_params']
    chans = config['processing']['output_chan']
    binning = config['processing']['binning']
    size_z_um = 1
    if 'size_z_um' in config['processing']:
        size_z_um = config['processing']['size_z_um']
    with open(registration_params_path, 'r') as f:
        registration_params = json.load(f)
    conditions = get_sub_dirs(input_dir)
    # conditions = ['RSV_IFNA_24']
    os.makedirs(output_dir, exist_ok=True)
    for condition in conditions:
        print('processing condition {}...'.format(condition))
        # Load frames metadata and determine indices if exists
        fmeta_path = os.path.join(input_dir, condition, 'frames_meta.csv')
        dst_dir = os.path.join(output_dir, condition)
        if os.path.isfile(fmeta_path):
            frames_meta = pd.read_csv(fmeta_path, index_col=0)
        else:
            raise FileNotFoundError('"frames_meta.csv" generated by microDL is required')
        pos_ids = frames_meta['pos_idx'].unique()
        pos_ids.sort()
        for pos_idx in pos_ids:
            frames_meta_p = frames_meta[frames_meta['pos_idx'] == pos_idx]
            t_ids = frames_meta_p['time_idx'].unique()
            t_ids.sort()
            for t_idx in t_ids:
                frames_meta_pt = frames_meta_p[frames_meta_p['time_idx'] == t_idx]
                # Assume all channels have the same number of z
                if 'z_ids' in config['processing']:
                    z_ids = config['processing']['z_ids']
                else:
                    z_ids = frames_meta_pt['slice_idx'].unique()
                    z_ids.sort()
                img_chann = []  # list of 2D or 3D images from different channels
                for chan in chans:
                    frames_meta_ptc = frames_meta_pt[frames_meta_pt['channel_name'] == chan]
                    img_stack = []
                    for z_idx in z_ids:
                        frames_meta_ptcz = frames_meta_ptc[frames_meta_ptc['slice_idx'] == z_idx]
                        im_path = os.path.join(frames_meta_ptcz['dir_name'].values[0],
                                               frames_meta_ptcz['file_name'].values[0])
                        img = read_img(im_path)
                        img_stack += [img]
                    img_stack = np.stack(img_stack)  # follow zyx order
                    img_chann += [img_stack]

                images_registered = translate_3D(img_chann,
                                                 chans,
                                                 registration_params,
                                                 size_z_um,
                                                 binning)

                for chan, images in zip(chans, images_registered):
                    for z_idx, image in zip(z_ids, images):
                        im_name_dst = get_sms_im_name(
                            time_idx=t_idx,
                            channel_name=chan,
                            slice_idx=z_idx,
                            pos_idx=pos_idx,
                            ext='.tif',
                        )
                        write_img(image, dst_dir, im_name_dst)

if __name__ == '__main__':
    args = parse_args()
    run_action(args)