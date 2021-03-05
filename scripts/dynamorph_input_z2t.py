import os
import numpy as np
import cv2
import warnings
import argparse
from ReconstructOrder.utils.mManagerIO import mManagerReader

def parse_args():
    """
    Parse command line arguments

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        type=str,
        help="Path to folder containing 2D images in .npy",
    )

    parser.add_argument(
        '-o',
        '--output',
        required=True,
        type=str,
        help="Path to output folder",
    )

    return parser.parse_args()


def get_sms_im_name(time_idx=None,
                    channel_name=None,
                    slice_idx=None,
                    pos_idx=None,
                    extra_field=None,
                    ext='.npy',
                    int2str_len=3):
    """
    Create an image name given parameters and extension
    This function is custom for the computational microscopy (SMS)
    group, who has the following file naming convention:
    File naming convention is assumed to be:
        img_channelname_t***_p***_z***.tif
    This function will alter list and dict in place.

    :param int time_idx: Time index
    :param str channel_name: Channel name
    :param int slice_idx: Slice (z) index
    :param int pos_idx: Position (FOV) index
    :param str extra_field: Any extra string you want to include in the name
    :param str ext: Extension, e.g. '.png'
    :param int int2str_len: Length of string of the converted integers
    :return st im_name: Image file name
    """

    im_name = "img"
    if channel_name is not None:
        im_name += "_" + str(channel_name)
    if time_idx is not None:
        im_name += "_t" + str(time_idx).zfill(int2str_len)
    if pos_idx is not None:
        im_name += "_p" + str(pos_idx).zfill(int2str_len)
    if slice_idx is not None:
        im_name += "_z" + str(slice_idx).zfill(int2str_len)
    if extra_field is not None:
        im_name += "_" + extra_field
    im_name += ext

    return im_name

def read_img(dir, img_name):
    """read a single image at (c,t,p,z)"""
    img_file = os.path.join(dir, img_name)
    img = cv2.imread(img_file, -1) # flag -1 to preserve the bit dept of the raw image
    if img is None:
        warnings.warn('image "{}" cannot be found. Return None instead.'.format(img_file))
    else:
        img = img.astype(np.float32, copy=False)  # convert to float32 without making a copy to save memory
    return img

if __name__ == '__main__':
    parsed_args = parse_args()
    input_dir = parsed_args.input
    output_dir = parsed_args.output
    img_io = mManagerReader(input_dir, output_dir)
    t_idx = 0
    img_io.binning = 1
    z_ids = list(range(29, 44))

    if not os.path.exists(img_io.img_sm_path):
        raise FileNotFoundError(
            "image file doesn't exist at:", img_io.img_sm_path
        )
    os.makedirs(img_io.img_output_path, exist_ok=True)

    for suffix in [None, 'NNProbabilities']:
        for pos_idx in range(img_io.n_pos):  # nXY
            img_io.pos_idx = pos_idx
            # for z_idx in z_ids:
            #     img_io.z_idx = z_idx
            imgs_tc = []
            # stack z in t dimension
            for z_idx in z_ids:
                print('Processing position %03d, time %03d, ...' % (pos_idx, t_idx))
                im_name = get_sms_im_name(
                    time_idx=t_idx,
                    channel_name=None,
                    slice_idx=z_idx,
                    pos_idx=pos_idx,
                    ext='.npy',
                    extra_field=suffix,
                )
                imgs_c = np.load(os.path.join(input_dir, im_name))
                # dynamorph uses tyxc format
                imgs_tc.append(np.squeeze(imgs_c))
            imgs_tc = np.stack(imgs_tc, axis=0)
            imgs_tc = imgs_tc.astype(np.float32)
            im_name = get_sms_im_name(
                time_idx=None,
                channel_name=None,
                slice_idx=None,
                pos_idx=pos_idx,
                ext='.npy',
                extra_field=suffix,
            )
            np.save(os.path.join(output_dir, im_name), imgs_tc)