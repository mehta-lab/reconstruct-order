import os
import glob
import numpy as np
import cv2
import warnings
from shutil import copy
from ReconstructOrder.utils.imgProcessing import im_bit_convert
from ReconstructOrder.utils.imgIO import get_sorted_names
from dynamorph_seg_map import get_sms_im_name

def parse_sms_name(im_name):
    """
    Parse metadata from file name or file path.
    This function is custom for the computational microscopy (SMS)
    group, who has the following file naming convention:
    File naming convention is assumed to be:
        img_channelname_t***_p***_z***.tif
    This function will alter list and dict in place.

    :param str im_name: File name or path
    :return dict meta_row: One row of metadata given image file name
    """

    time_idx = pos_idx = slice_idx = None
    im_name = im_name[:-4]
    str_split = im_name.split("_")[1:]
    channel_name = str_split[0]
    str_split = str_split[-3:]
    for s in str_split:
        if s.find("t") == 0 and len(s) == 4:
            time_idx = int(s[1:])
        elif s.find("p") == 0 and len(s) == 4:
            pos_idx = int(s[1:])
        elif s.find("z") == 0:
            slice_idx = int(s[1:])
    return channel_name, time_idx, pos_idx, slice_idx

def read_img(img_file):
    """read a single image at (c,t,p,z)"""
    img = cv2.imread(img_file, -1) # flag -1 to preserve the bit dept of the raw image
    if img is None:
        warnings.warn('image "{}" cannot be found. Return None instead.'.format(img_file))
    else:
        img = img.astype(np.float32, copy=False)  # convert to float32 without making a copy to save memory
    return img

def write_img(img, output_dir, img_name):
    """only supports recon_order image name format currently"""
    if not os.path.exists(output_dir):  # create folder for processed images
        os.makedirs(output_dir)
    if len(img.shape) < 3:
        cv2.imwrite(os.path.join(output_dir, img_name), img)
    else:
        cv2.imwrite(os.path.join(output_dir, img_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def split_azimuth(azimuth_degree):
    azimuth = azimuth_degree / 18000 * np.pi
    azimuth_imgs = [np.cos(2 * azimuth), np.sin(2 * azimuth)]
    azimuth_imgs = [im_bit_convert((img + 1) * 1000, bit=16) for img in azimuth_imgs]  # scale to [0, 1000]
    return azimuth_imgs

if __name__ == '__main__':
    input_path = '/CompMicro/projects/waveorderData/data_processing/20210209_Falcon_3D_uPTI_A549_RSV'
    output_path = '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV'
    conditions = ['Mock_24h_right', 'Mock_48h_right', 'RSV_24h_right', 'RSV_48h_right']
    # channels = ['phase3D', 'retardance3D', 'fluor_deconv']
    channels = ['fluor_deconv']
    exp_paths = glob.glob(os.path.join(input_path, '*/'))
    for condition in conditions:
        print('processing condition {}...'.format(condition))
        dst_dir = os.path.join(output_path, condition)
        os.makedirs(dst_dir, exist_ok=True)
        sub_exp_paths = [path for path in exp_paths if condition in path]
        pos_id_offset = 0
        for path in sub_exp_paths:
            src_dir = os.path.join(path, 'Full_FOV_process_full_bg_pad_z_large_patch_high_reg')
            for chan in channels:
                chan_dir = os.path.join(src_dir, chan)
                im_names = get_sorted_names(chan_dir)
                for im_name_src in im_names:
                    im_src_path = os.path.join(chan_dir, im_name_src)
                    channel_name, t_idx, pos_idx, z_idx = parse_sms_name(im_name_src)
                    if channel_name == 'azimuth':
                        # split azimuth into x, y components for model training
                        azimuth = read_img(im_src_path)
                        azimuths = split_azimuth(azimuth)
                        output_chans = ['Orientation_x', 'Orientation_y']
                        for img, channel_name in zip(azimuths, output_chans):
                            im_name_dst = get_sms_im_name(
                                time_idx=t_idx,
                                channel_name=channel_name,
                                slice_idx=z_idx,
                                pos_idx=pos_idx + pos_id_offset,
                                ext='.tif',
                            )
                            write_img(img, dst_dir, im_name_dst)
                    elif 'fluor' in channel_name:
                        # read fluor color image
                        fluor = read_img(im_src_path)
                        # print(fluor.shape)
                        fluors = [fluor[:, :, 0].astype(np.uint8), fluor[:, :, 1].astype(np.uint8)] # split channels
                        output_chans = ['DAPI', 'RSV']
                        for img, channel_name in zip(fluors, output_chans):
                            im_name_dst = get_sms_im_name(
                                time_idx=t_idx,
                                channel_name=channel_name,
                                slice_idx=z_idx,
                                pos_idx=pos_idx + pos_id_offset,
                                ext='.tif',
                            )
                            write_img(img, dst_dir, im_name_dst)
                    else:
                        im_name_dst = get_sms_im_name(
                            time_idx=t_idx,
                            channel_name=channel_name,
                            slice_idx=z_idx,
                            pos_idx=pos_idx + pos_id_offset,
                            ext='.tif',
                        )
                        copy(im_src_path,
                             os.path.join(dst_dir, im_name_dst))
            pos_id_offset += 1