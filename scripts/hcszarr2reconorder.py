import os
import glob
import numpy as np
import zarr
from waveorder2reconorder import parse_sms_name
from scripts.align_z_focus import read_img
from scripts.hcszarr2sigle_tif import write_img, get_sms_im_name
from ReconstructOrder.utils.imgIO import get_sub_dirs

if __name__ == '__main__':
    input_path = '/CompMicro/projects/HEK/2021_09_16_HEKs_8orgs_OC43_63x_04NA'
    output_path = '/CompMicro/projects/HEK/2021_09_16_HEKs_8orgs_OC43_63x_04NA_tif_test'
    conditions = ['first_TL.zarr']
    channels = ['Phase3D', 'DRAQ5']
    chan_ids = [3, 5]
    z_ids = list(range(24, 34, 2))
    t_ids = list(range(8, 2))


    # exp_paths = glob.glob(os.path.join(input_path, '*/'))
    for condition in conditions:
        print('processing condition {}...'.format(condition))
        # pos_zarrs = get_sub_dirs(os.path.join(input_path, condition + '.zarr'))
        dst_dir = os.path.join(output_path, condition.strip('.zarr'))
        os.makedirs(dst_dir, exist_ok=True)
        t_idx = 0
        zarr_store = zarr.open(os.path.join(input_path, condition), mode='r')
        rows = get_sub_dirs(os.path.join(input_path, condition))
        for row in rows:
            cols = get_sub_dirs(os.path.join(input_path, condition, row))
            for col in cols:
                poses = get_sub_dirs(os.path.join(input_path, condition, row, col))
                for pos in poses:
                    img_tcz = zarr_store[row][col][pos]['array']
                    pos_idx = int(pos_zarr[:-5].split("_")[1])
                    # pos_idx = 0
                    for t_idx, img_cz in enumerate(img_tcz):
                        if t_idx not in t_ids:
                            continue
                        for c_idx, chan in zip(chan_ids, channels):
                            img_z = img_cz[c_idx]
                            for z_idx, img in enumerate(img_z):
                                if z_idx not in z_ids:
                                    continue
                                print(
                                    'Processing position {}, time {}, channel {}, z {}...'.format(pos_idx, t_idx, chan, z_idx))
                                im_name_dst = get_sms_im_name(
                                    time_idx=t_idx,
                                    channel_name=chan,
                                    slice_idx=z_idx,
                                    pos_idx=pos_idx,
                                    ext='.tif',
                                )
                                write_img(img, dst_dir, im_name_dst)