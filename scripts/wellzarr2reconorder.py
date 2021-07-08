import os
import glob
import numpy as np
import zarr
from dynamorph_seg_map import get_sms_im_name
from waveorder2reconorder import parse_sms_name, read_img, write_img
from ReconstructOrder.utils.imgIO import get_sub_dirs

if __name__ == '__main__':
    input_path = '/CompMicro/projects/HEK/2021_04_20_HEK_OC43_63x_04NA_Widefield'
    output_path = '/CompMicro/projects/HEK/2021_04_20_HEK_OC43_63x_04NA_Widefield_tif'
    experiments = ['2021_04_20_HEK_OC43_widefield']
    channels = ['Phase3D', 'DAPI', 'Golgi', 'ER', 'MTub']
    # chan_ids = [0, 1]
    chan_ids = list(range(5))
    exp_paths = glob.glob(os.path.join(input_path, '*/'))
    for experiment in experiments:
        print('processing experiment {}...'.format(experiment))
        well_poses = get_sub_dirs(os.path.join(input_path, experiment + '.zarr'))
        dst_dir = os.path.join(output_path, experiment)
        os.makedirs(dst_dir, exist_ok=True)
        t_idx = 0
        for well_pos in well_poses:
            zarr_store = zarr.open(os.path.join(input_path, experiment + '.zarr'), mode='r')
            img_tcz = zarr_store[well_pos]['physical_data_deconvolved']['array']
            well_idx = int(well_pos.split("_")[1])
            pos_idx = int(well_pos.split("_")[3])
            dst_dir = os.path.join(output_path, experiment, 'Well' + str(well_idx))
            os.makedirs(dst_dir, exist_ok=True)
            # dst_dir = os.path.join(output_path, experiment, 'Well' + str(well_idx), 'Pos' + str(pos_idx)).zfill(3)
            # os.makedirs(dst_dir, exist_ok=True)
            for t_idx, img_cz in enumerate(img_tcz):
                for c_idx, chan in zip(chan_ids, channels):
                    img_z = img_cz[c_idx]
                    for z_idx, img in enumerate(img_z):
                        print(
                            'Processing position {}, time {}, channel {}, z {}...'.format(pos_idx, t_idx, chan, z_idx))
                        im_name_dst = get_sms_im_name(
                            time_idx=t_idx,
                            channel_name=chan,
                            slice_idx=z_idx,
                            pos_idx=pos_idx,
                            ext='.tif',
                        )
                        # print(img.shape)
                        write_img(img, dst_dir, im_name_dst)