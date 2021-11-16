import os
import numpy as np
import zarr
from dynamorph_seg_map import get_sms_im_name
from waveorder2reconorder import parse_sms_name, read_img, write_img
# from ReconstructOrder.utils.imgIO import get_sub_dirs
from waveorder.io.reader import WaveorderReader
import pprint

if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    # input_path = '/CompMicro/projects/HEK/2021_09_16_HEKs_8orgs_OC43_63x_04NA'
    # output_path = '/CompMicro/projects/HEK/2021_09_16_HEKs_8orgs_OC43_63x_04NA_tif_test'
    # conditions = ['first_TL.zarr']
    # channels = ['Phase3D', 'DRAQ5']
    # chan_ids = [3, 5]
    # z_ids = list(range(24, 34, 2))
    # t_ids = list(range(0, 8, 2))
    # pos_ids = list(range(0, 16, 2))
    input_path = '/CompMicro/projects/HEK/2021_08_25_LiveHEK_63x_09NA_StainedOrgs'
    output_path = '/CompMicro/projects/HEK/2021_08_25_LiveHEK_63x_09NA_StainedOrgs_tif'
    conditions = ['Timelapse.zarr']
    channels = ['Phase3D', 'DRAQ5']
    chan_ids = [0, 2]
    z_ids = list(range(81))
    t_ids = list(range(0, 100, 10))
    pos_ids = [3, 5, 9, 11, 17]
    for condition in conditions:
        print('processing condition {}...'.format(condition))
        # pos_zarrs = get_sub_dirs(os.path.join(input_path, condition + '.zarr'))
        dst_dir = os.path.join(output_path, condition.strip('.zarr'))
        os.makedirs(dst_dir, exist_ok=True)
        t_idx = 0
        reader = WaveorderReader(os.path.join(input_path, condition), data_type = 'zarr')
        pp.pprint(reader.stage_positions)
        pp.pprint(reader.reader.hcs_meta)
        for pos_idx in pos_ids:
            img_tcz = reader.get_zarr(position=pos_idx)  # Returns sliceable array that hasn't been loaded into memory
            for t_idx in t_ids:
                img_cz = img_tcz[t_idx]
                for c_idx, chan in zip(chan_ids, channels):
                    img_z = img_cz[c_idx]
                    for z_idx in z_ids:
                        img = img_z[z_idx]
                        print(
                            'Processing position {}, time {}, channel {}, z {}...'.format(pos_idx, t_idx, chan,
                                                                                          z_idx))
                        im_name_dst = get_sms_im_name(
                            time_idx=t_idx,
                            channel_name=chan,
                            slice_idx=z_idx,
                            pos_idx=pos_idx,
                            ext='.tif',
                        )
                        write_img(img, dst_dir, im_name_dst)


