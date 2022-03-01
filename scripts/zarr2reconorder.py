import os
import glob
import numpy as np
import zarr
from waveorder2reconorder import parse_sms_name
from scripts.align_z_focus import read_img
from scripts.hcszarr2sigle_tif import write_img, get_sms_im_name

if __name__ == '__main__':
    input_path = '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549'
    output_path = '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif'
    conditions = ['MOCK_IFNA_48', 'RSV_IFNA_24', 'RSV_IFNA_48', 'RSV_IFNL_24']
    channels = ['Phase3D', 'Retardance', 'GFP']
    # exp_paths = glob.glob(os.path.join(input_path, '*/'))
    for condition in conditions:
        print('processing condition {}...'.format(condition))
        zarr_store = zarr.open(os.path.join(input_path, condition + '.zarr'), mode='r')
        dst_dir = os.path.join(output_path, condition)
        os.makedirs(dst_dir, exist_ok=True)
        t_idx = 0
        for chan in channels:
            img_pz = zarr_store[chan]
            print(img_pz.shape)
            for pos_idx, img_z in enumerate(img_pz):
                for z_idx, img in enumerate(img_z):
                    im_name_dst = get_sms_im_name(
                        time_idx=t_idx,
                        channel_name=chan,
                        slice_idx=z_idx,
                        pos_idx=pos_idx,
                        ext='.tif',
                    )
                    # print(img.shape)
                    write_img(img, dst_dir, im_name_dst)