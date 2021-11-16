import cv2
import os
import numpy as np
import argparse
from ReconstructOrder.utils.imgIO import get_sorted_names, get_sub_dirs

def read_img(dir, img_name, to_float=True):
    """read a single image at (c,t,p,z)"""
    img_file = os.path.join(dir, img_name)
    img = cv2.imread(img_file, -1) # flag -1 to preserve the bit dept of the raw image
    if img is None:
        raise FileNotFoundError('image "{}" cannot be found'.format(img_file))
    if to_float:
        img = img.astype(np.float32, copy=False)  # convert to float32 without making a copy to save memory
    return img

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

if __name__ == '__main__':
    # parsed_args = parse_args()
    # input_dir = parsed_args.input
    # output_dir = parsed_args.output
    # input_chan = output_chan = ['phase3D', 'retardance3D']  # first channel is the reference channel
    # input_chan = output_chan = ['Phase3D', 'Retardance']  # first channel is the reference channel
    # input_dir = '/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/'
    # output_dir = '/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_input'
    # input_dirs = ['/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/SMS_2018_1227_1433_1_SMS_2018_1227_1433_1_registered/2D_4channel_nuclei_mask_do40']
    # output_dir = '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_input'
    # input_dirs = ['/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/Mock_24h_right/2D_bce_16_256_mean_pool_do40_monitor_dice',
    #               '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/Mock_48h_right/2D_bce_16_256_mean_pool_do40_monitor_dice',
    #               '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/RSV_24h_right/2D_bce_16_256_mean_pool_do40_monitor_dice',
    #               '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/RSV_48h_right/2D_bce_16_256_mean_pool_do40_monitor_dice']

    # input_dirs = ['/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/MOCK_IFNA_48/A549_QLIPP_16_256_MOCK_IFNA_temp_10_do40_rot',
    #               '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNA_24/A549_uPTI_16_256_do20_monitor_dice_zoom_0.8',
    #               '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNA_48/A549_uPTI_16_256_do20_monitor_dice_zoom_0.8',
    #               '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/A549_QLIPP_16_256_MOCK_IFNA_temp_10_do40_rot']
    # input_dirs = [
    #     # '/CompMicro/projects/HEK/2021_05_12_HEK_RSV_20x_055na_TimeLapse_tif/Mock/HEK_20X_16_256_do20_zoom_1.4_1.6_tilenorm_pt20',
    #               '/CompMicro/projects/HEK/2021_05_12_HEK_RSV_20x_055na_TimeLapse_tif/MOI_1/HEK_20X_16_256_do20_zoom_1.4_1.6_tilenorm_pt20'
    # ]
    exp_dirs = [
        # '/CompMicro/projects/HEK/2021_04_20_HEK_OC43_63x_04NA_Widefield_tif/2021_04_20_HEK_OC43_widefield_registered'
        '/CompMicro/projects/HEK/2021_07_29_LiveHEK_NoPerf_63x_09NA_tif/2021_07_29_LiveHEK_NoPerf_63x_09NA_tif_registered/'
    ]
    output_dir = '/CompMicro/projects/HEK/2021_07_29_LiveHEK_NoPerf_63x_09NA_tif_registered'
    # seg_map_dir = 'HEK_pool_63X_16_256_do20_zoom_1_pt20'
    seg_map_dir = 'Mock_seg'
    # p_ids_list = [[0, 1, 2, 3, 4, 5, 7, 9], [0, 1, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    # z_ids_list = [list(range(37, 45))] + [list(range(28, 36))] * 3
    # z_ids_list = [list(range(15, 36))]  # kidney
    # z_ids_list = [list(range(10, 15))]  # HEK 20X
    # z_ids_list = [list(range(32, 51))]  # HEK 63X
    z_ids_list = [list(range(31, 36))]  # HEK 63X
    swap_tz = False
    # for input_dir, pos_ids, z_ids in zip(input_dirs, p_ids_list, z_ids_list):
    for exp_dir in exp_dirs:
        conditions = [s for s in get_sub_dirs(exp_dir)]
        print(conditions)
        if len(z_ids_list) == 1 and len(conditions) > 1:
            z_ids_list = z_ids_list * len(conditions)
        for condition, z_ids in zip(conditions, z_ids_list):
            input_dir = os.path.join(exp_dir, condition, seg_map_dir)
            output_dir = os.path.join(output_dir, condition, 'dnm_input')
            im_names = [fname.strip('.png') for fname in os.listdir(input_dir) if 'cp_masks' in fname]
            print(int(im_names[0].split("_")[2].strip('p')))
            pos_ids = set([int(im_name.split("_")[2].strip('p')) for im_name in im_names])
            t_ids = set([int(im_name.split("_")[1].strip('t')) for im_name in im_names])
            # z_ids = list(range(29, 44)) # CM low MOI
            # z_ids = list(range(3, 45))  # kidney slice
            # z_ids = list(range(30, 40))

            if not os.path.exists(input_dir):
                raise FileNotFoundError(
                    "image file doesn't exist at:", input_dir
                )
            os.makedirs(output_dir, exist_ok=True)

            for suffix in ['NNProbabilities_cp_masks']:
                for pos_idx in pos_ids:  # nXY
                    imgs_tcz = []
                    for t_idx in t_ids:
                        imgs_z = []
                        for z_idx in z_ids:
                            print('Processing position %03d, time %03d, z %03d...' % (pos_idx, t_idx, z_idx))
                            im_name = get_sms_im_name(
                                time_idx=t_idx,
                                channel_name=None,
                                slice_idx=z_idx,
                                pos_idx=pos_idx,
                                ext='.png',
                                extra_field=suffix,
                            )
                            imgs = read_img(dir=input_dir, img_name=im_name, to_float=False)
                            # dynamorph segmentation map uses tczyx format
                            imgs_z.append(np.squeeze(imgs))
                        imgs_z = np.stack(imgs_z, axis=0)
                        imgs_cz = imgs_z[np.newaxis, ...]
                        imgs_tcz.append(imgs_cz)
                    imgs_tcz = np.stack(imgs_tcz, axis=0)
                    # imgs_tcz = imgs_tcz.astype(np.float32)
                    im_name = get_sms_im_name(
                        time_idx=None,
                        channel_name=None,
                        slice_idx=None,
                        pos_idx=pos_idx,
                        ext='.npy',
                        extra_field='NNProbabilities',
                    )
                    if swap_tz:
                        imgs_tcz = np.swapaxes(imgs_tcz, 0, 2)
                    # print(imgs_tcz.shape)
                    np.save(os.path.join(output_dir, im_name), imgs_tcz)