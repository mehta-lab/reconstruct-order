import os
import warnings
import math
import cv2
import numpy as np
from dynamorph_seg_map import get_sms_im_name
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

def crop2base(im, base=2):
    """
    Crop image to nearest smaller factor of the base (usually 2), assumes xyz
    format, will work for zyx too but the x_shape, y_shape and z_shape will be
    z_shape, y_shape and x_shape respectively

    :param nd.array im: Image
    :param int base: Base to use, typically 2
    :param bool crop_z: crop along z dim, only for UNet3D
    :return nd.array im: Cropped image
    :raises AssertionError: if base is less than zero
    """
    assert base > 0, "Base needs to be greater than zero, not {}".format(base)
    im_shape = im.shape

    x_shape = base ** int(math.log(im_shape[0], base))
    y_shape = base ** int(math.log(im_shape[1], base))
    if x_shape < im_shape[0]:
        # Approximate center crop
        start_idx = (im_shape[0] - x_shape) // 2
        im = im[start_idx:start_idx + x_shape, ...]
    if y_shape < im_shape[1]:
        # Approximate center crop
        start_idx = (im_shape[1] - y_shape) // 2
        im = im[:, start_idx:start_idx + y_shape, ...]
    return im

if __name__ == '__main__':

    input_chan = output_chan = ['Phase3D', 'Retardance']  # first channel is the reference channel
    # input_chan = output_chan = ['Phase3D', 'Retardance', 'GFP']  # first channel is the reference channel
    # input_chan = output_chan = ['Phase3D', 'DAPI', 'Golgi', 'ER', 'MTub']  # first channel is the reference channel
    # input_chan = output_chan = ['phase', 'Retardance']  # first channel is the reference channel
    # input_dirs = ['/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/SPS_Mock_0.4NA_LF_Fluor_2_SPS_Mock_0.4NA_LF_Fluor_2_registered',
    #               '/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/SPS_LowMOI_0.4NA_LF_Flour_5_SPS_LowMOI_0.4NA_LF_Flour_5_registered',]
    # output_dir = '/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_input'
    # input_dirs = ['/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/SMS_2018_1227_1433_1_SMS_2018_1227_1433_1_registered']
    # output_dir = '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_input'
    # input_dirs = ['/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/Mock_24h_right',
    #               '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/Mock_48h_right',
    #               '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/RSV_24h_right',
    #               '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/RSV_48h_right']
    # input_dirs = ['/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/MOCK_IFNA_48',
    #               '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNA_24',
    #               '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNA_48',
    #               '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24']
    # input_dirs = [
    #             # '/CompMicro/projects/HEK/2021_05_12_HEK_RSV_20x_055na_TimeLapse_tif/Mock',
    #               '/CompMicro/projects/HEK/2021_05_12_HEK_RSV_20x_055na_TimeLapse_tif/MOI_1'
    # ]
    exp_dirs = [
        # '/CompMicro/projects/HEK/2021_04_20_HEK_OC43_63x_04NA_Widefield_tif/2021_04_20_HEK_OC43_widefield_registered'
        '/CompMicro/projects/HEK/2021_07_29_LiveHEK_NoPerf_63x_09NA_tif/2021_07_29_LiveHEK_NoPerf_63x_09NA_tif_registered'
    ]

    # p_ids_list = [[0, 1, 2, 3, 4, 5, 7, 9], [0, 1, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 4, 5, 6, 7, 8, 9],
    #               [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    # z_ids_list = [list(range(37, 45))] + [list(range(28, 36))] * 3 # A549 02_25
    # z_ids_list = [list(range(26, 41))] + [list(range(29, 44))] # CM
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
            input_dir = os.path.join(exp_dir, condition)
            output_dir = os.path.join(input_dir, 'dnm_input')
            # parse positions from file name img_{channel}_t###_p###_z###.tif
            im_names = [fname.strip('.tif') for fname in os.listdir(input_dir) if 'im' in fname]
            im_names = [fname for fname in im_names for chan in input_chan if chan in fname]
            pos_ids = set([int(im_name.split("_")[3].strip('p')) for im_name in im_names])
            t_ids = set([int(im_name.split("_")[2].strip('t')) for im_name in im_names])
            # z_ids = set([int(im_name.split("_")[4].strip('z')) for im_name in im_names])
            # z_ids = list(range(26, 41)) # CM mock
            # z_ids = list(range(3, 45))  # kidney slice

            if not os.path.exists(input_dir):
                raise FileNotFoundError(
                    "image file doesn't exist at:", input_dir
                )
            os.makedirs(output_dir, exist_ok=True)

            for pos_idx in pos_ids:  # nXY
                imgs_tcz = []
                for t_idx in t_ids:
                    imgs_cz = []
                    for chan in input_chan:
                        imgs_z = []
                        for z_idx in z_ids:
                            print('Processing position {}, time {}, channel {}, z {}...'.format(pos_idx, t_idx, chan, z_idx))
                            im_name = get_sms_im_name(
                                time_idx=t_idx,
                                channel_name=chan,
                                slice_idx=z_idx,
                                pos_idx=pos_idx,
                                ext='.tif',
                            )
                            img = read_img(dir=input_dir, img_name=im_name)
                            # print(img[:11, :11])
                            img = crop2base(img)
                            imgs_z.append(img)
                        # dynamorph uses tczyx format
                        imgs_z = np.stack(imgs_z, axis=0)
                        imgs_cz.append(imgs_z)
                    imgs_cz = np.stack(imgs_cz, axis=0)
                    imgs_tcz.append(imgs_cz)
                imgs_tcz = np.stack(imgs_tcz, axis=0)
                # imgs_tcz = imgs_tcz.astype(np.uint16)
                img_name = 'img_p%03d.npy' % (pos_idx)
                if swap_tz:
                    imgs_tcz = np.swapaxes(imgs_tcz, 0, 2)
                # print(imgs_tcz.shape)
                # print(imgs_tcz[0, 0, 0, :11, :11])
                np.save(os.path.join(output_dir, img_name), imgs_tcz)