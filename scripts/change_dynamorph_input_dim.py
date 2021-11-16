"""Change array dimension to 5d (tczyx)"""
import os
import numpy as np
from dynamorph_seg_map import get_sms_im_name
from ReconstructOrder.utils.imgIO import get_sorted_names, get_sub_dirs

def make_dim_5d(array):
    """ Check segmentation mask dimension.
    Add a background channel if n(channels)==1

    Args:
        segmentation: (np.array): segmentation mask for the frame

    """
    if array.ndim == 5:
        pass
    elif array.ndim == 4:
        # add z dimension
        array = array[:, :, np.newaxis, ...]
    elif array.ndim == 3:
        array = array[np.newaxis, np.newaxis, ...]
    elif array.ndim == 2:
        array = array[np.newaxis, np.newaxis, np.newaxis, ...]
    else:
        raise ValueError('array must be at least 2D, not {}'.format(array.ndim))
    if array.shape[0] > 1 and array.shape[2] == 1:
        # swap tz dimension if t is not single and z is single
        array = np.swapaxes(array, 0, 2)

    return array

if __name__ == '__main__':

    input_chan = output_chan = ['Phase3D', 'Retardance']  # first channel is the reference channel
    # input_chan = output_chan = ['Phase3D', 'Retardance', 'GFP']  # first channel is the reference channel
    # input_chan = output_chan = ['Phase3D', 'DAPI', 'Golgi', 'ER', 'MTub']  # first channel is the reference channel
    # input_chan = output_chan = ['phase', 'Retardance']  # first channel is the reference channel
    # input_dirs = [
        # '/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_input_tstack',
        #           '/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_input_tstack',]
    # output_dir = '/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_input'
    # input_dirs = ['/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/SMS_2018_1227_1433_1_SMS_2018_1227_1433_1_registered']
    # input_dirs = ['/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_input']
    # input_dirs = ['/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/Mock_24h_right',
    #               '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/Mock_48h_right',
    #               '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/RSV_24h_right',
    #               '/CompMicro/projects/A549/20210209_Falcon_3D_uPTI_A549_RSV_registered/RSV_48h_right']
    input_dirs = ['/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_input',
                 '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/MOCK_IFNA_48/dnm_input',
                  '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNA_24/dnm_input',
                  '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNA_48/dnm_input',
                  '/CompMicro/projects/A549/2021_02_25_40X_04NA_A549_tif_registered/RSV_IFNL_24/dnm_input']
    # input_dirs = [
    #             # '/CompMicro/projects/HEK/2021_05_12_HEK_RSV_20x_055na_TimeLapse_tif/Mock',
    #               '/CompMicro/projects/HEK/2021_05_12_HEK_RSV_20x_055na_TimeLapse_tif/MOI_1'
    # ]
    # exp_dirs = [
    #     # '/CompMicro/projects/HEK/2021_04_20_HEK_OC43_63x_04NA_Widefield_tif/2021_04_20_HEK_OC43_widefield_registered'
    #     '/CompMicro/projects/HEK/2021_07_29_LiveHEK_NoPerf_63x_09NA_tif/2021_07_29_LiveHEK_NoPerf_63x_09NA_tif_registered'
    # ]

    for input_dir in input_dirs:
        output_dir = input_dir
        # parse positions from file name img_{channel}_t###_p###_z###.tif
        im_names = [fname for fname in os.listdir(input_dir) if 'img' in fname and 'masks.npy' in fname and '_seg' not in fname]
        os.makedirs(output_dir, exist_ok=True)
        for im_name in im_names:
            img = np.load(os.path.join(input_dir, im_name))
            print('before:', img.shape)
            img = make_dim_5d(img)
            print('after:', img.shape)
            np.save(os.path.join(output_dir, im_name), img)