import sys
sys.path.append(".") # Adds current directory to python search path.
sys.path.append("..") # Adds parent directory to python search path.
from scipy.ndimage import affine_transform, sobel
from skimage.feature import register_translation
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from ReconstructOrder.utils.mManagerIO import mManagerReader
from ReconstructOrder.utils.plotting import CompositeImg, plot_sub_images

def imshow_pair(images, chann_names, OutputPath, fig_name):
    image_pairs = []
    titles = []
    image_ref = images[0]
    chann_names_ref = chann_names[0]
    for image, chann_name in zip(images[1:], chann_names[1:]):
        image_pair = [image_ref, image]
        title = ['{}_{}'.format(chann_names_ref, chann_name)]
        # plot_sub_images(image_pairs, ['0', '1'], OutputPath, fig_name, colorbar=False)
        image_pair_rgb = CompositeImg(image_pair, norm=True)
        image_pairs += [image_pair_rgb]
        titles += title
    plot_sub_images(image_pairs, titles, OutputPath, fig_name, colorbar=False)

def channel_register(target_images, channels):
    """
    Returns a list of shift vectors for unregistered images.
    Requires scikit-image 0.15 or above.
    :param target_images: list of target images.
    :return registration_params: dictionary of channel (key) and its translational shifts [z, y, x] (value)
    """
    channels = ['640' if channel == 'ex561em700' else channel for channel in channels]
    shifts = [[0,0,0]]
    for image in target_images[1:]:
        shift, error, phase = register_translation(target_images[0], image, 4)
        shifts.append(list(shift))
    registration_params = dict(zip(channels, shifts))
    return registration_params

def translate_3D(images,
                 channels,
                 registration_params,
                 size_z_um,
                 binning):
    """
    
    Parameters
    ----------
    images : list
        list of images to translate
    channels : list 
        list of channels corresponding to the images
    registration_params : dict 
        dictionary of channel (key) and its translational shifts [z, y, x] (value)
    size_z_um : float 
        z step size in um
    binning : int 
        total xy binning that has been applied during processing compared to the raw images.  

    Returns
    -------
    registered_images : 
    """"""
    

    @param images: list of images.
    @param registration_params: 
    @return: reg_img - list of registered images.
    """

    # applying transformations to all images except for the first one (first channel is source img).
    registered_images = []
    for chan, image in zip(channels, images):
        # use shifts of retardance channel for all label-free channels
        if chan in ['Retardance', 'Orientation','Orientation_x',
                    'Orientation_y', 'Polarization', 'Scattering',
                    'Pol_State_0', 'Pol_State_1',
                    'Pol_State_2', 'Pol_State_3', 'Pol_State_4',
                    'Transmission', 'Brightfield', 'Brightfield_computed', 'phase']:
            chan = 'Retardance'
        elif chan == 'ex561em700':
            chan = '640'

        # Brightfield registration is not robust
        # elif chan in ['Transmission', 'Brightfield']:
        #     chan = '568'
        # !!!!"[:]" is necessary to create a copy rather than a reference of the list in the dict!!!!
        shift = registration_params[chan][:]
        # only warp the image if shift is non-zero
        if any(shift):
            if size_z_um == 0: # 2D translation
                shift[0] = 0
            else:
                # 3D translation. Scale z-shift according to the z-step size.
                shift[0] = shift[0]*registration_params['size_z_um']/size_z_um
            if not binning == 1:
                shift[1] = shift[1] / binning
                shift[2] = shift[2] / binning
            image = affine_transform(image, np.ones(3), [-x for x in shift], order=1)
        registered_images.append(image)
    return registered_images

def imshow_xy_xz_slice(img_stacks, img_io, y_crop_range, z_crop_range,
                       y_plot_range, z_plot_range):
    for z_idx in range(z_plot_range[0], z_plot_range[1]):
        img_io.z_idx = z_idx
        output_chan = img_io.output_chans
        img_stack = [img[z_idx - z_crop_range[0], :, :] for img in img_stacks]
        fig_name = 'img_pair_z%03d.png' % (z_idx)
        imshow_pair(img_stack, output_chan, img_io.img_output_path, fig_name)
        fig_name = 'img_pair_z%03d_2.png' % (z_idx)
        imshow_pair(img_stack[1:] + [img_stack[0]],
                    output_chan[1:] + [output_chan[0]], img_io.img_output_path, fig_name)
        plt.close("all")

    for yIdx in range(y_plot_range[0], y_plot_range[1]):
        img_io.yIdx = yIdx
        img_stack = [img[:, yIdx - y_crop_range[0], :] for img in img_stacks]
        fig_name = 'img_pair_y%03d.png' % (yIdx)
        imshow_pair(img_stack, output_chan, img_io.img_output_path, fig_name)
        fig_name = 'img_pair_y%03d_2.png' % (yIdx)
        imshow_pair(img_stack[1:] + [img_stack[0]],
                    output_chan[1:] + [output_chan[0]], img_io.img_output_path, fig_name)
        plt.close("all")

def edge_filter_2D(img):
    dx = sobel(img, 0)  # horizontal derivative
    dy = sobel(img, 1)  # vertical derivative
    img_edge = np.hypot(dx, dy)  # magnitude

    return img_edge



if __name__ == '__main__':
    RawDataPath = r'Y:/SpinningDisk/RawData/Dragonfly_Calibration'
    ProcessedPath = r'Y:/Projects/Dragonfly_Calibration'
    # ImgDir = '2019_05_20_Argolight_10X_widefield_zyla'
    # SmDir = 'SMS_052019_1842_1_SMS_052019_1842_1_fit_order2'
    ImgDir = 'BFalignment_20191114_CG'
    SmDir = 'BF_Confocal-DAPI-GFP-RFP-IFP_1'
    # input_chan = output_chan = ['640', 'Retardance', 'Brightfield_computed', '405', '488', '568'] # first channel is the reference channel
    input_chan = output_chan = ["EMCCD_Confocal40_RFP",
                                "EMCCD_BF_Confocal",
                                "EMCCD_Confocal40_DAPI",
                                "EMCCD_Confocal40_GFP",
                                "EMCCD_Confocal40_IFP"]  # first channel is the reference channel

    z_crop_range = [0, 161]
    x_crop_range = [0, 460]
    y_crop_range = [0, 383]
    z_plot_range = [0, 161]
    y_plot_range = [0, 383]
    img_sm_path = os.path.join(RawDataPath, ImgDir, SmDir) # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'
    OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir,'registration', 'raw')
    shift_file_path = os.path.join(ProcessedPath, ImgDir, SmDir, 'registration', 'registration_param_63X.json')
    img_io = mManagerReader(img_sm_path, OutputPath, input_chans=input_chan, output_chans=output_chan)
    img_io.pos_idx = 0
    img_io.t_idx = 0
    img_io.binning = 1
    target_images = img_io.read_multi_chan_img_stack(z_range=z_crop_range)
    os.makedirs(img_io.img_output_path, exist_ok=True)

    target_images_cropped = [target_image[:, y_crop_range[0]:y_crop_range[1],
                             x_crop_range[0]:x_crop_range[1]] for target_image in target_images]
    # use edge filter to change BF image to positive contrast (doesn't work for noisy images)
    target_images_filtered = []
    for chan, img in zip(input_chan, target_images_cropped):
        if any([name in chan for name in ['Transmission', 'Brightfield', 'BF']]):
            imgs_filtered = []
            for z_idx in range(img.shape[2]):
                img_filtered = edge_filter_2D(img[:, :, z_idx])
                imgs_filtered.append(img_filtered)
            img = np.stack(imgs_filtered, axis=2)
        target_images_filtered.append(img)

    imshow_xy_xz_slice(target_images_filtered, img_io, y_crop_range, z_crop_range,
                       y_plot_range, z_plot_range)

    registration_params = channel_register(target_images_filtered, output_chan)
    registration_params['size_z_um'] = size_z_um = img_io.size_z_um

    with open(shift_file_path, 'w') as f:
        json.dump(registration_params, f, indent=4)

    target_images_warped = translate_3D(target_images_filtered,
                                        output_chan,
                                        registration_params,
                                        size_z_um,
                                        img_io.binning)

    img_io.img_output_path = os.path.join(ProcessedPath, ImgDir, SmDir,'registration', 'processed')
    os.makedirs(img_io.img_output_path, exist_ok=True)

    imshow_xy_xz_slice(target_images_warped, img_io, y_crop_range, z_crop_range,
                       y_plot_range, z_plot_range)

