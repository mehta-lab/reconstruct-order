import sys
sys.path.append(".") # Adds current directory to python search path.
sys.path.append("..") # Adds parent directory to python search path.
from scipy.ndimage import affine_transform, sobel
from skimage.feature import register_translation
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils.mManagerIO import mManagerReader
from utils.plotting import CompositeImg, plot_sub_images

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
    shifts = [[0,0,0]]
    for image in target_images[1:]:
        shift, error, phase = register_translation(target_images[0], image, 4)
        shifts.append(list(shift))
    registration_params = dict(zip(channels, shifts))
    return registration_params

def translate_3D(images, channels, registration_params, size_z_um):
    """
    Warps images using the provided transformation objects, and displays the overlaid images.

    @param images: list of images.
    @param registration_params: dictionary of channel (key) and its translational shifts [z, y, x] (value)
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
                    'Transmission', 'Brightfield', 'Brightfield_computed']:
            chan = 'Retardance'

        # Brightfield registration is not robust
        # elif chan in ['Transmission', 'Brightfield']:
        #     chan = '568'
        # !!!!"[:]" is necessary to create a copy rather than a reference of the list in the dict!!!!
        shift = registration_params[chan][:]
        # only warp the image if shift is non-zero
        if any(shift):
            # scale z-shift according to the z-step size
            shift[0] = shift[0]*registration_params['size_z_um']/size_z_um
            image = affine_transform(image, np.ones(3), [-x for x in shift], order=1)
        registered_images.append(image)
    return registered_images

def imshow_xy_xz_slice(img_stacks, img_io, y_crop_range, z_crop_range,
                       y_plot_range, z_plot_range):
    for zIdx in range(z_plot_range[0], z_plot_range[1]):
        img_io.zIdx = zIdx
        output_chan = img_io.chNamesOut
        img_stack = [img[zIdx - z_crop_range[0], :, :] for img in img_stacks]
        fig_name = 'img_pair_z%03d.png' % (zIdx)
        imshow_pair(img_stack, output_chan, OutputPath, fig_name)
        fig_name = 'img_pair_z%03d_2.png' % (zIdx)
        imshow_pair(img_stack[1:] + [img_stack[0]],
                    output_chan[1:] + [output_chan[0]], OutputPath, fig_name)
        plt.close("all")

    for yIdx in range(y_plot_range[0], y_plot_range[1]):
        img_io.yIdx = yIdx
        img_stack = [np.squeeze(img[:, yIdx - y_crop_range[0], :]) for img in img_stacks]
        fig_name = 'img_pair_y%03d.png' % (yIdx)
        imshow_pair(img_stack, output_chan, OutputPath, fig_name)
        fig_name = 'img_pair_y%03d_2.png' % (yIdx)
        imshow_pair(img_stack[1:] + [img_stack[0]],
                    output_chan[1:] + [output_chan[0]], OutputPath, fig_name)
        plt.close("all")

def edge_filter_2D(img):
    dx = sobel(img, 0)  # horizontal derivative
    dy = sobel(img, 1)  # vertical derivative
    img_edge = np.hypot(dx, dy)  # magnitude

    return img_edge



if __name__ == '__main__':
    # RawDataPath = r'D:/Box Sync/Data'
    # ProcessedPath = r'D:/Box Sync/Processed/'
    RawDataPath = '//flexo/MicroscopyData/ComputationalMicroscopy\SpinningDisk\RawData/Dragonfly_Calibration'
    # ProcessedPath = '/flexo/ComputationalMicroscopy/Projects/Dragonfly_Calibration'
    RawDataPath = r'Z:/ComputationalMicroscopy/SpinningDisk/RawData/Dragonfly_Calibration'
    ProcessedPath = r'Z:/ComputationalMicroscopy/Projects/Dragonfly_Calibration'
    ImgDir = '2019_04_09_Argolight'
    SmDir = '2019_04_08_Argolight_488_561_637_Widefield_PolStates_BF_1_2019_04_08_Argolight_488_561_637_Widefield_PolStates_BF_1_flat'
    input_chan = output_chan = ['568', 'Retardance', 'Transmission', '488', '640'] # first channel is the reference channel

    z_crop_range = [0, 180]
    x_crop_range = [130, 700]
    y_crop_range = [120, 700]
    z_plot_range = [7,11]
    y_plot_range = [194, 204]
    ImgSmPath = os.path.join(ProcessedPath, ImgDir, SmDir) # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'
    OutputPath = os.path.join(ImgSmPath,'registration', 'raw')
    shift_file_path = os.path.join(ImgSmPath, 'registration', 'registration_param_ref_568_63X.json')
    img_io = mManagerReader(ImgSmPath, OutputPath, input_chan=input_chan, output_chan=output_chan)
    img_io.posIdx = 0
    img_io.tIdx = 0
    target_images = img_io.read_multi_chan_img_stack(z_range=z_crop_range)
    os.makedirs(img_io.ImgOutPath, exist_ok=True)

    target_images_cropped = [target_image[:, y_crop_range[0]:y_crop_range[1],
                             x_crop_range[0]:x_crop_range[1]] for target_image in target_images]
    # use edge filter to change BF image to positive contrast (doesn't work for noisy images)
    target_images_filtered = []
    for chan, img in zip(input_chan, target_images_cropped):
        if chan in ['Transmission', 'Brightfield', 'Brightfield_computed']:
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

    target_images_warped = translate_3D(target_images_filtered, output_chan, registration_params, size_z_um)

    OutputPath = os.path.join(ImgSmPath,'registration', 'processed')
    img_io.ImgOutPath = OutputPath
    os.makedirs(img_io.ImgOutPath, exist_ok=True)

    imshow_xy_xz_slice(target_images_warped, img_io, y_crop_range, z_crop_range,
                       y_plot_range, z_plot_range)

    with open(shift_file_path, 'w') as f:
        json.dump(registration_params, f, indent=4)