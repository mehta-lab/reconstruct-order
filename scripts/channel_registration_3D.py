import sys
sys.path.append(".") # Adds current directory to python search path.
sys.path.append("..") # Adds parent directory to python search path.
from skimage import io, exposure, util, filters
from skimage.transform import SimilarityTransform, warp
from skimage.feature import register_translation
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils.mManagerIO import mManagerReader
from utils.plotting import CompositeImg, plot_sub_images


def imshow_pair(images, img_io, fig_name):
    chann_names = img_io.chNamesOut[:] # adding "[:]" makes a copy of the variable instead of creating a reference
    image_pairs = []
    titles = []
    image_ref = images.pop(0)
    chann_names_ref = chann_names.pop(0)
    for image, chann_name in zip(images, chann_names):
        image_pair = [image_ref, image]
        title = ['{}_{}'.format(chann_names_ref, chann_name)]
        image_pair_rgb = CompositeImg(image_pair, norm=True)
        image_pairs += [image_pair_rgb]
        titles += title
    plot_sub_images(image_pairs, titles, img_io.ImgOutPath, fig_name, colorbar=False)

def channel_register(target_images, channels):
    """
    Returns a list of shift vectors for unregistered images.
    :param target_images: list of target images.
    :return registration_params: dictionary of channel (key) and its translational shifts [z, y, x] (value)
    """

    # Applying sobel edge filter to brightfield and phase images.
    # target_images[5] = filters.sobel(target_images[5])
    # target_images[6] = filters.sobel(target_images[6])

    # Finding shift vectors that register first channel to all others. Subpixel registration is achieved by
    # upsampling by a factor of 4.
    # no shift for the reference channel
    shifts = [[0,0,0]]
    for image in target_images[1:]:
        shift, error, phase = register_translation(target_images[0], image, 4)
        shifts.append(list(shift))
    registration_params = dict(zip(channels, shifts))
    return registration_params

def translate_3D(images, channels, registration_params):
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
        if chan in ['Transmission', 'Retardance', 'Orientation', 'Scattering']:
            chan = 'Retardance'
        shift = registration_params[chan]
        # only warp the image if shift is non-zero
        if any(shift):
            output_shape = image.shape
            coords = np.mgrid[:output_shape[0], :output_shape[1], :output_shape[2]]
            # change the data type to float to allow subpixel registration
            coords = coords.astype(np.float32)
            for i in range(3):
                coords[i] = coords[i] - shift[i]
            image = warp(image, coords)

        registered_images.append(image)
    return registered_images

# RawDataPath = r'D:/Box Sync/Data'
# ProcessedPath = r'D:/Box Sync/Processed/'
RawDataPath = '/flexo/ComputationalMicroscopy/SpinningDisk/RawData/Dragonfly_Calibration'
ProcessedPath = '/flexo/ComputationalMicroscopy/Processed/virtualstaining/Dragonfly_Calibration'
ImgDir = '2018_11_26_Argolight_channel_registration_63X_confocal'
SmDir = 'SMS_2018_1126_1625_1_BG_2018_1126_1621_1'
outputChann = ['568', '405', '488', '640', 'Retardance'] # first channel is the reference channel

z_load_range = [0,99]
y_load_range = [750,1300]
z_plot_range = [4,11]
y_plot_range = [790,810]
ImgSmPath = os.path.join(ProcessedPath, ImgDir, SmDir) # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'
OutputPath = os.path.join(ImgSmPath,'registration', 'raw')
shift_file_path = os.path.join(ImgSmPath, 'registration', 'registration_param_63X.json')
img_io = mManagerReader(ImgSmPath, OutputPath, outputChann)
img_io.posIdx = 0 # only read the first position
img_io.tIdx = 0 # only read the first time point
target_images = img_io.read_multi_chan_img_stack(z_range=z_load_range)
os.makedirs(img_io.ImgOutPath, exist_ok=True)
# for zIdx in range(z_range[0], z_range[1]):
#     img_io.zIdx = zIdx
#     target_image = [target_image[zIdx, 750:1300,750:1300] for target_image in target_images]
#     fig_name = 'img_pair_z%03d.png' % (zIdx)
#     imshow_pair(target_image, img_io, fig_name)
#     plt.close("all")
#
# for yIdx in range(y_range[0], y_range[1]):
#     img_io.yIdx = yIdx
#     target_image = [np.squeeze(target_image[:, yIdx, 750:1300]) for target_image in target_images]
#     fig_name = 'img_pair_y%03d.png' % (yIdx)
#     imshow_pair(target_image, img_io, fig_name)
#     plt.close("all")

target_images_cropped = [target_image[:, 750:1300, 750:1300] for target_image in target_images]
registration_params = channel_register(target_images_cropped, outputChann)

target_images_warped = translate_3D(target_images_cropped, outputChann, registration_params)

OutputPath = os.path.join(ImgSmPath,'registration', 'processed')
img_io.ImgOutPath = OutputPath
os.makedirs(img_io.ImgOutPath, exist_ok=True)

# for zIdx in range(z_plot_range[0], z_plot_range[1]):
#     img_io.zIdx = zIdx
#     target_image_warped = [target_image[zIdx-z_load_range[0], :, :] for target_image in target_images_warped]
#     fig_name = 'img_pair_z%03d.png' % (zIdx)
#     imshow_pair(target_image_warped, img_io, fig_name)
#     plt.close("all")
#
# for yIdx in range(y_plot_range[0], y_plot_range[1]):
#     img_io.yIdx = yIdx
#     target_image_warped = [np.squeeze(target_image[:, yIdx-y_load_range[0], :]) for target_image in target_images_warped]
#     fig_name = 'img_pair_y%03d.png' % (yIdx)
#     imshow_pair(target_image_warped, img_io, fig_name)
#     plt.close("all")


registration_params['size_z_um'] = img_io.size_z_um
with open(shift_file_path, 'w') as f:
    json.dump(registration_params, f, indent=4)