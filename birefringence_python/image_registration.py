from skimage import io, exposure, util, filters
from skimage.transform import SimilarityTransform, warp
from skimage.feature import register_translation
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.mManagerIO import mManagerReader
from utils.plotting import CompositeImg, plot_sub_images


def load_img(img_io, z_range=None):
    if not os.path.exists(img_io.ImgSmPath):
        raise FileNotFoundError(
            "image file doesn't exist at:", img_io.ImgSmPath
        )
    os.makedirs(img_io.ImgOutPath, exist_ok=True)
    img_io.posIdx = 0 # only read the first position
    img_io.tIdx = 0 # only read the first time point
    img_chann = [] # list of 2D or 3D images from different channels
    for chanIdx in range(img_io.nChannOut):
        img_stack = []
        img_io.chanIdx = chanIdx
        for zIdx in range(z_range[0], z_range[1]):
            img_io.zIdx = zIdx
            img = img_io.readmManager()
            img_stack += [img]
        img_stack = np.stack(img_stack) # follow zyx order
        img_stack = np.squeeze(img_stack)
        img_chann += [img_stack]
    return img_chann

def imshow_pair(images, img_io):
    chann_names = img_io.chNamesOut
    image_pairs = []
    titles = []
    for image, chann_name in zip(images, chann_names):
        image_pair = [images[0], image]
        title = [chann_names[0] + chann_name]
        image_pair_rgb = CompositeImg(image_pair, norm=True)
        image_pairs += [image_pair_rgb]
        titles += title
    fig_name = 'img_pair_z%03d.png'%(zIdx)
    plot_sub_images(image_pairs, titles, img_io.ImgOutPath, fig_name, colorbar=False)
def cross_correlation(target_images):
    """
    Returns a list of shift vectors, transform objects, and the unregistered images. Transform objects can be used to
    register all images to the first image in the folder.

    @param filepath: The file path of the folder containing target images. String.
    @return: shifts - list of vectors (y, x), transforms - list of transformation objects, target_images - list of images
    """


    # Applying sobel edge filter to brightfield and phase images.
    # target_images[5] = filters.sobel(target_images[5])
    # target_images[6] = filters.sobel(target_images[6])

    # Finding shift vectors that register first channel to all others. Subpixel registration is achieved by
    # upsampling by a factor of 4.
    shifts = []
    errors = []
    phases = []
    for image in target_images[1:]:
        shift, error, phase = register_translation(target_images[0], image, 4)
        shifts.append(shift)
        errors.append(error)
        phases.append(phase)

    # flips shift array so that shifts are in (y, x) format
    # shifts = [np.flipud(shift) for shift in shifts]

    # transformation objects, 3x3 matricies
    transforms = [SimilarityTransform(translation=shift) for shift in shifts]

    return shifts, transforms


def warp_batch(images, transforms):
    """
    Warps images using the provided transformation objects, and displays the overlaid images.

    @param images: list of images.
    @param transformations: list of transformation objects.
    @return: reg_img - list of registered images.
    """

    # applying transformations to all images except for the first one (first channel is source img).
    registered_images = []
    for transform, image in zip(transforms, images[1:]):
        registered_images.append(warp(image, transform.inverse))

    # Need to rescale intensity to see effect of overlay.
    registered_images = [exposure.rescale_intensity(img, out_range=(0, 65535)) for img in registered_images]

    # Display overlays.
    for i in registered_images:
        plt.figure()
        overlay = target_images[0] + i
        plt.imshow(overlay, cmap='gray')

    return registered_images

RawDataPath = r'D:/Box Sync/Data'
ProcessedPath = r'D:/Box Sync/Processed/'

ImgDir = '2018_11_26_Argolight_channel_registration_63X_confocal'
SmDir = 'SMS_2018_1126_1625_1_BG_2018_1126_1621_1'

outputChann = ['405', '488', '568', '640', 'Retardance']
z_range = [8,10]
ImgSmPath = os.path.join(ProcessedPath, ImgDir, SmDir) # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'

OutputPath = os.path.join(ImgSmPath,'registration', 'raw')
img_io_sm = mManagerReader(ImgSmPath, OutputPath, outputChann)
target_images = load_img(img_io_sm, z_range=z_range)
for zIdx in range(z_range[0], z_range[1]):
    img_io_sm.zIdx = zIdx
    target_image = [target_image[zIdx,:,:] for target_image in target_images]
    imshow_pair(target_image, img_io_sm)


# shifts, transforms = cross_correlation(target_images)