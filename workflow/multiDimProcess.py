import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import cv2
import time
import copy
from utils.imgIO import parse_tiff_input, exportImg, GetSubDirName, FindDirContainPos
from compute.reconstruct import ImgReconstructor
from utils.imgProcessing import ImgMin
from utils.plotting import plot_birefringence, plot_stokes, plot_pol_imgs, plot_Polacquisition_imgs
from utils.mManagerIO import mManagerReader, PolAcquReader
from utils.imgProcessing import ImgLimit, imBitConvert
from skimage.restoration import denoise_tv_chambolle

def create_metadata_object(config, RawDataPath, ImgDir, SmDir, BgDir):
    """
    Create metadata_object for sample and background images.
    Pass PolAcquistion specific paramters from background to sample object

    """
    outputChann = config.processing.output_channels
    ImgSmPath = os.path.join(RawDataPath, ImgDir, SmDir)  # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'
    ImgSmPath = FindDirContainPos(ImgSmPath)
    ImgBgPath = os.path.join(RawDataPath, ImgDir, BgDir)  # Background image folder path, of form 'BG_yyyy_mmdd_hhmm_X'
    try:
        img_io = PolAcquReader(ImgSmPath, output_chan=outputChann)
    except:
        img_io = mManagerReader(ImgSmPath, output_chan=outputChann)
    img_io_bg = PolAcquReader(ImgBgPath)
    img_io.bg = img_io_bg.bg
    img_io.swing = img_io_bg.swing
    img_io.wavelength = img_io_bg.wavelength
    img_io.blackLevel = img_io_bg.blackLevel
    return img_io, img_io_bg

def parse_bg_options(img_io, img_io_bg, config, RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir):
    """
    Parse background correction options; construct output path

    """
    img_io.bg_method = 'Global'
    bgCorrect = config.processing.background_correction

    if bgCorrect == 'None':
        print('No background correction is performed...')
        img_io.bg_correct = False
        OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir)

    elif bgCorrect == 'Input':
        print('Background correction mode set as "Input". Use user input background directory')
        OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + BgDir)
        img_io.bg_correct = True
    elif bgCorrect == 'Local_filter':
        print('Background correction mode set as "Local_filter". Additional background correction using local '
              'background estimated from sample images will be performed')
        OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + SmDir)
        img_io.bg_method = 'Local_filter'
        img_io.bg_correct = True
    elif bgCorrect == 'Local_defocus':
        print('Background correction mode set as "Local_defocus". Use images from' + BgDir +
              'at the same position as background')
        img_bg_path = os.path.join(RawDataPath, ImgDir,
                                   BgDir)
        OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + BgDir)
        img_io.bg_method = 'Local_defocus'
        img_io.bg_correct = True
        img_io_bg_local = mManagerReader(img_bg_path, OutputPath)
        img_io_bg_local.blackLevel = img_io_bg.blackLevel
        img_io.bg_local = img_io_bg_local

    elif bgCorrect == 'Auto':
        if hasattr(img_io, 'bg'):
            if img_io.bg == 'No Background':
                BgDir = SmDir
                img_io.bg_correct = False
                print('No background correction is performed for background measurements...')
            else:
                print('Background info found in metadata. Use background specified in metadata')
                BgDir = img_io.bg
                OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + BgDir)
                img_io.bg_correct = True
        else:
            print('Background not specified in metadata. Use user input background directory')
            OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + BgDir)
            img_io.bg_correct = True
    img_io.ImgOutPath = OutputPath
    os.makedirs(OutputPath, exist_ok=True)  # create folder for processed images

    return img_io, img_io_bg

# similar pattern to "create_metadata_object":
#   - we call 'process_background' but nothing it does is specific to the background.
#   - more accurately, this function assigns attributes to the input "img_io" and to the output "img_reconstructor"
#   - img_io is an input only for the attribute "bg_method", and to have bg-stokes appended.
def process_background(img_io, img_io_bg, config):
    """
    Read backgorund images, initiate ImgReconstructor to compute background stokes parameters

    """
    ImgRawBg, ImgProcBg, ImgFluor, ImgBF = parse_tiff_input(img_io_bg)  # 0 for z-index
    img_io.img_raw_bg = ImgRawBg
    circularity = config.processing.circularity
    azimuth_offset = config.processing.azimuth_offset
    n_slice_local_bg = config.processing.n_slice_local_bg
    img_reconstructor = ImgReconstructor(ImgRawBg,
                                         bg_method=img_io.bg_method,
                                         n_slice_local_bg=n_slice_local_bg,
                                         swing=img_io_bg.swing,
                                         wavelength=img_io_bg.wavelength,
                                         output_path=img_io_bg.ImgOutPath,
                                         azimuth_offset=azimuth_offset,
                                         circularity=circularity)
    if img_io.bg_correct:
        stokes_param_bg = img_reconstructor.compute_stokes(ImgRawBg)
        # print('denoising the background...')
        # img_stokes_bg = [denoise_tv_chambolle(img, weight=10**6) for img in img_stokes_bg]
        # img_stokes_bg = [cv2.GaussianBlur(img, (5, 5), 0) for img in img_stokes_bg]
        # img_stokes_bg = [cv2.medianBlur(img, 5) for img in img_stokes_bg]
    else:
        stokes_param_bg = None
    img_reconstructor.stokes_param_bg = stokes_param_bg
    return img_io, img_reconstructor

def compute_flat_field(img_io, config):
    """
    Compute illumination function of fluorescence channels
    for flat-field correction

    """
    print('Calculating illumination function for flatfield correction...')
    ff_method = config.processing.ff_method
    img_io.ff_method = ff_method
    img_io.ImgFluorMin = np.full((4, img_io.height, img_io.width), np.inf)  # set initial min array to to be Inf
    img_io.ImgFluorSum = np.zeros(
        (4, img_io.height, img_io.width))  # set the default background to be Ones (uniform field)
    img_io.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (100, 100))  # kernel for image opening operation, 100-200 is usually good
    img_io.loopZ = 'background'
    img_io = loopPos(img_io, config)
    if ff_method == 'open':
        img_fluor_bg = img_io.ImgFluorSum
    elif ff_method == 'empty':
        img_fluor_bg = img_io.ImgFluorMin
    for channel in range(img_fluor_bg.shape[0]):
        img_fluor_bg[channel] = img_fluor_bg[channel] - min(np.nanmin(img_fluor_bg[channel]), 0) + 1 #add 1 to avoid 0
        img_fluor_bg[channel] /= np.mean(img_fluor_bg[channel])  # normalize the background to have mean = 1
    img_io.img_fluor_bg = img_fluor_bg
    return img_io

def correct_flat_field(img_io, ImgFluor):
    """ flat-field correction for fluorescence channels """

    for i in range(ImgFluor.shape[0]):
        if np.any(ImgFluor[i, :, :]):  # if the flour channel exists
            ImgFluor[i, :, :] = ImgFluor[i, :, :] / img_io.img_fluor_bg[i, :, :]
    return ImgFluor

def loopPos(img_io, config, img_reconstructor=None):
    """
    Loop through each position in the sample metadata, check if it is on the user input
    position list; make separate folder for each position if separate_pos == True
    """
    separate_pos = config.processing.separate_positions

    for posIdx, pos_name in enumerate(img_io.pos_names):
        if pos_name in img_io.PosList or img_io.PosList == 'all':
            plt.close("all")  # close all the figures from the last run            
            img_io.img_in_pos_path = os.path.join(img_io.ImgSmPath, pos_name)
            img_io.pos_name = pos_name
            if separate_pos:
                img_io.img_out_pos_path = os.path.join(img_io.ImgOutPath, pos_name)
                os.makedirs(img_io.img_out_pos_path, exist_ok=True)  # create folder for processed images
            else:
                img_io.img_out_pos_path = img_io.ImgOutPath
    
            if img_io.bg_method == 'Local_defocus':
                img_io_bg = img_io.bg_local
                img_io_bg.pos_name = os.path.join(img_io_bg.ImgSmPath, pos_name)
                img_io_bg.posIdx = posIdx
            img_io.posIdx = posIdx
    
            img_io = loopT(img_io, config, img_reconstructor)
    return img_io


def loopT(img_io, config, img_reconstructor=None):
    """
    Loop through each time point in the sample metadata, call loopZSm or loopZBg
    depending on the looZ mode
    TODO: only process user input list of time points
    """

    for tIdx in range(0, img_io.nTime):
        img_io.tIdx = tIdx
        if img_io.loopZ == 'sample':
            # if img_io.bg_method == 'Local_defocus':
            #     img_io_bg = img_io.bg_local
            #     print('compute defocused backgorund at pos{} ...'.format(img_io_bg.posIdx))
            #     img_io_bg.tIdx = tIdx
            #     img_io_bg.zIdx = 0
            #     ImgRawBg, ImgProcBg, ImgFluor, ImgBF = parse_tiff_input(img_io_bg)  # 0 for z-index
            #     img_stokes_bg = img_reconstructor.compute_stokes(ImgRawBg)
            #     img_io.param_bg = img_stokes_bg
            img_io = loopZSm(img_io, config, img_reconstructor)

        else:
            img_io = loopZBg(img_io)
    return img_io


def loopZSm(img_io, config, img_reconstructor=None):
    """
    Loop through each z in the sample metadata; computes and export only images in the
    supplied output channels (stokes, birefringence, background corrected raw pol images);
    save figures of images if save_fig is True
    TODO: only process user input list of z
    """

    tIdx = img_io.tIdx
    posIdx = img_io.posIdx
    norm = config.plotting.normalize_color_images
    save_fig = config.plotting.save_birefringence_fig
    save_stokes_fig = config.plotting.save_stokes_fig
    save_pol_fig = config.plotting.save_polarization_fig
    save_mm_fig = config.plotting.save_micromanager_fig
    pol_names = ['Pol_State_0', 'Pol_State_1', 'Pol_State_2', 'Pol_State_3', 'Pol_State_4']
    stokes_names = ['Stokes_0', 'Stokes_1', 'Stokes_2', 'Stokes_3']
    stokes_names_sm = [x + '_sm' for x in stokes_names]
    birefring_names = ['Brightfield', 'Retardance', 'Orientation', 'Transmission', 'Polarization',
                       'Retardance+Orientation', 'Polarization+Orientation', 'Transmission+Retardance+Orientation',
                       'Retardance+Fluorescence', 'Retardance+Fluorescence_all']
    fluor_names = ['405', '488', '568', '640']
    save_stokes = any(chan in stokes_names + stokes_names_sm
               for chan in img_io.chNamesOut) or any([save_stokes_fig, save_mm_fig])
    save_birefring = any(chan in birefring_names
               for chan in img_io.chNamesOut) or save_fig
    save_pol = any(chan in pol_names for chan in img_io.chNamesOut) or save_pol_fig
    save_fluor = any(chan in fluor_names for chan in img_io.chNamesOut)
    n_slice_local_bg = img_reconstructor.n_slice_local_bg
    for z_stack_idx in range(0, img_io.nZ, n_slice_local_bg):
        stokes_param_sm_stack = [[] for i in range(len(stokes_names))]
        fluor_list = []
        for zIdx in range(z_stack_idx, z_stack_idx + n_slice_local_bg):
            print('Processing position %03d, time %03d, z %03d ...' % (posIdx, tIdx, zIdx))
            plt.close("all")  # close all the figures from the last run
            img_io.zIdx = zIdx
            ImgRawSm, ImgProcSm, ImgFluor, ImgBF = parse_tiff_input(img_io)
            ImgFluor = correct_flat_field(img_io, ImgFluor)
            fluor_list.append(ImgFluor)
            img_dict = {}
            if save_stokes or save_birefring:
                stokes_param_sm = img_reconstructor.compute_stokes(ImgRawSm)
                for stack, img in zip(stokes_param_sm_stack, stokes_param_sm):
                    stack.append(img)
                # retard = removeBubbles(retard)     # remove bright speckles in mounted brain slice images
            #TODO: Make experimental and computed BF separate channels
            if isinstance(ImgBF, np.ndarray):
                ImgBF = ImgBF[0, :, :] / img_reconstructor.stokes_param_bg[0]  # flat-field correction
                img_dict.update({'Brightfield': ImgBF})
            if save_pol:
                imgs_pol = []
                for i in range(ImgRawSm.shape[0]):
                    imgs_pol += [ImgRawSm[i, ...] / img_io.img_raw_bg[i, ...]]
                if save_pol_fig:
                    plot_pol_imgs(img_io, imgs_pol, pol_names)
                imgs_pol = [imBitConvert(img * 10 ** 4, bit=16) for img in imgs_pol]
                img_dict.update(dict(zip(pol_names, imgs_pol)))
            if save_fluor:
                ImgFluor = imBitConvert(ImgFluor, bit=16, norm=False)
                img_dict.update(dict(zip(fluor_names, [ImgFluor[chan, :, :] for chan in range(ImgFluor.shape[0])])))
            exportImg(img_io, img_dict)


        if save_stokes or save_birefring:
            stokes_param_sm_stack = [np.stack(stack, axis=-1) for stack in stokes_param_sm_stack]
            stokes_param_sm_stack_tm = img_reconstructor.correct_background(stokes_param_sm_stack)
            birfring_stacks = \
                img_reconstructor.reconstruct_birefringence(stokes_param_sm_stack_tm)
            img_dict = {}
            for zIdx in range(z_stack_idx, z_stack_idx + n_slice_local_bg):
                plt.close("all")  # close all the figures from the last run
                img_io.zIdx = zIdx
                z_sub_idx = zIdx - z_stack_idx
                [s0, retard, azimuth, polarization, s1, s2, s3] = [stack[..., z_sub_idx] for stack in birfring_stacks]
                ImgFluor = fluor_list[z_sub_idx]
                if save_birefring:
                    imgs = [s0, retard, azimuth, polarization, ImgFluor]
                    img_io, img_dict = plot_birefringence(img_io, imgs, spacing=20, vectorScl=2, zoomin=False,
                                                          dpi=200,
                                                          norm=norm, plot=save_fig)
                if save_stokes:
                    img_stokes = [s0, s1, s2, s3]
                    img_stokes_sm = [stack[..., z_sub_idx] for stack in stokes_param_sm_stack]
                    if save_stokes_fig:
                        plot_stokes(img_io, img_stokes, img_stokes_sm)
                    img_stokes = [x.astype(np.float32, copy=False) for x in img_stokes]
                    img_stokes_sm = [x.astype(np.float32, copy=False) for x in img_stokes_sm]
                    img_stokes_dict = dict(zip(stokes_names, img_stokes))
                    img_stokes_sm_dict = dict(zip(stokes_names_sm, img_stokes_sm))
                    img_dict.update(img_stokes_dict)
                    img_dict.update(img_stokes_sm_dict)
                exportImg(img_io, img_dict)
    return img_io

def loopZBg(img_io):
    """
    Loop through each z in the sample metadata; computes the illumination function
    of fluorescence channels using image opening or looking for empty images,
    currently only process the first Z for speed

    """
    for zIdx in range(0, 1):  # only use the first z
        img_io.zIdx = zIdx
        ImgRawSm, ImgProcSm, ImgFluor, ImgBF = parse_tiff_input(img_io)
        for i in range(ImgFluor.shape[0]):
            if np.any(ImgFluor[i, :, :]):  # if the flour channel exists
                if img_io.ff_method == 'open':
                    img_io.ImgFluorSum[i, :, :] += cv2.morphologyEx(ImgFluor[i, :, :], cv2.MORPH_OPEN, img_io.kernel,
                                                                    borderType=cv2.BORDER_REPLICATE)
                elif img_io.ff_method == 'empty':
                    img_io.ImgFluorMin[i, :, :] = ImgMin(ImgFluor[i, :, :], img_io.ImgFluorMin[i, :, :])
    return img_io

