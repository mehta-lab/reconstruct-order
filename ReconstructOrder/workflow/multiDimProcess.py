"""
Process data collected over multiple positions, timepoints and z slices
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from ..utils.imgIO import parse_tiff_input, exportImg
from ..compute.reconstruct import ImgReconstructor
from ..utils.imgProcessing import ImgMin, imBitConvert, correct_flat_field
from ..utils.plotting import render_birefringence_imgs, plot_stokes, plot_pol_imgs, plot_Polacquisition_imgs
from ..utils.mManagerIO import mManagerReader, PolAcquReader


def create_metadata_object(data_path, config):
    """
    Reads PolAcquisition metadata, if possible. Otherwise, reads MicroManager metadata.
    TODO: move to imgIO?

    Parameters
    __________
    data_path : str
        Path to data directory
    config : obj
        ConfigReader object

    Returns
    _______
    obj
        Metadata object
    """

    try:
        img_obj = PolAcquReader(data_path, output_chan=config.processing.output_channels)
    except:
        img_obj = mManagerReader(data_path, output_chan=config.processing.output_channels)
    return img_obj


def read_metadata(config):
    """
    Reads the metadata for the sample and background data sets. Passes some
    of the parameters (e.g. swing, wavelength, back level, etc.) from the
    background metadata object into the sample metadata object
    TODO: move to imgIO?

    Parameters
    __________
    config : obj
        ConfigReader object

    Returns
    _______
    obj
        Metadata object
    """

    img_obj_list = []
    bg_obj_list = []

    # If one background is used for all samples, read only once
    if len(set(config.dataset.background)) <= 1:
        background_path = os.path.join(config.dataset.data_dir,config.dataset.background[0])
        bg_obj = create_metadata_object(background_path, config)
        bg_obj_list.append(bg_obj)
    else:
        for background in config.dataset.background:
            background_path = os.path.join(config.dataset.data_dir, background)
            bg_obj = create_metadata_object(background_path, config)
            bg_obj_list.append(bg_obj)

    for sample in config.dataset.samples:
        sample_path = os.path.join(config.dataset.data_dir, sample)
        img_obj = create_metadata_object(sample_path, config)
        img_obj_list.append(img_obj)

    if len(bg_obj_list) == 1:
        bg_obj_list = bg_obj_list*len(img_obj_list)

    for i in range(len(config.dataset.samples)):
        img_obj_list[i].bg = bg_obj_list[i].bg
        img_obj_list[i].swing = bg_obj_list[i].swing
        img_obj_list[i].wavelength = bg_obj_list[i].wavelength
        img_obj_list[i].blackLevel = bg_obj_list[i].blackLevel

    return img_obj_list, bg_obj_list


def parse_bg_options(img_obj_list, config):
    """
    Parse background correction options and make output directories

    Parameters
    __________
    img_obj_list: list
        List of img_obj objects
    config : obj
        ConfigReader object

    """

    for i in range(len(config.dataset.samples)):
        bgCorrect = config.processing.background_correction
        data_dir = config.dataset.data_dir
        processed_dir = os.path.join(config.dataset.processed_dir, os.path.basename(data_dir))
        sample = config.dataset.samples[i]
        background = config.dataset.background[i]

        if bgCorrect == 'None':
            print('No background correction is performed...')
            img_obj_list[i].bg_method = 'Global'
            img_obj_list[i].bg_correct = False
            OutputPath = os.path.join(processed_dir, sample)

        elif bgCorrect == 'Input':
            print('Background correction mode set as "Input". Use user input background directory')
            OutputPath = os.path.join(processed_dir, sample + '_' + background)
            img_obj_list[i].bg_method = 'Global'
            img_obj_list[i].bg_correct = True

        elif bgCorrect == 'Local_filter':
            print('Background correction mode set as "Local_filter". Additional background correction using local '
                  'background estimated from sample images will be performed')
            OutputPath = os.path.join(processed_dir, sample + '_' + sample)
            img_obj_list[i].bg_method = 'Local_filter'
            img_obj_list[i].bg_correct = True

        elif bgCorrect == 'Local_defocus':
            raise RuntimeError('Local_defocus is not longer supported')

        elif bgCorrect == 'Auto':
            if hasattr(img_obj_list[i], 'bg'):
                if img_obj_list[i].bg == 'No Background':
                    print('No background correction is performed for background measurements...')
                    OutputPath = os.path.join(processed_dir, sample + '_' + sample)
                    img_obj_list[i].bg_method = 'Global'
                    img_obj_list[i].bg_correct = False

                else:
                    print('Background info found in metadata. Use background specified in metadata')
                    background = img_obj_list[i].bg
                    OutputPath = os.path.join(processed_dir, sample + '_' + background)
                    img_obj_list[i].bg_method = 'Global'
                    img_obj_list[i].bg_correct = True

            else:
                print('Background not specified in metadata. Use user input background directory')
                OutputPath = os.path.join(processed_dir, sample + '_' + background)
                img_obj_list[i].bg_method = 'Global'
                img_obj_list[i].bg_correct = True

        img_obj_list[i].ImgOutPath = OutputPath
        os.makedirs(OutputPath, exist_ok=True)  # create folder for processed images
    return img_obj_list


def process_background(img_io, img_io_bg, config):
    """
    Read backgorund images, initiate ImgReconstructor to compute background stokes parameters

    """
    ImgRawBg = parse_tiff_input(img_io_bg)[0]  # 0 for z-index
    circularity = config.processing.circularity
    azimuth_offset = config.processing.azimuth_offset
    n_slice_local_bg = config.processing.n_slice_local_bg
    if n_slice_local_bg == 'all':
        n_slice_local_bg = len(img_io.ZList)
    img_reconstructor = ImgReconstructor(ImgRawBg.shape,
                                         bg_method=img_io.bg_method,
                                         n_slice_local_bg=n_slice_local_bg,
                                         swing=img_io.swing,
                                         wavelength=img_io.wavelength,
                                         azimuth_offset=azimuth_offset,
                                         circularity=circularity)
    if img_io.bg_correct:
        stokes_param_bg = img_reconstructor.compute_stokes(ImgRawBg)
        stokes_param_bg_tm = img_reconstructor.stokes_transform(stokes_param_bg)
        # print('denoising the background...')
        # img_stokes_bg = [denoise_tv_chambolle(img, weight=10**6) for img in img_stokes_bg]
        # img_stokes_bg = [cv2.GaussianBlur(img, (5, 5), 0) for img in img_stokes_bg]
        # img_stokes_bg = [cv2.medianBlur(img, 5) for img in img_stokes_bg]
    else:
        stokes_param_bg_tm = None

    img_reconstructor.stokes_param_bg_tm = stokes_param_bg_tm
    return img_io, img_reconstructor


def loopPos(img_io, config, img_reconstructor=None):
    """
    Loop through each position in the sample metadata, check if it is on the user input
    position list; make separate folder for each position if separate_pos == True
    """
    separate_pos = config.processing.separate_positions
    for posIdx, pos_name in enumerate(img_io.pos_list):
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
    Loop through each time point supplied in the config, call loopZSm or loopZBg
    depending on the looZ mode

    Parameters
    ----------
    img_io: object
        mManagerReader object that holds the image parameters
    config: object
        ConfigReader object that holds the user input config parameters
    img_reconstructor: object
        ImgReconstructor object for image reconstruction
    -------

    """
    for tIdx in img_io.TimeList:
        img_io.tIdx = tIdx
        if img_io.loopZ == 'reconstruct':
            img_io = loopZSm(img_io, config, img_reconstructor)

        elif img_io.loopZ == 'flat_field':
            img_io = loopZBg(img_io)
    return img_io


def loopZSm(img_io, config, img_reconstructor=None):
    """
    Loop through each z supplied in the config; computes and export only images in the
    supplied output channels (stokes, birefringence, background corrected raw pol images);

    Parameters
    ----------
    img_io: object
        mManagerReader object that holds the image parameters
    config: object
        ConfigReader object that holds the user input config parameters
    img_reconstructor: object
        ImgReconstructor object for image reconstruction
    -------
    """

    t_idx = img_io.tIdx
    pos_idx = img_io.posIdx
    z_list = img_io.ZList
    n_slice_local_bg = img_reconstructor.n_slice_local_bg
    norm = config.plotting.normalize_color_images
    save_fig = config.plotting.save_birefringence_fig
    save_stokes_fig = config.plotting.save_stokes_fig
    save_pol_fig = config.plotting.save_polarization_fig
    save_mm_fig = config.plotting.save_micromanager_fig
    pol_names = ['Pol_State_0', 'Pol_State_1', 'Pol_State_2', 'Pol_State_3', 'Pol_State_4']
    stokes_names = ['Stokes_0', 'Stokes_1', 'Stokes_2', 'Stokes_3']
    stokes_names_sm = [x + '_sm' for x in stokes_names]
    birefring_names = ['Brightfield_computed', 'Retardance', 'Orientation', 'Transmission', 'Polarization',
                       'Retardance+Orientation', 'Polarization+Orientation', 'Brightfield+Retardance+Orientation',
                       'Brightfield_computed+Retardance+Orientation',
                       'Retardance+Fluorescence', 'Retardance+Fluorescence_all']
    fluor_names = ['405', '488', '568', '640']
    save_stokes = any(chan in stokes_names + stokes_names_sm
               for chan in img_io.chNamesOut) or any([save_stokes_fig, save_mm_fig])
    save_birefring = any(chan in birefring_names
               for chan in img_io.chNamesOut) or save_fig
    save_BF = 'Brightfield' in img_io.chNamesOut
    save_pol = any(chan in pol_names for chan in img_io.chNamesOut) or save_pol_fig
    save_fluor = any(chan in fluor_names for chan in img_io.chNamesOut)

    for z_stack_idx in range(0, len(z_list), n_slice_local_bg):
        stokes_param_sm_stack = [[] for i in range(len(stokes_names))]
        fluor_list = []
        for z_list_idx in range(z_stack_idx, z_stack_idx + n_slice_local_bg):
            z_idx = z_list[z_list_idx]
            print('Processing position %03d, time %03d, z %03d ...' % (pos_idx, t_idx, z_idx))
            plt.close("all")  # close all the figures from the last run
            img_io.zIdx = z_idx
            ImgRawSm, ImgProcSm, ImgFluor, ImgBF = parse_tiff_input(img_io)
            ImgFluor = correct_flat_field(img_io, ImgFluor)
            fluor_list.append(ImgFluor)
            img_dict = {}
            if save_stokes or save_birefring:
                stokes_param_sm = img_reconstructor.compute_stokes(ImgRawSm)
                for stack, img in zip(stokes_param_sm_stack, stokes_param_sm):
                    stack.append(img)
                # retard = removeBubbles(retard)     # remove bright speckles in mounted brain slice images
            if save_BF and isinstance(ImgBF, np.ndarray):
                ImgBF = ImgBF[0, :, :] / img_reconstructor.stokes_param_bg_tm[0]  # flat-field correction
                ImgBF = imBitConvert(ImgBF*config.plotting.transmission_scaling, bit=16, norm=False)
                img_dict.update({'Brightfield': ImgBF})

            if isinstance(ImgProcSm, np.ndarray):
                retardMMSm = ImgProcSm[0, :, :]
                azimuthMMSm = ImgProcSm[1, :, :]
                if save_mm_fig:
                    imgs_mm_py = [retardMMSm, azimuthMMSm, retard, azimuth]
                    plot_Polacquisition_imgs(img_io, imgs_mm_py)

            if save_pol:
                imgs_pol = []
                for i in range(ImgRawSm.shape[0]):
                    imgs_pol += [ImgRawSm[i, ...] / img_reconstructor.img_raw_bg[i, ...]]
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
            stokes_param_sm_stack_tm = img_reconstructor.stokes_transform(stokes_param_sm_stack)
            if not img_io.bg_correct == 'None':
                stokes_param_sm_stack_tm = img_reconstructor.correct_background(stokes_param_sm_stack_tm)
            birfring_stacks = \
                img_reconstructor.reconstruct_birefringence(stokes_param_sm_stack_tm)
            img_dict = {}
            for z_idx in range(z_stack_idx, z_stack_idx + n_slice_local_bg):
                plt.close("all")  # close all the figures from the last run
                img_io.zIdx = z_list[z_idx]
                z_sub_idx = z_idx - z_stack_idx
                [s0, retard, azimuth, polarization, s1, s2, s3] = [stack[..., z_sub_idx] for stack in birfring_stacks]
                ImgFluor = fluor_list[z_sub_idx]
                if save_birefring:
                    imgs = [s0, retard, azimuth, polarization, ImgFluor]
                    img_io, img_dict = render_birefringence_imgs(img_io, imgs, config, spacing=20, vectorScl=2, zoomin=False,
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

def compute_flat_field(img_io, config):
    """
    Compute illumination function of fluorescence channels
    for flat-field correction

    Parameters
    ----------
    img_io: object
        mManagerReader object that holds the image parameters
    config: object
        ConfigReader object that holds the user input config parameters

    Returns
    -------
    img_io: object
        mManagerReader object that holds the image parameters with
        illumination function saved in img_io.img_fluor_bg

    """
    print('Calculating illumination function for flatfield correction...')
    ff_method = config.processing.ff_method
    img_io.ff_method = ff_method
    img_io.loopZ = 'flat_field'
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

def loopZBg(img_io):
    """
    Loop through each z in the sample metadata; computes the illumination function
    of fluorescence channels using image opening or looking for empty images,
    currently only process the first Z for speed
    Parameters
    ----------
    img_io: object
        mManagerReader object that holds the image parameters

    Returns
    -------
    img_io: object
        mManagerReader object that holds the image parameters

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

