"""
Process data collected over multiple positions, timepoints and z slices
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from ..utils.imgIO import parse_tiff_input, exportImg
from ..compute.reconstruct import ImgReconstructor
from ..compute.reconstruct_phase import phase_reconstructor
from ..utils.imgProcessing import ImgMin, imBitConvert, correct_flat_field, mean_pooling_2d_stack
from ..utils.plotting import render_birefringence_imgs, plot_stokes, plot_pol_imgs, plot_Polacquisition_imgs
from ..utils.mManagerIO import mManagerReader, PolAcquReader

from ..datastructures import StokesData
from ..utils.ConfigReader import ConfigReader
from typing import Union

matplotlib.use('Agg')


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
        img_obj = PolAcquReader(data_path,
                                output_chan=config.processing.output_channels,
                                binning=config.processing.binning
                                )
    except:
        img_obj = mManagerReader(data_path,
                                 output_chan=config.processing.output_channels,
                                 binning=config.processing.binning)
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

        elif bgCorrect in ['Local_filter', 'Local_fit']:
            print('Background correction mode set as "{}". Additional background correction using local '
                  'background estimated from sample images will be performed'.format(bgCorrect))
            OutputPath = os.path.join(
                processed_dir, ''.join([sample, '_', sample]))
            img_obj_list[i].bg_method = bgCorrect
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

        else:
            raise AttributeError("Must define a background correction method")

        if not config.processing.binning == 1:
            OutputPath = ''.join([OutputPath, '_binning_', str(config.processing.binning)])

        img_obj_list[i].ImgOutPath = OutputPath
        os.makedirs(OutputPath, exist_ok=True)  # create folder for processed images
    return img_obj_list


def process_background(img_io, img_io_bg, config):
    """
    Read background images, initiate ImgReconstructor to compute background stokes parameters

    """
    ImgRawBg         = parse_tiff_input(img_io_bg, config.dataset.ROI)[0]  # 0 for z-index
    circularity      = config.processing.circularity
    azimuth_offset   = config.processing.azimuth_offset
    n_slice_local_bg = config.processing.n_slice_local_bg
    local_fit_order  = config.processing.local_fit_order
    binning          = config.processing.binning
    use_gpu          = config.processing.use_gpu
    gpu_id           = config.processing.gpu_id
    
    if n_slice_local_bg == 'all':
        n_slice_local_bg = len(img_io.ZList)

    img_reconstructor = ImgReconstructor(ImgRawBg.data.shape,
                                         bg_method        = img_io.bg_method,
                                         n_slice_local_bg = n_slice_local_bg,
                                         poly_fit_order   = local_fit_order,
                                         swing            = img_io.swing,
                                         wavelength       = img_io.wavelength,
                                         azimuth_offset   = azimuth_offset,
                                         circularity      = circularity,
                                         binning          = binning,
                                         use_gpu          = use_gpu,
                                         gpu_id           = gpu_id)

    if img_io.bg_correct:
        background_stokes = img_reconstructor.compute_stokes(ImgRawBg)
        background_stokes_normalized = img_reconstructor.stokes_normalization(background_stokes)
        # print('denoising the background...')
        # img_stokes_bg = [denoise_tv_chambolle(img, weight=10**6) for img in img_stokes_bg]
        # img_stokes_bg = [cv2.GaussianBlur(img, (5, 5), 0) for img in img_stokes_bg]
        # img_stokes_bg = [cv2.medianBlur(img, 5) for img in img_stokes_bg]
    else:
        background_stokes_normalized = None

    # img_reconstructor.stokes_param_bg_tm = stokes_param_bg_tm
    # return img_io, img_reconstructor
    return background_stokes_normalized, img_reconstructor

def phase_reconstructor_initializer(img_io: Union[mManagerReader, PolAcquReader],
                                    config: ConfigReader):
    
    '''
    
    draw parameters from metadata and config file to inialize phase reconstructor
    
    Parameters
    ----------
    
        img_io   : object
                   mManagerReader object that holds the image parameters
                
        config   : object
                   ConfigReader object that holds the user input config parameters
    
    Returns
    -------
        
        ph_recon : object
                   phase_reconstructor object that enables phase reconstruction
    
    '''
    
    lambda_illu  = img_io.wavelength*1e-3
    mag          = config.processing.magnification
    ps           = config.processing.pixel_size/mag
    psz          = img_io.size_z_um
    NA_obj       = config.processing.NA_objective
    NA_illu      = config.processing.NA_condenser
    focus_idx    = config.processing.focus_zidx
    n_objective_media      = config.processing.n_objective_media
    pad_z        = config.processing.pad_z
    phase_deconv = []
    N_defocus    = img_io.nZ

    
    if config.dataset.ROI is None:
        ROI = [0,0, img_io.height, img_io.width]
    else:
        ROI = config.dataset.ROI
        
    
    opt_mapping = {'Phase2D': '2D', 'Phase_semi3D': 'semi-3D', 'Phase3D': '3D'}
    phase_deconv = [opt_mapping[opt] for opt in img_io.chNamesOut if opt in opt_mapping.keys()]
    
    ph_recon = phase_reconstructor((ROI[2], ROI[3], N_defocus), lambda_illu, ps, psz, NA_obj, NA_illu, focus_idx = focus_idx, \
                                   n_objective_media=n_objective_media, phase_deconv=phase_deconv, pad_z=pad_z,\
                                   use_gpu=config.processing.use_gpu, gpu_id=config.processing.gpu_id)

    ph_recon.Phase_solver_para_setter(denoiser_2D    = config.processing.phase_denoiser_2D, \
                                      Tik_reg_abs_2D = config.processing.Tik_reg_abs_2D, Tik_reg_ph_2D = config.processing.Tik_reg_ph_2D, \
                                      TV_reg_abs_2D  = config.processing.TV_reg_abs_2D,  TV_reg_ph_2D  = config.processing.TV_reg_ph_2D,\
                                      rho_2D         = config.processing.rho_2D,         itr_2D        = config.processing.itr_2D, \
                                      denoiser_3D    = config.processing.phase_denoiser_3D, \
                                      Tik_reg_ph_3D  = config.processing.Tik_reg_ph_3D,  TV_reg_ph_3D  = config.processing.TV_reg_ph_3D, \
                                      rho_3D         = config.processing.rho_3D,         itr_3D        = config.processing.itr_3D, \
                                      verbose        = False,                            bg_filter     = True)
    print('Finish phase transfer function computing')
    
    return ph_recon



def compute_flat_field(img_io: Union[mManagerReader, PolAcquReader],
                       config: ConfigReader,
                       img_reconstructor: ImgReconstructor,
                       background_data):
    """
    Compute illumination function of fluorescence channels
    for flat-field correction

    Parameters
    ----------
    img_io: object
        mManagerReader object that holds the image parameters
    config: object
        ConfigReader object that holds the user input config parameters
    img_reconstructor: ImgReconstructor
        ImgReconstructor object for image reconstruction
    background_data: StokesData
        object of type StokesData
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

    img_io = loopPos(img_io, config, img_reconstructor, background_data)

    if ff_method == 'open':
        img_fluor_bg = img_io.ImgFluorSum
    elif ff_method == 'empty':
        img_fluor_bg = img_io.ImgFluorMin
    else:
        raise ValueError("ff_method must be 'open' or 'empty'")

    for channel in range(img_fluor_bg.shape[0]):
        img_fluor_bg[channel] = img_fluor_bg[channel] - min(np.nanmin(img_fluor_bg[channel]), 0) + 1 #add 1 to avoid 0
        img_fluor_bg[channel] /= np.mean(img_fluor_bg[channel])  # normalize the background to have mean = 1
    img_io.img_fluor_bg = img_fluor_bg

    return img_io




def loopPos(img_io: Union[mManagerReader, PolAcquReader],
            config: ConfigReader,
            img_reconstructor: ImgReconstructor,
            background_corrected_data=None,
            ph_recon=None):
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
        if ph_recon is None:
            img_io = loopT(img_io, config, img_reconstructor, background=background_corrected_data)
        else:
            img_io = loopT(img_io, config, img_reconstructor, background=background_corrected_data, ph_recon=ph_recon)

    return img_io


def loopT(img_io: Union[mManagerReader, PolAcquReader],
          config: ConfigReader,
          img_reconstructor: ImgReconstructor,
          background=None,
          ph_recon=None):
    """
    Loop through each time point supplied in the config, call loopZSm or loopZBg
    depending on the looZ mode

    Parameters
    ----------
    img_io: Union[mManagerReader, PolAcquReader]
        mManagerReader object that holds the image parameters
    config: ConfigReader
        ConfigReader object that holds the user input config parameters
    img_reconstructor: ImgReconstructor
        ImgReconstructor object for image reconstruction
    background: BackgroundData
        object of type BackgroundData
    ph_recon: object
        phase_reconstructor object that enables phase reconstruction
    -------

    """
    for tIdx in img_io.TimeList:
        img_io.tIdx = tIdx
        if img_io.loopZ == 'reconstruct':
            if ph_recon is None:
                img_io = loopZSm(img_io, config, img_reconstructor, background=background)
            else:
                img_io = loopZSm(img_io, config, img_reconstructor, background=background, ph_recon=ph_recon)

        elif img_io.loopZ == 'flat_field':
            img_io = loopZBg(img_io, config)
    return img_io


def loopZSm(img_io: Union[mManagerReader, PolAcquReader],
            config: ConfigReader,
            img_reconstructor: ImgReconstructor,
            background=None,
            ph_recon=None):
    """
    Loop through each z supplied in the config; computes and export only images in the
    supplied output channels (stokes, birefringence, background corrected raw pol images);

    Parameters
    ----------
    img_io: mManagerReader
        mManagerReader object that holds the image parameters
    config: ConfigReader
        ConfigReader object that holds the user input config parameters
    img_reconstructor: ImgReconstructor
        ImgReconstructor object for image reconstruction
    background: BackgroundData
        BackgroundData object containing normalized stokes images
    ph_recon: object
        phase_reconstructor object that enables phase reconstruction
    -------
    """

    t_idx            = img_io.tIdx
    pos_idx          = img_io.posIdx
    z_list           = img_io.ZList
    n_slice_local_bg = img_reconstructor.n_slice_local_bg
    binning          = config.processing.binning
    norm             = config.plotting.normalize_color_images
    save_fig         = config.plotting.save_birefringence_fig
    save_stokes_fig  = config.plotting.save_stokes_fig
    save_pol_fig     = config.plotting.save_polarization_fig
    save_mm_fig      = config.plotting.save_micromanager_fig
    

    # 1) define allowed config values
    pol_names       = ['Pol_State_0', 'Pol_State_1', 'Pol_State_2', 'Pol_State_3', 'Pol_State_4']
    stokes_names    = ['Stokes_0', 'Stokes_1', 'Stokes_2', 'Stokes_3']
    stokes_names_sm = [x + '_sm' for x in stokes_names]
    birefring_names = ['Brightfield_computed', 'Retardance', 'Orientation', 'Transmission', 'Polarization',
                       'Retardance+Orientation', 'Polarization+Orientation', 'Brightfield+Retardance+Orientation',
                       'Brightfield_computed+Retardance+Orientation',
                       'Retardance+Fluorescence', 'Retardance+Fluorescence_all']
    phase_names     = ['Phase2D', 'Phase_semi3D', 'Phase3D']
    fluor_names     = ['405', '488', '568', '640', 'ex561em700']

    # 2 )set flags based on names defined in 1)
    save_stokes    = any(chan in stokes_names + stokes_names_sm for chan in img_io.chNamesOut) \
                     or any([save_stokes_fig, save_mm_fig])
    save_phase     = any(chan in phase_names for chan in img_io.chNamesOut)
    save_birefring = any(chan in birefring_names for chan in img_io.chNamesOut) or save_fig or save_phase
    save_BF        = 'Brightfield' in img_io.chNamesOut
    save_pol       = any(chan in pol_names for chan in img_io.chNamesOut) or save_pol_fig
    save_fluor     = any(chan in fluor_names for chan in img_io.chNamesOut)


    for z_stack_idx in range(0, len(z_list), n_slice_local_bg):

        stokes_param_sm_stack = [[] for i in range(len(stokes_names))]
        fluor_list = []

        for z_list_idx in range(z_stack_idx, z_stack_idx + n_slice_local_bg):
            z_idx = z_list[z_list_idx]
            print('Processing position %03d, time %03d, z %03d ...' % (pos_idx, t_idx, z_idx))
            plt.close("all")  # close all the figures from the last run
            img_io.zIdx = z_idx

            # load raw intensity data
            ImgRawSm, ImgProcSm, ImgFluor, ImgBF = parse_tiff_input(img_io, config.dataset.ROI)

            if isinstance(ImgFluor, np.ndarray):
                ImgFluor = mean_pooling_2d_stack(ImgFluor, binning)
            if isinstance(ImgBF, np.ndarray):
                ImgBF = mean_pooling_2d_stack(ImgBF, binning)
            ImgFluor = correct_flat_field(img_io, ImgFluor)
            fluor_list.append(ImgFluor)

            img_dict = {}
            if save_stokes or save_birefring:
                # compute stokes
                stokes_param_sm = img_reconstructor.compute_stokes(ImgRawSm)
                for stack, img in zip(stokes_param_sm_stack, stokes_param_sm.data):
                    stack.append(img)
                # retard = removeBubbles(retard)     # remove bright speckles in mounted brain slice images

            if save_BF and isinstance(ImgBF, np.ndarray):
                ImgBF = ImgBF[0, :, :] / background.s0  # flat-field correction
                ImgBF = imBitConvert(ImgBF*config.plotting.transmission_scaling, bit=16, norm=False)
                img_dict.update({'Brightfield': ImgBF})

            if isinstance(ImgProcSm, np.ndarray):
                retardMMSm = ImgProcSm[0, :, :]
                azimuthMMSm = ImgProcSm[1, :, :]
                if save_mm_fig:
                    # todo: retard and azimuth are called before definitiion here
                    imgs_mm_py = [retardMMSm, azimuthMMSm, retard, azimuth]
                    plot_Polacquisition_imgs(img_io, imgs_mm_py)

            if save_pol:
                imgs_pol = []
                for i in range(ImgRawSm.shape[0]):
                    #todo: I don't know where "img_raw_bg" comes from.  where is it assigned?
                    imgs_pol += [ImgRawSm[i, ...] / img_reconstructor.img_raw_bg[i, ...]]
                if save_pol_fig:
                    plot_pol_imgs(img_io, imgs_pol, pol_names)
                imgs_pol = [imBitConvert(img * 10 ** 4, bit=16) for img in imgs_pol]
                img_dict.update(dict(zip(pol_names, imgs_pol)))

            if save_fluor:
                ImgFluor = imBitConvert(ImgFluor, bit=16, norm=False)
                img_dict.update(dict(zip(fluor_names, [ImgFluor[chan, :, :] for chan in range(ImgFluor.shape[0])])))
            exportImg(img_io, img_dict)

        # generate images?
        if save_stokes or save_birefring:

            # build StokesData
            stk_dat = StokesData()
            [stk_dat.s0,
             stk_dat.s1,
             stk_dat.s2,
             stk_dat.s3] = [np.stack(stack, axis=-1) for stack in stokes_param_sm_stack]

            norm_sample = img_reconstructor.stokes_normalization(stk_dat)
            if img_io.bg_correct:
                norm_sample = img_reconstructor.correct_background(norm_sample, background)

            physical_data = img_reconstructor.reconstruct_birefringence(norm_sample)
            
            print('Finish birefringence reconstruction')
            if save_phase:
                
        
                for deconv_dim in ph_recon.phase_deconv:
                    
                    if deconv_dim == '2D':
                        physical_data.absorption_2D, physical_data.phase_2D = ph_recon.Phase_recon_2D(norm_sample)
                        print('Finish 2D phase reconstruction')
                    if deconv_dim == 'semi-3D':
                        physical_data.absorption_semi3D, physical_data.phase_semi3D = ph_recon.Phase_recon_semi_3D(norm_sample)
                        print('Finish semi3D phase reconstruction')
                    if deconv_dim == '3D':
                        physical_data.phase_3D = ph_recon.Phase_recon_3D(norm_sample)
                        print('Finish 3D phase reconstruction')
                    
                


            img_dict = {}
            for z_idx in range(z_stack_idx, z_stack_idx + n_slice_local_bg):
                plt.close("all")  # close all the figures from the last run
                img_io.zIdx = z_list[z_idx]
                z_sub_idx = z_idx - z_stack_idx

                # extract the relevant z slice out of the data
                s0           = physical_data.I_trans[..., z_sub_idx]
                retard       = physical_data.retard[..., z_sub_idx]
                azimuth      = physical_data.azimuth[..., z_sub_idx]
                polarization = physical_data.polarization[..., z_sub_idx]
                s1           = (norm_sample.s1_norm * norm_sample.s3)[..., z_sub_idx]
                s2           = (norm_sample.s2_norm * norm_sample.s3)[..., z_sub_idx]
                s3           = (norm_sample.s3)[..., z_sub_idx]


                ImgFluor = fluor_list[z_sub_idx]

                if save_birefring:

                    imgs = [s0, retard, azimuth, polarization, ImgFluor]

                    img_io, img_dict = render_birefringence_imgs(img_io, imgs, config, spacing=20, vectorScl=8, zoomin=False,
                                                                 dpi=200,
                                                                 norm=norm, plot=save_fig)
                    
                if save_phase:
                    
                    for channel in list(set(phase_names) & set(img_io.chNamesOut)):

                    
                        if ph_recon.focus_idx == z_sub_idx and channel == 'Phase2D':
                            img = imBitConvert(physical_data.phase_2D*config.plotting.phase_2D_scaling, bit=16, norm=True, limit=[-5, 5])
                            img_dict[channel] = img.copy()
                        elif channel == 'Phase_semi3D':
                            img = imBitConvert(physical_data.phase_semi3D[..., z_sub_idx]*config.plotting.phase_2D_scaling, bit=16, norm=True, limit=[-5, 5])
                            img_dict[channel] = img.copy()
                        elif channel == 'Phase3D':
                            img = imBitConvert(physical_data.phase_3D[..., z_sub_idx]*config.plotting.phase_3D_scaling, bit=16, norm=True, limit=[-5, 5])
                            img_dict[channel] = img.copy()
                            
                    
                        
                        
                if save_stokes:
                    img_stokes = [s0, s1, s2, s3]
                    img_stokes_sm = [stack[..., z_sub_idx] for stack in norm_sample.data]

                    if save_stokes_fig:
                        plot_stokes(img_io, img_stokes, img_stokes_sm)
                    img_stokes         = [x.astype(np.float32, copy=False) for x in img_stokes]
                    img_stokes_sm      = [x.astype(np.float32, copy=False) for x in img_stokes_sm]
                    img_stokes_dict    = dict(zip(stokes_names, img_stokes))
                    img_stokes_sm_dict = dict(zip(stokes_names_sm, img_stokes_sm))
                    img_dict.update(img_stokes_dict)
                    img_dict.update(img_stokes_sm_dict)
                exportImg(img_io, img_dict)
            print('Finish plotting')

    return img_io


def loopZBg(img_io, config):
    """
    Loop through each z in the sample metadata; computes the illumination function
    of fluorescence channels using image opening or looking for empty images,
    currently only process the first Z for speed
    Parameters
    ----------
    img_io: mManagerReader
        mManagerReader object that holds the image parameters

    Returns
    -------
    img_io: mManagerReader
        mManagerReader object that holds the image parameters

    """
    binning = config.processing.binning
    for zIdx in range(0, 1):  # only use the first z
        img_io.zIdx = zIdx
        ImgRawSm, ImgProcSm, ImgFluor, ImgBF = parse_tiff_input(img_io, config.dataset.ROI)
        if isinstance(ImgFluor, np.ndarray):
            ImgFluor = mean_pooling_2d_stack(ImgFluor, binning)
        for i in range(ImgFluor.shape[0]):
            if np.any(ImgFluor[i, :, :]):  # if the flour channel exists
                if img_io.ff_method == 'open':
                    img_io.ImgFluorSum[i, :, :] += cv2.morphologyEx(ImgFluor[i, :, :], cv2.MORPH_OPEN, img_io.kernel,
                                                                    borderType=cv2.BORDER_REPLICATE)
                elif img_io.ff_method == 'empty':
                    img_io.ImgFluorMin[i, :, :] = ImgMin(ImgFluor[i, :, :], img_io.ImgFluorMin[i, :, :])
    return img_io

