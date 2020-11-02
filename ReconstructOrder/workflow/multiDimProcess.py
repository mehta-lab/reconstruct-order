"""
Process data collected over multiple positions, timepoints and z slices
"""

import os
import numpy as np
import matplotlib
import time
import matplotlib.pyplot as plt
from ..utils.imgIO import export_img
from ..compute.reconstruct import ImgReconstructor
from ..compute.reconstruct_phase import phase_reconstructor
from ..utils.imgProcessing import im_bit_convert
from ..utils.plotting import render_birefringence_imgs, plot_stokes, plot_pol_imgs
from ReconstructOrder.utils.mManagerIO import mManagerReader, PolAcquReader
from ..datastructures import StokesData, IntensityDataCreator, IntensityData
from ..utils.ConfigReader import ConfigReader
from ..utils.flat_field import FlatFieldCorrector
from ..utils.aux_utils import loop_pt
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
                                output_chans=config.processing.output_channels,
                                binning=config.processing.binning
                                )
    except:
        img_obj = mManagerReader(data_path,
                                 output_chans=config.processing.output_channels,
                                 binning=config.processing.binning)

    # img_obj = PolAcquReader(data_path,
    #                         output_chans=config.processing.output_channels,
    #                         binning=config.processing.binning
    #                         )
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

        img_obj_list[i].img_output_path = OutputPath
        os.makedirs(OutputPath, exist_ok=True)  # create folder for processed images
    return img_obj_list


def process_background(img_io, img_io_bg, config: ConfigReader, img_int_creator: IntensityDataCreator):
    """
    Read background images, initiate ImgReconstructor to compute background stokes parameters

    """
    img_int_bg         = img_int_creator.get_data_object(config, img_io_bg)
    circularity      = config.processing.circularity
    azimuth_offset   = config.processing.azimuth_offset
    n_slice_local_bg = config.processing.n_slice_local_bg
    local_fit_order  = config.processing.local_fit_order
    use_gpu          = config.processing.use_gpu
    gpu_id           = config.processing.gpu_id
    
    if n_slice_local_bg == 'all':
        n_slice_local_bg = len(img_io.z_list)

    img_reconstructor = ImgReconstructor(img_int_bg,
                                         config           = config,
                                         bg_method        = img_io.bg_method,
                                         n_slice_local_bg = n_slice_local_bg,
                                         poly_fit_order   = local_fit_order,
                                         swing            = img_io.swing,
                                         wavelength       = img_io.wavelength,
                                         azimuth_offset   = azimuth_offset,
                                         circularity      = circularity,
                                         use_gpu          = use_gpu,
                                         gpu_id           = gpu_id)

    if img_io.bg_correct:
        background_stokes = img_reconstructor.compute_stokes(config, img_int_bg)
        background_stokes_normalized = img_reconstructor.stokes_normalization(background_stokes)
        # print('denoising the background...')
        # img_stokes_bg = [denoise_tv_chambolle(img, weight=10**6) for img in img_stokes_bg]
        # img_stokes_bg = [cv2.GaussianBlur(img, (5, 5), 0) for img in img_stokes_bg]
        # img_stokes_bg = [cv2.medianBlur(img, 5) for img in img_stokes_bg]
    else:
        background_stokes_normalized = None

    return background_stokes_normalized, img_int_bg, img_reconstructor

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
    N_defocus    = img_io.n_z
    
    if config.dataset.ROI is None:
        ROI = [0,0, img_io.height, img_io.width]
    else:
        ROI = config.dataset.ROI

    start_time=time.time()
    print('Computing phase transfer function (t=0 min)')

    opt_mapping = {'Phase2D': '2D', 'Phase_semi3D': 'semi-3D', 'Phase3D': '3D'}
    phase_deconv = [opt_mapping[opt] for opt in img_io.output_chans if opt in opt_mapping.keys()]
    
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

    elapsed_time=(time.time()-start_time)/60
    print('Finished computing phase transfer function (t=%3.2f min)'% elapsed_time)
    
    return ph_recon

@loop_pt
def process_sample_imgs(img_io: Union[mManagerReader, PolAcquReader]=None,
                        config: ConfigReader=None,
                        img_reconstructor: ImgReconstructor=None,
                        img_int_creator: IntensityDataCreator=None,
                        ff_corrector: FlatFieldCorrector=None,
                        int_bg: IntensityData=None,
                        stokes_bg: StokesData=None,
                        ph_recon: phase_reconstructor=None):
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

    t_idx            = img_io.t_idx
    pos_idx          = img_io.pos_idx
    z_list           = img_io.z_list
    n_slice_local_bg = img_reconstructor.n_slice_local_bg
    norm             = config.plotting.normalize_color_images
    separate_pos = config.processing.separate_positions
    save_fig         = config.plotting.save_birefringence_fig
    save_stokes_fig  = config.plotting.save_stokes_fig
    save_pol_fig     = config.plotting.save_polarization_fig
    

    # 1) define allowed config values
    #TODO: move the flag parser to ConfigReader and make flags part of the config attributes
    pol_names       = ['Pol_State_0', 'Pol_State_1', 'Pol_State_2', 'Pol_State_3', 'Pol_State_4']
    stokes_names    = ['Stokes_0', 'Stokes_1', 'Stokes_2', 'Stokes_3']
    stokes_names_sm = [x + '_sm' for x in stokes_names]
    birefring_names = ['Brightfield_computed', 'Retardance', 'Orientation', 'Orientation_x', 'Orientation_y',
                       'Transmission', 'Polarization',
                       'Retardance+Orientation', 'Polarization+Orientation', 'Brightfield+Retardance+Orientation',
                       'Brightfield_computed+Retardance+Orientation',
                       'Retardance+Fluorescence', 'Retardance+Fluorescence_all']
    phase_names     = ['Phase2D', 'Phase_semi3D', 'Phase3D']
    fluor_names     = ['405', '488', '568', '640', 'ex561em700']

    # 2 )set flags based on names defined in 1)
    save_stokes    = any(chan in stokes_names + stokes_names_sm for chan in img_io.output_chans) \
                     or save_stokes_fig
    save_phase     = any(chan in phase_names for chan in img_io.output_chans)
    save_birefring = any(chan in birefring_names for chan in img_io.output_chans) or save_fig or save_phase
    save_BF        = 'Brightfield' in img_io.output_chans
    save_pol       = any(chan in pol_names for chan in img_io.output_chans) or save_pol_fig
    save_fluor     = any(chan in fluor_names for chan in img_io.output_chans)

    print('Processing position %03d, time %03d ... (t=0 min)' % (pos_idx, t_idx))
    start_time=time.time()
    for z_stack_idx in range(0, len(z_list), n_slice_local_bg):
        stokes_param_sm_stack = [[] for i in range(len(stokes_names))]
        fluor_stack_list = []

        for z_list_idx in range(z_stack_idx, z_stack_idx + n_slice_local_bg):
            z_idx = z_list[z_list_idx]

            plt.close("all")  # close all the figures from the last run
            img_io.z_idx = z_idx

            # load raw intensity data
            img_int_sm = img_int_creator.get_data_object(config, img_io)
            img_int_sm = ff_corrector.correct_flat_field(img_int_sm)

            img_dict = {}
            if save_stokes or save_birefring:
                # compute stokes
                stokes_param_sm = img_reconstructor.compute_stokes(config, img_int_sm)
                for stack, img in zip(stokes_param_sm_stack, stokes_param_sm.data):
                    stack.append(img)
                # retard = removeBubbles(retard)     # remove bright speckles in mounted brain slice images
            img_bf = img_int_sm.get_image('BF')
            if save_BF and isinstance(img_bf, np.ndarray):
                img_bf = img_bf / stokes_bg.s0  # flat-field correction
                img_bf = im_bit_convert(img_bf * config.plotting.transmission_scaling, bit=16, norm=False)
                img_dict.update({'Brightfield': img_bf})

            if save_pol:
                imgs_pol = []
                for chan_name in int_bg.channel_names:
                    imgs_pol += [img_int_sm.get_image(chan_name) / int_bg.get_image(chan_name)]
                if save_pol_fig:
                    plot_pol_imgs(img_io, imgs_pol, pol_names)
                imgs_pol = [im_bit_convert(img * 10 ** 4, bit=16) for img in imgs_pol]
                img_dict.update(dict(zip(pol_names, imgs_pol)))

            img_fluor_list = [im_bit_convert(img_int_sm.get_image(chan)) for chan in fluor_names]
            img_fluor = np.stack(img_fluor_list)
            fluor_stack_list.append(img_fluor)
            if save_fluor:
                img_dict.update(dict(zip(fluor_names, img_fluor_list)))


            export_img(img_io, img_dict, separate_pos)

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
                norm_sample = img_reconstructor.correct_background(norm_sample, stokes_bg)

            elapsed_time=(time.time()-start_time)/60
            print('Reconstructing retardance and orientation (t=%3.2f min)' % elapsed_time)

            physical_data = img_reconstructor.reconstruct_birefringence(norm_sample)

            elapsed_time=(time.time()-start_time)/60
            print('Finished reconstructing retardance and orientation (t=%3.2f min)' % elapsed_time)

            if ph_recon:
                for deconv_dim in ph_recon.phase_deconv:
                    
                    if deconv_dim == '2D':
                        elapsed_time = (time.time() - start_time) / 60
                        print('Reconstructing 2D phase (t=%3.2f min)' % elapsed_time)

                        physical_data.absorption_2D, physical_data.phase_2D = ph_recon.Phase_recon_2D(norm_sample)

                        elapsed_time = (time.time() - start_time) / 60
                        print('Finished reconstructing 2D phase (t=%3.2f min)' % elapsed_time)
                    if deconv_dim == 'semi-3D':
                        elapsed_time = (time.time() - start_time) / 60
                        print('Reconstructing semi-3D phase (t=%3.2f min)' % elapsed_time)

                        physical_data.absorption_semi3D, physical_data.phase_semi3D = ph_recon.Phase_recon_semi_3D(norm_sample)

                        elapsed_time = (time.time() - start_time) / 60
                        print('Finished reconstructing semi-3D phase (t=%3.2f min)' % elapsed_time)
                    if deconv_dim == '3D':
                        elapsed_time = (time.time() - start_time) / 60
                        print('Reconstructing 3D phase (t=%3.2f min)' % elapsed_time)

                        physical_data.phase_3D = ph_recon.Phase_recon_3D(norm_sample)

                        elapsed_time = (time.time() - start_time) / 60
                        print('Finished reconstructing 3D phase (t=%3.2f min)' % elapsed_time)


            img_dict = {}
            for z_idx in range(z_stack_idx, z_stack_idx + n_slice_local_bg):
                plt.close("all")  # close all the figures from the last run
                img_io.z_idx = z_list[z_idx]
                z_sub_idx = z_idx - z_stack_idx

                # extract the relevant z slice out of the data
                s0           = physical_data.I_trans[..., z_sub_idx]
                retard       = physical_data.retard[..., z_sub_idx]
                azimuth      = physical_data.azimuth[..., z_sub_idx]
                polarization = physical_data.polarization[..., z_sub_idx]
                s1           = (norm_sample.s1_norm * norm_sample.s3)[..., z_sub_idx]
                s2           = (norm_sample.s2_norm * norm_sample.s3)[..., z_sub_idx]
                s3           = (norm_sample.s3)[..., z_sub_idx]


                img_fluor = fluor_stack_list[z_sub_idx]

                if save_birefring:
                    imgs = [s0, retard, azimuth, polarization, img_fluor]
                    img_io, img_dict = render_birefringence_imgs(img_io, imgs, config, spacing=20, vectorScl=8, zoomin=False,
                                                                 dpi=200,
                                                                 norm=norm, plot=save_fig)
                if save_phase:
                    for channel in list(set(phase_names) & set(img_io.output_chans)):
                        if ph_recon.focus_idx == z_sub_idx and channel == 'Phase2D':
                            img = im_bit_convert(physical_data.phase_2D * config.plotting.phase_2D_scaling, bit=16, norm=True, limit=[-5, 5])
                            img_dict[channel] = img.copy()
                        elif channel == 'Phase_semi3D':
                            img = im_bit_convert(physical_data.phase_semi3D[..., z_sub_idx] * config.plotting.phase_2D_scaling, bit=16, norm=True, limit=[-5, 5])
                            img_dict[channel] = img.copy()
                        elif channel == 'Phase3D':
                            img = im_bit_convert(physical_data.phase_3D[..., z_sub_idx] * config.plotting.phase_3D_scaling, bit=16, norm=True, limit=[-5, 5])
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

                export_img(img_io, img_dict, separate_pos)

            elapsed_time = (time.time()-start_time) / 60
            print('Finish processing and exporting (t=%3.2f min)' % elapsed_time)




