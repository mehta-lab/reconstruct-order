from ..workflow.multiDimProcess import process_background, process_sample_imgs, parse_bg_options, phase_reconstructor_initializer
# from ReconstructOrder.metadata.MicromanagerMetadata import read_metadata
# from ReconstructOrder.metadata.ConfigReader import ConfigReader
from ..metadata.MicromanagerMetadata import read_metadata
from ..metadata.ConfigReader import ConfigReader
from ..utils.flat_field import FlatFieldCorrector
from ..datastructures import IntensityDataCreator
import os


def process_position_list(img_obj_list, config):
    """Make sure all members of positions are part of io_obj.
    If positions = 'all', replace with actual list of positions

    Parameters
    ----------
    img_obj_list: list
        list of mManagerReader instances
    config: obj
        ConfigReader instance

    Returns
    -------
        img_obj_list: list
        list of modified mManagerReader instances

    """
    for idx, io_obj in enumerate(img_obj_list):
        config_pos_list = config.dataset.positions[idx]

        if not config_pos_list[0] == 'all':
            try:
                img_obj_list[idx].pos_list = config_pos_list
            except Exception as e:
                print('Position list {} for sample in {} is invalid'.format(config_pos_list, io_obj.img_sm_path))
                ValueError(e)
    return img_obj_list


def process_z_slice_list(img_obj_list, config):
    """Make sure all members of z_slices are part of io_obj.
    If z_slices = 'all', replace with actual list of z_slices

    Parameters
    ----------
    img_obj_list: list
        list of mManagerReader instances
    config: obj
        ConfigReader instance

    Returns
    -------
        img_obj_list: list
        list of modified mManagerReader instances

    """
    n_slice_local_bg = config.processing.n_slice_local_bg
    for idx, io_obj in enumerate(img_obj_list):
        config_z_list = config.dataset.z_slices[idx]
        if not config_z_list[0] == 'all':
            try:
                img_obj_list[idx].z_list = config_z_list
            except Exception as e:
                print('z_slice list {} for sample in {} is invalid'.format(config_z_list, io_obj.img_sm_path))
                ValueError(e)
        if not n_slice_local_bg == 'all':
            # adjust slice number to be multiple of n_slice_local_bg
            img_obj_list[idx].z_list = \
                img_obj_list[idx].z_list[0:len(img_obj_list[idx].z_list) // n_slice_local_bg * n_slice_local_bg]
    return img_obj_list


def process_timepoint_list(img_obj_list, config):
    """Make sure all members of timepoints are part of io_obj.
    If timepoints = 'all', replace with actual list of timepoints

    Parameters
    ----------
    img_obj_list: list
        list of mManagerReader instances
    config: obj
        ConfigReader instance

    Returns
    -------
        img_obj_list: list
        list of modified mManagerReader instances
    """
    for idx, io_obj in enumerate(img_obj_list):
        config_t_list = config.dataset.positions[idx]

        if not config_t_list[0] == 'all':
            try:
                img_obj_list[idx].t_list = config_t_list
            except Exception as e:
                print('Timepoint list {} for sample in {} is invalid'.format(config_t_list, io_obj.img_sm_path))
                ValueError(e)
    return img_obj_list


def _process_one_acqu(img_obj, bg_obj, config):
    """

    Parameters
    ----------
    img_obj : mManagerReader
        mManagerReader instance for sample images
    bg_obj : mManagerReader instance for background images
    config : obj
        ConfigReader instance

    Returns
    -------

    """
    ph_recon = None
    print('Processing SAMPLE = ' + img_obj.name + ' ....')
    img_int_creator_bg = IntensityDataCreator(ROI=config.dataset.ROI,
                                           binning=config.processing.binning)

    # Write metadata in processed folder
    img_obj.writeMetaData()

    # Write config file in processed folder
    config.write_config(os.path.join(img_obj.img_output_path, 'config.yml'))  # save the config file in the processed folder

    # img_obj, img_reconstructor = process_background(img_obj, bg_obj, config)
    stokes_bg_norm, int_bg, img_reconstructor = process_background(img_obj, bg_obj, config, img_int_creator_bg)

    ff_corrector = FlatFieldCorrector(img_obj, config, method='open')
    flatField = config.processing.flatfield_correction
    if flatField:  # find background fluorescence for flatField correction
        ff_corrector.compute_flat_field()
     # determine if we will initiate phase reconstruction
    phase_names = ['Phase2D', 'Phase_semi3D', 'Phase3D']
    save_phase = any(chan in phase_names for chan in img_obj.output_chans)
    if save_phase:
        ph_recon = phase_reconstructor_initializer(img_obj, config)

    # old int_creator object has bg-channels assigned.  Need to create a new one
    img_int_creator_sm = IntensityDataCreator(ROI=config.dataset.ROI,
                                           binning=config.processing.binning)

    print("COMPUTING SAMPLE DATA")
    process_sample_imgs(img_io=img_obj,
                        config=config,
                        img_reconstructor=img_reconstructor,
                        img_int_creator=img_int_creator_sm,
                        ff_corrector=ff_corrector,
                        int_bg=int_bg,
                        stokes_bg=stokes_bg_norm,
                        ph_recon=ph_recon)

    #TODO: Write log file and metadata at the end of reconstruction


def reconstruct_batch(configfile):
    config = ConfigReader()
    config.read_config(configfile)

    # read meta data, create list of mManagerIO objects (one object per sample)
    # curate the list of mManagerIO objects based on supplied config values
    img_obj_list, bg_obj_list = read_metadata(config)
    img_obj_list = process_position_list(img_obj_list, config)
    img_obj_list = process_z_slice_list(img_obj_list, config)
    img_obj_list = process_timepoint_list(img_obj_list, config)

    # process background options
    img_obj_list = parse_bg_options(img_obj_list, config)

    for img_obj, bg_obj in zip(img_obj_list, bg_obj_list):
        _process_one_acqu(img_obj, bg_obj, config)
