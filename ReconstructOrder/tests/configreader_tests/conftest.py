import numpy as np
import pytest
import os
import yaml


"""
pytest fixtures are "setup" code that makes resources available for tests
pass a keyword scope = "session", or scope = "module" if the fixture would not
be re-created for each test.
"""


@pytest.fixture
def setup_temp_yaml():
    """
    resource for yaml file

    :return:
    """

    print("setting up data, proc dirs")
    DATA_DIR = os.getcwd()+'/temp_data'
    PROCESSED_DIR = os.getcwd()+"/temp_proc"
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.isdir(PROCESSED_DIR):
        os.mkdir(PROCESSED_DIR)

    print("setting up subfolders in data")
    BACKGROUND_DIR = DATA_DIR+"/background1"
    SAMPLE_DIRS = [DATA_DIR+"/sample1", DATA_DIR+"/sample2"]
    if not os.path.isdir(BACKGROUND_DIR):
        os.mkdir(BACKGROUND_DIR)
    for sample in SAMPLE_DIRS:
        if not os.path.isdir(sample):
            os.mkdir(sample)

    config_in = {
        "dataset": {'data_dir': DATA_DIR,
                    'samples': ['sample1', 'sample2'],
                    'positions': 'all',
                    'z_slices': 'all',
                    'timepoints': 'all',
                    'ROI': [0, 0, 2048, 2048],
                    'background': 'background1',
                    'processed_dir': PROCESSED_DIR
                    },

        "processing": {'output_channels': ['Brightfield_computed',
                                            'Retardance',
                                            'Orientation',
                                            'Polarization',
                                            'Phase2D',
                                            'Phase3D'],
                       'circularity': 'rcp',
                       'background_correction': 'Input',
                       'local_fit_order': 2,
                       'flatfield_correction': False,
                       'azimuth_offset': 0,
                       'separate_positions': True,
                       'n_slice_local_bg': 'all',
                       'binning': 1,

                       'use_gpu': True,
                       'gpu_id': 0,

                       'pixel_size': 6.5,
                       'magnification': 20,
                       'NA_objective': 0.55,
                       'NA_condenser': 0.4,
                       'n_objective_media': 1,
                       'focus_zidx': 10,

                       'phase_denoiser_2D': 'TV',

                       'Tik_reg_abs_2D': 1.0e-3,
                       'Tik_reg_ph_2D': 1.0e-3,

                       'rho_2D': 1,
                       'itr_2D': 50,
                       'TV_reg_abs_2D': 1.0e-3,
                       'TV_reg_ph_2D': 1.0e-5,

                       'phase_denoiser_3D': 'Tikhonov',

                       'Tik_reg_ph_3D': 1.0e-3,

                       'rho_3D': 1.0e-3,
                       'itr_3D': 50,
                       'TV_reg_ph_3D': 5.0e-5,

                       'pad_z': 5
                       },

        "plotting": {'normalize_color_images': True,

                     'transmission_scaling': 1E4,
                     'retardance_scaling': 1E3,
                     'phase_2D_scaling': 1,
                     'phase_3D_scaling': 1,

                     'save_birefringence_fig': False,
                     'save_stokes_fig': False,
                     'save_polarization_fig': False,
                     'save_micromanager_fig': False
                     },
    }

    with open(DATA_DIR+"/temp_yaml.yml", 'w') as file:
        out = yaml.dump(config_in, file)

    yield DATA_DIR+"/temp_yaml.yml"

    # breakdown files
    print("breaking down yaml file")
    if os.path.isfile(DATA_DIR+'/temp_yaml.yml'):
        os.remove(DATA_DIR+'/temp_yaml.yml')

    # breakdown subfolders
    print("breaking down subfolders")
    if os.path.isdir(BACKGROUND_DIR):
        os.rmdir(BACKGROUND_DIR)
    for sample in SAMPLE_DIRS:
        if os.path.isdir(sample):
            os.rmdir(sample)

    print("breaking down folders")
    # breakdown folders
    if os.path.isdir(DATA_DIR):
        os.rmdir(DATA_DIR)
    if os.path.isdir(BACKGROUND_DIR):
        os.rmdir(BACKGROUND_DIR)
    if os.path.isdir(PROCESSED_DIR):
        os.rmdir(PROCESSED_DIR)