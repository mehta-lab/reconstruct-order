# bchhun, {2019-12-12}

"""
these tests will download a zip file from GDD
the zip file contains a configuration yaml and sample data

dataset:
    data_dir: './temp/src'
    processed_dir: './temp/predict'
    samples: ['SM_2019_0612_20x_1']
    positions: ['B3-Site_1']
    z_slices: 'all'
    timepoints: [0,1,2]
    ROI: [0, 0, 256, 256] # [Ny_start, Nx_start, number_of_pixel_in_y, number_of_pixel_in_x]
    background: 'BG_2019_0612_1515_1'

processing:
    output_channels: ['Brightfield_computed', 'Retardance', 'Orientation', 'Polarization', 'Phase2D', 'Phase3D']
    circularity: 'rcp'
    background_correction: 'Input'
    flatfield_correction: False
    n_slice_local_bg: all
    azimuth_offset: 0
    separate_positions: True

    use_gpu: True
    gpu_id: 0
"""
from ReconstructOrder.workflow.reconstructBatch import reconstructBatch
import os, glob
import tifffile as tf
import pytest
import numpy as np
from ..testMetrics import mse


def test_reconstruct_source(setup_multidim_src):
    config = setup_multidim_src
    try:
        reconstructBatch(config)
    except Exception as ex:
        pytest.fail("Exception thrown during reconstruction = "+str(ex))


def test_src_target_mse(setup_multidim_src, setup_multidim_target):
    config = setup_multidim_src
    reconstructBatch(config)

    processed_folder = os.getcwd() + '/temp/predict/src/SM_2019_0612_20x_1_BG_2019_0612_1515_1/B3-Site_1'
    processed_files = glob.glob(processed_folder+'/*.tif')
    print("PROCESSED FILES" + str(processed_files))
    print("number of proc images ="+str(len(processed_files)))

    target_folder = setup_multidim_target
    target_files = glob.glob(target_folder+'/*.tif')
    print("TARGET FILES" + str(target_files))
    print("number of target images ="+str(len(target_files)))

    p_sort = sorted(processed_files)
    s_sort = sorted(target_files)

    for idx, file in enumerate(p_sort):
        if os.path.basename(file) == os.path.basename(s_sort[idx]):
            predict = tf.imread(file)
            target = tf.imread(s_sort[idx])

            try:
                assert mse(predict, target) <= np.finfo(np.float32).eps
            except AssertionError as ae:
                print(f"MSE relative = {mse(predict, target)}")
                print(f"MSE FAIL ON PREDICT = " + file)
                print(f"MSE FAIL ON TARGET  = " + target + "\n")
                if 'img_Phase' in target:
                    continue
                else:
                    pytest.fail("Assertion Error = " + str(ae))
