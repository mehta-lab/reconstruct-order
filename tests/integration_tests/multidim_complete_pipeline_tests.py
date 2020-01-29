# bchhun, {2019-12-12}

"""
these tests will download a zip file from GDD
the zip file contains a configuration yaml and sample data


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
    print("length of proc ="+str(len(processed_files)))

    target_folder = setup_multidim_target
    target_files = glob.glob(target_folder+'/*.tif')
    print("TARGET FILES" + str(target_files))
    print("length of proc ="+str(len(target_files)))

    if len(target_files) != len(processed_files):
        pytest.fail("unable to process all necessary files")

    for file in processed_files:
        target_match = [match for match in target_files if os.path.basename(match) == os.path.basename(file)]
        if len(target_match) > 1:
            pytest.fail("more than one target for processed file = "+file)
        if len(target_match) == 0:
            pytest.fail("too many processed files generated without expected matches. Processed file name = "+file)

        predict = tf.imread(file)
        target = tf.imread(target_match)

        try:
            assert mse(predict, target) == np.finfo(np.float32).eps
        except AssertionError as ae:
            print(f"MSE relative = {mse(predict, target)}")
            print(f"MSE FAIL ON FILE = "+file)
            print(f"MSE FAIL ON TARGET MATCH = "+target_match[0])
