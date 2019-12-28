# bchhun, {2019-09-16}


import pytest
import os
import tifffile as tf
import numpy as np

from google_drive_downloader import GoogleDriveDownloader as gdd
from ReconstructOrder.datastructures import IntensityData, PhysicalData
from ReconstructOrder.compute.reconstruct import ImgReconstructor


@pytest.fixture(scope="session")
def setup_gdrive_src_data():
    """
    load 5-polstate intensity images of background and sample from google drive

    Data is of microglia extracted from primary brain tissue:
    https://drive.google.com/drive/folders/15JlTPe0cr-V-Hs7czbKAqNt3h80M_4jV?usp=sharing
    SM_2019_0612_20x_1
    B3_Site-1

    Background data is from a blank region of the same sample:
    https://drive.google.com/drive/folders/1TF_JNsThkdk501m3y8i8xjytDakL7FIl?usp=sharing
    SM_2019_0612_20x_1
    B3_Site-1

    Returns
    -------
    Background IntensityData object, Sample IntensityData object
    """

    temp_folder = os.getcwd()+'/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # DO NOT ADJUST THESE VALUES
    bg_state0_url = '1IKMWWg4wy1kacD45M6vANHgfKSjiamht'
    bg_state1_url = '1GEvuNCCOpDFH4MqirKXmxF8Y_Cj9uy50'
    bg_state2_url = '1gh32BPvNPiOcZKpCa8LZiFNPmocNoPkq'
    bg_state3_url = '15y8jv1_FoK3fUyKuQn_8EACmW6Os4wG_'
    bg_state4_url = '1-1k0OC1JIwkPtQ4ezvUjF5DRJ8FEYDsS'

    sm_state0_url = '1U4SGHMN3u24x9siWi7Amn8i6HRJiMu1B'
    sm_state1_url = '1zCXGbfhadNrRYzxGCEB0iyplYl1hdTaU'
    sm_state2_url = '1nwy-axwZz98SaArML0yQEBsMw1O4cD5Z'
    sm_state3_url = '1VBHWQeii3a11EegqKqM9S6cSfT5g3G27'
    sm_state4_url = '10Z2r0QjAxxte7GB_EleEryUn0URcWNGK'

    bg_urls = [bg_state0_url, bg_state1_url, bg_state2_url, bg_state3_url, bg_state4_url]
    sm_urls = [sm_state0_url, sm_state1_url, sm_state2_url, sm_state3_url, sm_state4_url]

    bg_dat = IntensityData()
    bg_dat.channel_names = ['IExt', 'I90', 'I135', 'I45', 'I0']
    bg_paths = []
    for idx, url in enumerate(bg_urls):
        output = temp_folder+"/bg_%d.tif" % idx
        bg_paths.append(output)
        gdd.download_file_from_google_drive(file_id=url,
                                            dest_path=output,
                                            unzip=False,
                                            showsize=True,
                                            overwrite=True)
        bg_dat.replace_image(tf.imread(output), idx)

    sm_dat = IntensityData()
    sm_dat.channel_names = ['IExt', 'I90', 'I135', 'I45', 'I0']
    sm_paths = []
    for idx, url in enumerate(sm_urls):
        output = temp_folder+"/sm_%d.tif" % idx
        sm_paths.append(output)
        gdd.download_file_from_google_drive(file_id=url,
                                            dest_path=output,
                                            unzip=False,
                                            showsize=True,
                                            overwrite=True)
        sm_dat.replace_image(tf.imread(output), idx)
        print("download complete "+output)

    # return intensity data objects
    yield bg_dat, sm_dat

    # breakdown files
    for bg in bg_paths:
        if os.path.isfile(bg):
            os.remove(bg)
            print("\nbreaking down temp file")
    for sm in sm_paths:
        if os.path.isfile(sm):
            os.remove(sm)
            print("\nbreaking down temp file")
    if os.path.isdir(temp_folder) and not os.listdir(temp_folder):
        os.rmdir(temp_folder)
        print("breaking down temp folder")
    else:
        print("temp folder is not empty")


@pytest.fixture(scope="session")
def setup_gdrive_target_data():
    """
    load physical data object populated from above reconstructions.

    https://drive.google.com/drive/folders/1hyBwSZ3IXiWkE_sHkSKr8X8pC5kjJr9C?usp=sharing

    Returns
    -------
    Physical Data object populated with reconstructions from above

    """
    temp_folder = os.getcwd()+'/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # DO NOT ADJUST THESE VALUES
    target_state0_url = '19R3-pyEWmv8ZBAAqVgZsOGc-2j197kmq'
    target_state1_url = '1_rzqix5YtH1TWAOF65e4FhAvKdpluPnM'
    target_state2_url = '10X4TrbK2fEwNAwf8BpkK0CfdoRxn0939'
    target_state3_url = '1fUfq0qogmnhsElR0t_ILqsfFz8CeUHO2'

    target_urls = [target_state0_url, target_state1_url, target_state2_url, target_state3_url]
    target_names = ['reconstructed_birefring_microglia_single_I_trans.npy',
                    'reconstructed_birefring_microglia_single_retard.npy',
                    'reconstructed_birefring_microglia_single_azimuth.npy',
                    'reconstructed_birefring_microglia_single_polarization.npy']

    target_dat = PhysicalData()
    target_paths = []
    for idx, url in enumerate(target_urls):
        output = temp_folder + '/' + target_names[idx]
        target_paths.append(output)
        gdd.download_file_from_google_drive(file_id=url,
                                            dest_path=output,
                                            unzip=False,
                                            showsize=True,
                                            overwrite=True)
        print("download complete "+output)

    target_dat.I_trans = np.load(temp_folder+'/'+target_names[0])
    target_dat.retard = np.load(temp_folder+'/'+target_names[1])
    target_dat.azimuth = np.load(temp_folder+'/'+target_names[2])
    target_dat.polarization = np.load(temp_folder+'/'+target_names[3])

    yield target_dat

    # breakdown files
    for target in target_paths:
        if os.path.isfile(target):
            os.remove(target)
            print("\nbreaking down temp file")
    if os.path.isdir(temp_folder) and not os.listdir(temp_folder):
        os.rmdir(temp_folder)
        print("breaking down temp folder")
    else:
        print("temp folder is not empty")


@pytest.fixture(scope="session")
def setup_reconstructed_data(setup_gdrive_src_data):
    bg_dat, sm_dat = setup_gdrive_src_data

    # compute initial reconstructor using background data
    img_reconstructor = ImgReconstructor(bg_dat,
                                         swing=0.03,
                                         wavelength=532)
    bg_stokes = img_reconstructor.compute_stokes(bg_dat)
    bg_stokes_normalized = img_reconstructor.stokes_normalization(bg_stokes)

    # compute sample stokes and correct with background data
    sm_stokes = img_reconstructor.compute_stokes(sm_dat)
    sm_stokes_normalized = img_reconstructor.stokes_normalization(sm_stokes)
    sm_stokes_normalized = img_reconstructor.correct_background(sm_stokes_normalized, bg_stokes_normalized)

    reconstructed_birefring = img_reconstructor.reconstruct_birefringence(sm_stokes_normalized)

    yield reconstructed_birefring

#todo: write a fixture to load multi-dim data (2-p, 2-t, 5-z)

#todo: write a fixture to create a dummy config

#todo: write a fixture to replicate simulated data