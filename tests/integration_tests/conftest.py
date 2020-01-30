# bchhun, {2019-09-16}


import pytest
import os
import tifffile as tf

from google_drive_downloader import GoogleDriveDownloader as gdd
from ReconstructOrder.datastructures import IntensityData, PhysicalData
from ReconstructOrder.compute.reconstruct import ImgReconstructor


# ==================================================================================================
# ============================= Single frame Sample and background =================================
# ==================================================================================================


# ============================= SOURCE DATA ========================================================
@pytest.fixture(scope="session")
def setup_gdrive_src_data_small():
    """
    load 5-polstate intensity images of background and sample from google drive

    SIZE: 512 x 512
    TYPE: .tif

    Data is of microglia extracted from primary brain tissue:
    https://drive.google.com/drive/folders/1u4YQ4j3QjbFXLAoFRro2XTYDcwa4-X76?usp=sharing
    SM_2019_0612_20x_1
    B3_Site-1

    Background data is from a blank region of the same sample:
    https://drive.google.com/drive/folders/1u4YQ4j3QjbFXLAoFRro2XTYDcwa4-X76?usp=sharing
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
    # 512 x 512
    bg_state0_url = '1B8RxK5QaS_TJo3_GEWWnswRePKEoJe9g'
    bg_state1_url = '1VGKgO-JRhyhx7WWRABAIWYR1OqukM2qz'
    bg_state2_url = '1R-0b95DG0mMa9ArzKlvMcHmzPGHK80bV'
    bg_state3_url = '1WrTHe4NvaoTW6Uo2Xg8sB2Xk76mpr1qV'
    bg_state4_url = '1BK2DbBPhFQYhy25RbOxHO7t2skT6NaKU'

    sm_state0_url = '1OBTAq_SOG1QhlErm9ewezh8j_eNjP5E9'
    sm_state1_url = '18o8QfpfBO-JuU-Pw-pn_kp7HW-DYIOHv'
    sm_state2_url = '146drgC8HAUH8qvOj_ubn04TNzw84IAEI'
    sm_state3_url = '18fT2z-oOAxfalXlSh--iclEWUp0Tyhdy'
    sm_state4_url = '1CoHOM6XsCqlkJ-Wb8gGEEm59wd8pC_-U'

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
def setup_gdrive_src_data_large():
    """
    load 5-polstate intensity images of background and sample from google drive

    SIZE: 2048 x 2048
    TYPE: .tif

    Data is of microglia extracted from primary brain tissue:
    https://drive.google.com/drive/folders/1u4YQ4j3QjbFXLAoFRro2XTYDcwa4-X76?usp=sharing
    SM_2019_0612_20x_1
    B3_Site-1

    Background data is from a blank region of the same sample:
    https://drive.google.com/drive/folders/1u4YQ4j3QjbFXLAoFRro2XTYDcwa4-X76?usp=sharing
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

    # 2048 x 2048
    bg_state0_url = '1T35V5hXy0e38ZMmcD-Dc6SqtSsAAOpjl'
    bg_state1_url = '1WXzIzgc0tqiJInHf1OIPB6LYcxzxrsTH'
    bg_state2_url = '1Uo2FZn84gPv66ltVdvi4EnGFvmKhvNxR'
    bg_state3_url = '15j8elI2AfXHHiitsFaFTBE8CljnNwpWI'
    bg_state4_url = '1tzJYQH2_b4PLa2NRuQLalBjZ5033VNIa'

    sm_state0_url = '1AdGBX-xjZyiXEkhw4pOn1vGEpwX-SDxE'
    sm_state1_url = '1qNPJloQ6qYu681nJNrRCLrRqZqofRohb'
    sm_state2_url = '10G3Hn1awWPUmDXYG9KjQnI0dCTbOnUcc'
    sm_state3_url = '1J_1yMco91Q0tRxdSn2gYIfhJnA5x_7Xc'
    sm_state4_url = '1bae637cdITWLxpaq6iVzMbWERC4zTCWv'

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


# ============================= TARGET DATA ========================================================
@pytest.fixture(scope="session")
def setup_gdrive_target_data_large_tif():
    """
    load physical data object populated from above reconstructions.

    SIZE: 2048 x 2048
    TYPE: .tif

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
    # 2048 x 2048
    # .tif
    target_I_trans_url = '1TmA5XaSyznOKHaBalQzxlFVR2TNCFMjz'
    target_retard_url = '1w-xFqVB3w1YA22xh0bv1jtvUW3h1-8VT'
    target_polarization_url = '17C35xRnkcjchNfxqsxChV6j8XDbHXLPA'
    target_azimuth_url = '1obXQia04nOgR7gSRmOh9Wza--Monb0GC'

    target_urls = [target_I_trans_url, target_retard_url, target_polarization_url, target_azimuth_url]
    target_names = ['reconstructed_birefring_microglia_single_I_trans.npy',
                    'reconstructed_birefring_microglia_single_retard.npy',
                    'reconstructed_birefring_microglia_single_polarization.npy',
                    'reconstructed_birefring_microglia_single_azimuth.npy']

    # download files to temp and remember the paths
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

    # assign data to datastructure
    target_dat = PhysicalData()
    target_dat.I_trans = tf.imread(temp_folder+'/'+target_names[0])
    target_dat.retard = tf.imread(temp_folder+'/'+target_names[1])
    target_dat.polarization = tf.imread(temp_folder+'/'+target_names[2])
    target_dat.azimuth = tf.imread(temp_folder+'/'+target_names[3])

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
def setup_gdrive_target_data_small_tif():
    """
    load physical data object populated from above reconstructions.

    SIZE: 512 x 512
    TYPE: .tif

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

    # 512 x 512
    # .tif
    target_I_trans_url = '1pkBDuFRa237QUcqRbSZsLt7udfrh0Bwz'
    target_retard_url = '1QO6Qu1fhtLpm2vK6STdWEIMmRH6ZPntd'
    target_polarization_url = '1wOTGeCp2xmJE1ts4irzDNT4KkUuR0vrd'
    target_azimuth_url = '14XNmiw-NJXwRPAfkF4oSZFk1AlF-Jwu1'

    target_urls = [target_I_trans_url, target_retard_url, target_polarization_url, target_azimuth_url]
    target_names = ['reconstructed_birefring_microglia_single_I_trans.npy',
                    'reconstructed_birefring_microglia_single_retard.npy',
                    'reconstructed_birefring_microglia_single_polarization.npy',
                    'reconstructed_birefring_microglia_single_azimuth.npy']

    # download files to temp and remember the paths
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

    # assign data to datastructure
    target_dat = PhysicalData()

    target_dat.I_trans = tf.imread(temp_folder+'/'+target_names[0])
    target_dat.retard = tf.imread(temp_folder+'/'+target_names[1])
    target_dat.polarization = tf.imread(temp_folder+'/'+target_names[2])
    target_dat.azimuth = tf.imread(temp_folder+'/'+target_names[3])

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
def setup_gdrive_target_data_large_npy():
    """
    load physical data object populated from above reconstructions.

    SIZE: 2048 x 2048
    TYPE: .npy

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

    # 2048 x 2048
    # .npy
    target_I_trans_url = '1_g36irdTc9qWAG1ylWwA1cBx_un37SuS'
    target_retard_url = '1IAX3Y4vWVf0nKL-Td4p9DcrxcmZ6mPnd'
    target_polarization_url = '14lGQJQwaTvL-5bLQKa2SbaRzOV-QdM2e'
    target_azimuth_url = '1gDFx1nsmyt4C0QuoHV-59ZBxdh5w1FQD'

    target_urls = [target_I_trans_url, target_retard_url, target_polarization_url, target_azimuth_url]
    target_names = ['reconstructed_birefring_microglia_single_I_trans.npy',
                    'reconstructed_birefring_microglia_single_retard.npy',
                    'reconstructed_birefring_microglia_single_polarization.npy',
                    'reconstructed_birefring_microglia_single_azimuth.npy']

    # download files to temp and remember the paths
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

    # assign data to datastructure
    import numpy as np
    target_dat = PhysicalData()
    target_dat.I_trans = np.load(temp_folder+'/'+target_names[0])
    target_dat.retard = np.load(temp_folder+'/'+target_names[1])
    target_dat.polarization = np.load(temp_folder+'/'+target_names[2])
    target_dat.azimuth = np.load(temp_folder+'/'+target_names[3])

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
def setup_gdrive_target_data_small_npy():
    """
    load physical data object populated from above reconstructions.

    SIZE: 512 x 512
    TYPE: .npy

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

    # .npy
    target_I_trans_url = '1pT8zewV_7H-0s2AW7a3q2ZZJxpuRuBMU'
    target_retard_url = '1D0SnrLcZsoK6BSI96dPLwPbKf6NfnIMt'
    target_polarization_url = '1xtQtxj-NWorVPLgJFV4IFbu1luQHZvEi'
    target_azimuth_url = '1C0T1b-KMEIrZzHNYeGWG-YT69SBbZ1Xc'

    target_urls = [target_I_trans_url, target_retard_url, target_polarization_url, target_azimuth_url]
    target_names = ['reconstructed_birefring_microglia_single_I_trans.npy',
                    'reconstructed_birefring_microglia_single_retard.npy',
                    'reconstructed_birefring_microglia_single_polarization.npy',
                    'reconstructed_birefring_microglia_single_azimuth.npy']

    # download files to temp and remember the paths
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

    # assign data to datastructure
    import numpy as np
    target_dat = PhysicalData()
    target_dat.I_trans = np.load(temp_folder+'/'+target_names[0])
    target_dat.retard = np.load(temp_folder+'/'+target_names[1])
    target_dat.polarization = np.load(temp_folder+'/'+target_names[2])
    target_dat.azimuth = np.load(temp_folder+'/'+target_names[3])

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


# ==================================================================================================
# ====================== Single frame Sample and background RECONSTRUCTIONS ========================
# ==================================================================================================

# todo: figure out parameterized fixtures so that much of the below repeated code can be eliminated

@pytest.fixture(scope="session")
def setup_reconstructed_data_npy(setup_gdrive_src_data_small):
    """
    Loads AND Reconstructs SOURCE data.

    SIZE: 512 x 512
    TYPE: .tif

    :param setup_gdrive_src_data_small:
    :return: physical data object containing reconstructed data
    """
    bg_dat, sm_dat = setup_gdrive_src_data_small

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


@pytest.fixture(scope="session")
def setup_reconstructed_data_large_tif(setup_gdrive_src_data_large):
    """
    Loads AND Reconstructs SOURCE data.

    SIZE: 2048 x 2048
    TYPE: .tif

    :param setup_gdrive_src_data_large:
    :return: physical data object containing reconstructed data
    """
    bg_dat, sm_dat = setup_gdrive_src_data_large

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

    # target data is scaled to uint16 (.tif)
    spol = (65536*(reconstructed_birefring.polarization - reconstructed_birefring.polarization.min()) /
            (reconstructed_birefring.polarization.ptp())).astype('uint16')
    sazi = (65536*(reconstructed_birefring.azimuth - reconstructed_birefring.azimuth.min()) /
            (reconstructed_birefring.azimuth.ptp())).astype('uint16')
    stran = (65536*(reconstructed_birefring.I_trans - reconstructed_birefring.I_trans.min()) /
             (reconstructed_birefring.I_trans.ptp())).astype('uint16')
    sret = (65536*(reconstructed_birefring.retard - reconstructed_birefring.retard.min()) /
            (reconstructed_birefring.retard.ptp())).astype('uint16')

    reconstructed_birefring.I_trans = stran
    reconstructed_birefring.retard = sret
    reconstructed_birefring.polarization = spol
    reconstructed_birefring.azimuth = sazi

    yield reconstructed_birefring


@pytest.fixture(scope="session")
def setup_reconstructed_data_small_tif(setup_gdrive_src_data_small):
    """
    Loads AND Reconstructs SOURCE data.

    SIZE: 512 x 512
    TYPE: .tif

    :param setup_gdrive_src_data_small:
    :return: physical data object containing reconstructed data
    """
    bg_dat, sm_dat = setup_gdrive_src_data_small

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

    # target data is scaled to uint16 (.tif)
    spol = (65536*(reconstructed_birefring.polarization - reconstructed_birefring.polarization.min()) /
            (reconstructed_birefring.polarization.ptp())).astype('uint16')
    sazi = (65536*(reconstructed_birefring.azimuth - reconstructed_birefring.azimuth.min()) /
            (reconstructed_birefring.azimuth.ptp())).astype('uint16')
    stran = (65536*(reconstructed_birefring.I_trans - reconstructed_birefring.I_trans.min()) /
             (reconstructed_birefring.I_trans.ptp())).astype('uint16')
    sret = (65536*(reconstructed_birefring.retard - reconstructed_birefring.retard.min()) /
            (reconstructed_birefring.retard.ptp())).astype('uint16')

    reconstructed_birefring.I_trans = stran
    reconstructed_birefring.retard = sret
    reconstructed_birefring.polarization = spol
    reconstructed_birefring.azimuth = sazi

    yield reconstructed_birefring


@pytest.fixture(scope="session")
def setup_reconstructed_data_large_npy(setup_gdrive_src_data_large):
    """
    Loads AND Reconstructs SOURCE data.

    SIZE: 2048 x 2048
    TYPE: .npy

    :param setup_gdrive_src_data_large:
    :return: physical data object containing reconstructed data
    """
    bg_dat, sm_dat = setup_gdrive_src_data_large

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


@pytest.fixture(scope="session")
def setup_reconstructed_data_small_npy(setup_gdrive_src_data_small):
    """
    Loads AND Reconstructs SOURCE data.

    SIZE: 512 x 512
    TYPE: .npy

    :param setup_gdrive_src_data_small:
    :return: physical data object containing reconstructed data
    """
    bg_dat, sm_dat = setup_gdrive_src_data_small

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


# ==================================================================================================
# ================================== MULTI-DIM complete test =======================================
# ==================================================================================================


@pytest.fixture(scope="session")
def setup_multidim_src():
    """
    Downloads and unzips a folder containing MULTIDIM SOURCE data

    Microglia coculture with neurons and progenitor cells
    acquired on micro-manager 1.4.22

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

    :return: str
        path to configuration file (.yml)
    """

    temp_folder = os.getcwd() + '/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder "+temp_folder)
    if not os.path.isdir(temp_folder+'/src'):
        os.mkdir(temp_folder+'/src')
        print("\nsetting up src folder "+temp_folder+'/src')
    if not os.path.isdir(temp_folder+'/predict'):
        os.mkdir(temp_folder+'/predict')
        print("\nsetting up predict folder "+temp_folder+'/predict')

    # DO NOT ADJUST THESE VALUES
    bulk_file = '1v0Nb_puvttbe31z-poPdkGL1PxFiiqqz'

    srczip = temp_folder + '/src' + '/src_zip.zip'
    gdd.download_file_from_google_drive(file_id=bulk_file,
                                        dest_path=srczip,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    # read and modify downloaded .yml if necessary here
    config_file = [file for file in os.listdir(temp_folder+'/src') if '.yml' in file]

    yield temp_folder + '/src/' + config_file[0]

    # breakdown files
    import shutil
    print("breaking down temp files in "+temp_folder)
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)


@pytest.fixture(scope="session")
def setup_multidim_target():
    """
    Same as SOURCE data but reconstructed using standard ReconstructOrder pipeline

    :return: str
        path to target .tif files
    """

    temp_folder = os.getcwd() + '/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")
    if not os.path.isdir(temp_folder + '/target'):
        os.mkdir(temp_folder + '/target')

    # DO NOT ADJUST THESE VALUES
    bulk_file = '1aVIwx-qADT0adMv7XWlLHvXxgfjGW_V-'

    output = temp_folder + '/target' + '/target_zip.zip'
    gdd.download_file_from_google_drive(file_id=bulk_file,
                                        dest_path=output,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    yield temp_folder + '/target'

    # this breakdown will destroy the temp_folder, which may contain fixtures for other tests
    # breakdown files
    # import shutil
    # # print("breaking down temp files in "+temp_folder)
    # # if os.path.isdir(temp_folder):
    # #     shutil.rmtree(temp_folder)
