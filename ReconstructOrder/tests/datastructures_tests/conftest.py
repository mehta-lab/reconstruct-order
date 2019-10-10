import numpy as np
import pytest
import os
from ReconstructOrder.datastructures.intensity_data import IntensityData
from ReconstructOrder.datastructures.stokes_data import StokesData
from ReconstructOrder.datastructures.physical_data import PhysicalData


"""
pytest fixtures are "setup" code that makes resources available for tests
pass a keyword scope = "session", or scope = "module" if the fixture would not
be re-created for each test.
"""


@pytest.fixture
def setup_temp_data():
    """
    resource for memory mapped files

    :return:
    """
    temp_folder = os.getcwd()+'/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # setup files, provide memmap type
    img = np.random.random((2048, 2048))
    np.save(temp_folder+'/img.npy', img)
    print("setting up temp file")
    yield np.memmap(temp_folder+'/img.npy', shape=(2048, 2048), dtype=np.uint8)

    # breakdown files
    if os.path.isfile(temp_folder+'/img.npy'):
        os.remove(temp_folder+'/img.npy')
        print("\nbreaking down temp file")
    if os.path.isdir(temp_folder):
        os.rmdir(temp_folder)
        print("breaking down temp folder")


@pytest.fixture
def setup_temp_int_stk_phys(setup_temp_data):
    # reference data is a memory mapped file
    mm = setup_temp_data

    int_data = IntensityData()
    int_data.append_image(mm)
    int_data.append_image(2 * mm)
    int_data.append_image(3 * mm)
    int_data.append_image(4 * mm)
    int_data.append_image(5 * mm)

    stk_data = StokesData()
    stk_data.s0 = 10*mm
    stk_data.s1 = 20*mm
    stk_data.s2 = 30*mm
    stk_data.s3 = 40*mm

    phys_data = PhysicalData()
    phys_data.I_trans = 100*mm
    phys_data.polarization = 200*mm
    phys_data.retard = 300*mm
    phys_data.depolarization = 400*mm
    phys_data.azimuth = 500*mm
    phys_data.azimuth_degree = 600*mm
    phys_data.azimuth_vector = 700*mm

    yield int_data, stk_data, phys_data


@pytest.fixture
def setup_int_stk_phys(setup_intensity_data, setup_stokes_data, setup_physical_data):
    intensity, _, _, _, _, _ = setup_intensity_data
    stokes = setup_stokes_data
    physical = setup_physical_data

    yield intensity, stokes, physical


@pytest.fixture
def setup_intensity_data():
    """
    resource for IntensityData, no channel names

    :return: Intensity Object, component arrays
    """
    int_data = IntensityData()

    a = np.ones((512, 512))
    b = 2*np.ones((512, 512))
    c = 3*np.ones((512, 512))
    d = 4*np.ones((512, 512))
    e = 5*np.ones((512, 512))

    int_data.append_image(a)
    int_data.append_image(b)
    int_data.append_image(c)
    int_data.append_image(d)
    int_data.append_image(e)

    yield int_data, a, b, c, d, e


@pytest.fixture
def setup_stokes_data():
    """
    resource for Stokes Data

    :return: Stokes Data Object
    """
    stk = StokesData()

    a = 10*np.ones((512, 512))
    b = 20*np.ones((512, 512))
    c = 30*np.ones((512, 512))
    d = 40*np.ones((512, 512))

    stk.s0 = a
    stk.s1 = b
    stk.s2 = c
    stk.s3 = d

    yield stk


@pytest.fixture
def setup_physical_data():
    """
    resource for PhysicalData

    :return: PhysicalData object
    """
    phys = PhysicalData()

    phys.I_trans = 100*np.ones((512, 512))
    phys.polarization = 200*np.ones((512, 512))
    phys.retard = 300*np.ones((512, 512))
    phys.depolarization = 400*np.ones((512, 512))
    phys.azimuth = 500*np.ones((512, 512))
    phys.azimuth_degree = 600*np.ones((512, 512))
    phys.azimuth_vector = 700*np.ones((512, 512))

    yield phys


@pytest.fixture
def setup_inst_matrix():
    """
    resource to create a 5-frame default instrument matrix

    :return: the INVERSE instrument matrix
    """
    chi = 0.03*2*np.pi                    # if the images were taken using 5-frame scheme
    inst_mat = np.array([[1, 0, 0, -1],
                         [1, np.sin(chi), 0, -np.cos(chi)],
                         [1, 0, np.sin(chi), -np.cos(chi)],
                         [1, -np.sin(chi), 0, -np.cos(chi)],
                         [1, 0, -np.sin(chi), -np.cos(chi)]])

    iim = np.linalg.pinv(inst_mat)
    yield iim


@pytest.fixture
def setup_ndarrays():
    """
    resource to create various numpy arrays of different dimensions

    :return:
    """
    p = np.ones(shape=(512,512,1,2))
    q = np.ones(shape=(32,32,5))
    r = np.zeros(shape=(32,32,5,2,3))

    yield p, q, r