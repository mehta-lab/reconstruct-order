import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ReconstructOrder.datastructures.stokes_data import StokesData

# ===== test construction =====

def test_naked_constructor():
    """
    test simple construction
    """
    stk = StokesData()

    assert(stk.s0 is None)
    assert(stk.s1 is None)
    assert(stk.s2 is None)
    assert(stk.s3 is None)


def test_basic_constructor_nparray():
    """
    test assignment using numpy arrays
    """
    stk = StokesData()

    a = np.ones((512, 512))
    b = 2*np.ones((512, 512))
    c = 3*np.ones((512, 512))
    d = 4*np.ones((512, 512))

    stk.s0 = a
    stk.s1 = b
    stk.s2 = c
    stk.s3 = d

    assert_array_equal(stk.s0, a)
    assert_array_equal(stk.s1, b)
    assert_array_equal(stk.s2, c)
    assert_array_equal(stk.s3, d)

    assert_array_equal(stk.data, np.array([a, b, c, d]))


def test_stokes_constructor_nparray(setup_intensity_data, setup_inst_matrix):
    """
    check that auto-construction assigns to instance, not class
    """
    # this intensity data has frames = 5
    int_data, _, _, _, _, _,  = setup_intensity_data

    # create default inv inst matrix
    iim = setup_inst_matrix

    stk_data_unassigned = StokesData()
    stk_data = StokesData(inv_inst_matrix=iim, intensity_data=int_data)

    assert(stk_data.s0 is not None)
    assert(stk_data.s1 is not None)
    assert(stk_data.s2 is not None)
    assert(stk_data.s3 is not None)
    assert(stk_data.s1_norm is not None)
    assert(stk_data.s2_norm is not None)
    assert(stk_data.data is not None)

    assert(stk_data_unassigned.s0 is None)
    assert(stk_data_unassigned.s1 is None)
    assert(stk_data_unassigned.s2 is None)
    assert(stk_data_unassigned.s3 is None)
    assert(stk_data_unassigned.s1_norm is None)
    assert(stk_data_unassigned.s2_norm is None)


def test_stokes_assignment_after_construction(setup_intensity_data, setup_inst_matrix):
    """
    test that one can compute stokes after class construction
    """
    # this intensity data has frames = 5
    int_data, _, _, _, _, _,  = setup_intensity_data
    iim = setup_inst_matrix
    stk_data = StokesData()

    # normal construction is fine
    assert (stk_data.s0 is None)
    assert (stk_data.s1 is None)
    assert (stk_data.s2 is None)
    assert (stk_data.s3 is None)
    assert (stk_data.s1_norm is None)
    assert (stk_data.s2_norm is None)

    # calculate stokes and reassign
    stk_data.compute_stokes(inv_inst_matrix=iim, intensity_data=int_data)

    assert(stk_data.s0 is not None)
    assert(stk_data.s1 is not None)
    assert(stk_data.s2 is not None)
    assert(stk_data.s3 is not None)
    assert(stk_data.s1_norm is not None)
    assert(stk_data.s2_norm is not None)
    assert(stk_data.data is not None)


def test_stokes_constructor_exception(setup_intensity_data):
    """
    test that instrument-matrix and intensity data shape mismatches are caught
    """
    # this intensity data has frames = 5
    int_data, _, _, _, _, _,  = setup_intensity_data

    # create incorrect inst matrix
    chi = 0.03*2*np.pi                    # if the images were taken using 5-frame scheme
    inst_mat = np.array([[1, 0, 0, -1],
                         [1, 0, np.sin(chi), -np.cos(chi)],
                         [1, -np.sin(chi), 0, -np.cos(chi)],
                         [1, 0, -np.sin(chi), -np.cos(chi)]])

    iim = np.linalg.pinv(inst_mat)
    with pytest.raises(ValueError):
        StokesData(inv_inst_matrix=iim, intensity_data=int_data)


def test_basic_constructor_memap(setup_temp_data):
    """
    test assignment using memory mapped files
    """

    mm = setup_temp_data
    stk = StokesData()

    stk.s0 = mm
    stk.s1 = 2*mm
    stk.s2 = 3*mm
    stk.s3 = 4*mm

    assert_array_equal(stk.s0, mm)
    assert_array_equal(stk.s1, 2*mm)
    assert_array_equal(stk.s2, 3*mm)
    assert_array_equal(stk.s3, 4*mm)

    assert_array_equal(stk.data, np.array([mm, 2*mm, 3*mm, 4*mm]))

# ===== test instances and access =====


def test_instances():
    """
    test instance attributes
    """
    stk1 = StokesData()
    stk2 = StokesData()

    with pytest.raises(AssertionError):
        assert(stk1 == stk2)

    with pytest.raises(AssertionError):
        stk1.s0 = np.ones((512,512))
        stk2.s0 = 2*np.ones((512,512))
        assert_array_equal(stk1.s0, stk2.s0)


def test_private_access(setup_stokes_data):
    """
    test that private attributes are not accessible
    """
    stk_data = setup_stokes_data
    with pytest.raises(AttributeError):
        print(stk_data.__s0)
        print(stk_data.__s1)


# ===== test dimensionality =====

def test_data_dims():
    """
    test that data dimensionality mismatches are caught
    """
    stk_data = StokesData()

    a = np.ones((512, 512))
    b = np.ones((256, 256))

    stk_data.s0 = a
    stk_data.s1 = a
    stk_data.s2 = a

    # None type check
    with pytest.raises(ValueError):
        dat = stk_data.data

    stk_data.s3 = a

    # data dim check
    with pytest.raises(ValueError):
        stk_data.s2 = b
        dat = stk_data.data


# ==== Attribute assignment ==========

def test_assignment(setup_stokes_data):
    """
    test exception handling of improper assignment
    """
    stk = setup_stokes_data
    with pytest.raises(TypeError):
        stk.s4 = 10*np.ones((512, 512))