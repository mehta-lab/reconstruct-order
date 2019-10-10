import numpy as np
import pytest, os
from numpy.testing import assert_array_equal
from ReconstructOrder.datastructures.physical_data import PhysicalData


def test_basic_constructor_nparray():
    """
    test assignment using numpy arrays
    """

    phys = PhysicalData()

    phys.I_trans = np.ones((512, 512))
    phys.polarization = 2 * np.ones((512, 512))
    phys.retard = 3 * np.ones((512, 512))
    phys.depolarization = 4 * np.ones((512, 512))
    phys.azimuth = 5 * np.ones((512, 512))
    phys.azimuth_degree = 6 * np.ones((512, 512))
    phys.azimuth_vector = 7 * np.ones((512, 512))

    assert_array_equal(phys.I_trans, np.ones((512, 512)))
    assert_array_equal(phys.polarization, 2*np.ones((512, 512)))
    assert_array_equal(phys.retard, 3*np.ones((512, 512)))
    assert_array_equal(phys.depolarization, 4*np.ones((512, 512)))
    assert_array_equal(phys.azimuth, 5*np.ones((512, 512)))
    assert_array_equal(phys.azimuth_degree, 6*np.ones((512, 512)))
    assert_array_equal(phys.azimuth_vector, 7*np.ones((512, 512)))


def test_basic_constructor_memap(setup_temp_data):
    """
    test assignment using memory mapped files
    """

    mm = setup_temp_data
    phys = PhysicalData()

    phys.I_trans = mm
    phys.polarization = 2 * mm
    phys.retard = 3 * mm
    phys.depolarization = 4 * mm
    phys.azimuth = 5 * mm
    phys.azimuth_degree = 6 * mm
    phys.azimuth_vector = 7 * mm

    assert_array_equal(phys.I_trans, mm)
    assert_array_equal(phys.polarization, 2*mm)
    assert_array_equal(phys.retard, 3*mm)
    assert_array_equal(phys.depolarization, 4*mm)
    assert_array_equal(phys.azimuth, 5*mm)
    assert_array_equal(phys.azimuth_degree, 6*mm)
    assert_array_equal(phys.azimuth_vector, 7*mm)


def test_instances():
    """
    test instance attributes
    """
    phs1 = PhysicalData()
    phs2 = PhysicalData()

    with pytest.raises(AssertionError):
        assert(phs1 == phs2)

    with pytest.raises(AssertionError):
        phs1.retard = 1
        phs2.retard = 2
        assert(phs1.retard == phs2.retard)


def test_private_access(setup_physical_data):
    """
    test that private attributes are not accessible
    """
    phys = setup_physical_data
    with pytest.raises(AttributeError):
        print(phys.__I_trans)
        print(phys.__retard)


# ==== Attribute assignment ==========

def test_assignment(setup_physical_data):
    """
    test exception handling of improper assignment
    """
    phys = setup_physical_data
    with pytest.raises(TypeError):
        phys.incorrect_attribute = 1