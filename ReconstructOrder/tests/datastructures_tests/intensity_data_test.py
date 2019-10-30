import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ReconstructOrder.datastructures.intensity_data import IntensityData

# ==== test basic construction =====


def test_basic_constructor_nparray():
    """
    test assignment using numpy arrays
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

    assert_array_equal(int_data.get_image(0), a)
    assert_array_equal(int_data.get_image(1), b)
    assert_array_equal(int_data.get_image(2), c)
    assert_array_equal(int_data.get_image(3), d)
    assert_array_equal(int_data.get_image(4), e)

    assert_array_equal(int_data.data, np.array([a, b, c, d, e]))


def test_basic_constructor_memap(setup_temp_data):
    """
    test assignment using memory mapped files
    """

    mm = setup_temp_data
    int_data = IntensityData()

    int_data.append_image(mm)
    int_data.append_image(2 * mm)
    int_data.append_image(3 * mm)
    int_data.append_image(4 * mm)
    int_data.append_image(5 * mm)

    assert_array_equal(int_data.get_image(0), mm)
    assert_array_equal(int_data.get_image(1), 2*mm)
    assert_array_equal(int_data.get_image(2), 3*mm)
    assert_array_equal(int_data.get_image(3), 4*mm)
    assert_array_equal(int_data.get_image(4), 5*mm)

    assert_array_equal(int_data.data, np.array([mm, 2*mm, 3*mm, 4*mm, 5*mm]))



def test_basic_constructor_with_names():
    """
    test construction with channel names

    Returns
    -------

    """
    int_data = IntensityData()
    int_data.channel_names = ['IExt', 'I0', 'I45', 'I90', 'I135']

    a = np.ones((512, 512))
    b = 2 * np.ones((512, 512))
    c = 3 * np.ones((512, 512))
    d = 4 * np.ones((512, 512))
    e = 5 * np.ones((512, 512))

    int_data.replace_image(a, 'IExt')
    int_data.replace_image(b, 'I0')
    int_data.replace_image(c, 'I45')
    int_data.replace_image(d, 'I90')
    int_data.replace_image(e, 'I135')

    assert_array_equal(int_data.get_image("IExt"), a)


def test_basic_constructor_without_names():
    """
    test construction with channel names

    Returns
    -------

    """
    int_data = IntensityData()
    # int_data.channel_names = ['IExt', 'I0', 'I45', 'I90', 'I135']

    a = np.ones((512, 512))
    b = 2 * np.ones((512, 512))
    c = 3 * np.ones((512, 512))
    d = 4 * np.ones((512, 512))
    e = 5 * np.ones((512, 512))

    int_data.append_image(a)
    int_data.append_image(b)
    int_data.append_image(c)
    int_data.append_image(d)
    int_data.append_image(e)

    assert_array_equal(int_data.get_image(0), a)


# ==== test instances and private/public access =====

def test_instances():
    """
    test instance attributes
    """
    I1 = IntensityData()
    I2 = IntensityData()

    with pytest.raises(AssertionError):
        assert(I1 == I2)

    with pytest.raises(AssertionError):
        I1.append_image(np.ones((32, 32)))
        I2.append_image(np.ones((64, 64)))
        assert_array_equal(I1.get_image(0),I2.get_image(0))


def test_private_access(setup_intensity_data):
    """
    should not have access to private variables
    access is restricted to setters/getters
    """
    int_data, a, b, c, d, e = setup_intensity_data
    with pytest.raises(AttributeError):
        print(int_data.__IExt)
    with pytest.raises(AttributeError):
        print(int_data.__I0)


# ==== test methods =====


# replace_image method
def test_replace_image_shape(setup_intensity_data):
    int_data, a, b, c, d, e = setup_intensity_data

    newim = np.ones((5,5))
    with pytest.raises(ValueError):
        int_data.replace_image(newim, 0)


def test_replace_image_dtype(setup_intensity_data):
    int_data, a, b, c, d, e = setup_intensity_data

    newim = 0
    with pytest.raises(TypeError):
        int_data.replace_image(newim, 0)



def test_replace_image_by_index(setup_intensity_data):
    int_data, a, b, c, d, e = setup_intensity_data

    newim = np.ones((512, 512))
    int_data.replace_image(newim, 0)
    assert_array_equal(int_data.data[0], newim)



def test_replace_image_by_string(setup_intensity_data):
    int_data, a, b, c, d, e = setup_intensity_data

    int_data.channel_names = ['IExt', 'I0', 'I45', 'I90', 'I135']

    newim = np.ones((512,512))
    int_data.replace_image(newim, 'I90')
    assert_array_equal(int_data.get_image('I90'), newim)


# channel_names property
def test_channel_names(setup_intensity_data):
    int_data, a, b, c, d, e = setup_intensity_data

    names = ['a','b','c','d','e']

    int_data.channel_names = names



# get_image method
def test_get_image_str(setup_intensity_data):
    """
    test query by string channel name
    """
    int_data, a, b, c, d, e = setup_intensity_data

    names = ['a','b','c','d','e']
    int_data.channel_names = names

    dat = int_data.get_image('e')
    assert(dat.shape, (512,512))
    assert(dat[0][0], 5)


def test_get_img_str_undef(setup_intensity_data):
    """
    test exception handling of query by string channel name
    """
    int_data, a, b, c, d, e = setup_intensity_data

    names = ['a','b','c','d','e','f','g','h']
    int_data.channel_names = names

    with pytest.raises(ValueError):
        dat = int_data.get_image('q')


def test_get_image_int(setup_intensity_data):
    """
    test query by int channel index
    """
    int_data, a, b, c, d, e = setup_intensity_data
    names = ['a','b','c','d','e']
    int_data.channel_names = names

    dat = int_data.get_image(4)
    assert(dat.shape, (512,512))
    assert(dat[0][0], 5)



# axis_names property
def test_axis_names(setup_intensity_data):
    int_data, a, b, c, d, e = setup_intensity_data
    names = ['c', 'x', 'y', 'z', 't']
    int_data.axis_names = names

    assert(int_data.axis_names, names)


# ==== test data dimensions =====

def test_ndims_1(setup_ndarrays):
    """
    test that shape is preserved
    """
    p, q, r = setup_ndarrays
    int_data = IntensityData()

    int_data.append_image(p)
    int_data.append_image(p)
    int_data.append_image(p)

    assert(int_data.data[0].shape == p.shape)
    assert(int_data.data.shape == (3,)+p.shape)


def test_ndims_2(setup_ndarrays):
    """
    test exception handling for image data that is not \
    numpy array or numpy memmap
    """
    int_data = IntensityData()

    with pytest.raises(TypeError):
        int_data.append_image(1)
    with pytest.raises(TypeError):
        int_data.append_image([1, 2, 3])
    with pytest.raises(TypeError):
        int_data.append_image({1, 2, 3})
    with pytest.raises(TypeError):
        int_data.append_image((1, 2, 3))


def test_ndims_3(setup_ndarrays):
    """
    test exception handling upon assignment of dim mismatch image
    """
    p, q, r = setup_ndarrays
    int_data = IntensityData()

    int_data.append_image(p)

    with pytest.raises(ValueError):
        int_data.append_image(q)


# ==== Attribute assignment ==========

def test_assignment(setup_intensity_data):
    """
    test exception handling of improper assignment
    """
    int_data, a, b, c, d, e = setup_intensity_data
    with pytest.raises(TypeError):
        int_data.Iext = a
    with pytest.raises(TypeError):
        int_data.__IExt = a


def test_set_data(setup_intensity_data):
    """
    test that neither data nor frames are set-able attributes
    """
    int_data, a, b, c, d, e = setup_intensity_data
    with pytest.raises(AttributeError):
        int_data.data = 0
    with pytest.raises(AttributeError):
        int_data.num_channels = 0
