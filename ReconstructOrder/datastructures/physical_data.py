import numpy as np


class PhysicalData(object):
    """
    Data Structure that contains all computed physical data
    only attributes with getters/setters can be assigned to this class
    """

    __I_trans = None
    __retard = None
    __polarization = None
    __depolarization = None
    __azimuth = None
    __azimuth_vector = None
    __azimuth_degree = None

    def __setattr__(self, name, value):
        """
        Prevent attribute assignment other than those defined below
        :param name: any attribute
        :param value: corresponding value
        :return:
        """
        if hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise TypeError('Cannot set name %r on object of type %s' % (
                name, self.__class__.__name__))

    def __init__(self):
        """
        Initialize instance variables.
        """
        super(PhysicalData, self).__init__()
        self.__I_trans = None
        self.__retard = None
        self.__polarization = None
        self.__depolarization = None
        self.__azimuth = None
        self.__azimuth_vector = None
        self.__azimuth_degree = None

    @property
    def I_trans(self):
        return self.__I_trans

    @I_trans.setter
    def I_trans(self, data: np.ndarray):
        self.__I_trans = data

    @property
    def retard(self):
        return self.__retard

    @retard.setter
    def retard(self, data: np.ndarray):
        self.__retard = data

    @property
    def polarization(self):
        return self.__polarization

    @polarization.setter
    def polarization(self, data: np.ndarray):
        self.__polarization = data

    @property
    def azimuth(self):
        return self.__azimuth

    @azimuth.setter
    def azimuth(self, data: np.ndarray):
        self.__azimuth = data

    @property
    def azimuth_vector(self):
        return self.__azimuth_vector

    @azimuth_vector.setter
    def azimuth_vector(self, data: np.ndarray):
        self.__azimuth_vector = data

    @property
    def azimuth_degree(self):
        return self.__azimuth_degree

    @azimuth_degree.setter
    def azimuth_degree(self, data: np.ndarray):
        self.__azimuth_degree = data

    @property
    def depolarization(self):
        return self.__depolarization

    @depolarization.setter
    def depolarization(self, data: np.ndarray):
        self.__depolarization = data


