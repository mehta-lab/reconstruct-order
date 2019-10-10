import numpy as np


class IntensityData(object):
    """
    Data Structure that contains all raw intensity images used for computing Stokes matrices
    only attributes with getters/setters can be assigned to this class
    """

    __data = None
    __channel_names = None
    __axis_names = None
    __current_shape = None

    def __setattr__(self, name, value):
        """
        Prevent attribute assignment other than those defined below

        Parameters
        ----------
        name : str
        value : value

        Returns
        -------

        """
        if hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise TypeError('Cannot set name %r on object of type %s' % (
                name, self.__class__.__name__))

    def __init__(self, num_channels=None, channel_names: list = None, axis_names: list = None):
        """
        Initialize instance variables

        Parameters
        ----------
        num_channels : int
        channel_names : list of str
        axis_names : list of str
        """

        super(IntensityData, self).__init__()
        if num_channels and type(num_channels) != int:
            raise ValueError("number of channels must be integer type")

        self.__data = []
        if channel_names:
            self.channel_names = channel_names
        if axis_names:
            self.axis_names = axis_names
        if num_channels:
            for _ in range(num_channels):
                self.__data.append([])
        self.__current_shape = ()

    def check_shape(self, input_shape=None):
        """
        compare the supplied input_shape with the shape of images already in the self.__data
        self.__current_shape is updated at "append_image" and "replace_image"

        Parameters
        ----------
        input_shape : tuple
        supplied shape from image to add to self.__data

        Returns
        -------

        """

        # check for empty __data possibilities:
        if len(self.__data) == 0:
            return True
        if self.__current_shape == ():
            return True

        # check for shape consistency
        if input_shape and input_shape != self.__current_shape:
            return False

        return True

    def check_dtype(self, input_data):
        """
        check that supplied images are either numpy arrays or memmaps

        Parameters
        ----------
        input_data : np.ndarray, or np.memmap

        Returns
        -------
        boolean

        """

        if type(input_data) is not np.ndarray and \
                type(input_data) is not np.memmap:
            return False
        else:
            return True

    @property
    def data(self):
        """
        get the underlying np.array data that is built

        Returns
        -------

        np.array of data object

        """
        if not self.check_shape():
            raise ValueError("Inconsistent data dimensions or data not assigned\n")
        return np.array(self.__data)

    @property
    def num_channels(self):
        """
        Get the number of channels already assigned to the data

        Returns
        -------
        number of images assigned to this data

        """

        return len(self.__data)

    @property
    def channel_names(self):
        return self.__channel_names

    @channel_names.setter
    def channel_names(self, names: list):
        """
        assign channel names to images in data

        Parameters
        ----------
        names : list of str

        Returns
        -------

        """
        for chan in names:
            if type(chan) != str:
                raise ValueError("channel names must be a list of strings")

        self.__channel_names = names

        # check that data contains entries for each of the channel names
        # if it does not, then add a blank entry
        if len(self.__data) <= len(self.__channel_names):
            for i in range(len(self.__channel_names)-len(self.__data)):
                self.__data.append([])

    @property
    def axis_names(self):
        return self.__channel_names

    @axis_names.setter
    def axis_names(self, ax_names: list):
        """
        set names for axes

        Parameters
        ----------
        value : list of str

        Returns
        -------

        """
        for axis in ax_names:
            if type(axis) != str:
                raise ValueError("axis names must be a list of strings")

        if len(set(ax_names)) != len(ax_names):
            raise ValueError("duplicate entries in axis_name list")
        else:
            self.__axis_names = ax_names

    def append_image(self, image):
        """
        append image to end of data

        Parameters
        ----------
        image : np.ndarray or np.memmap

        Returns
        -------

        """
        if not self.check_dtype(image):
            raise TypeError("image is not ndarray")
        if not self.check_shape(image.shape):
            raise ValueError("image does not conform to current data dimensions")
        if self.__channel_names:
            raise ValueError("channel names are already defined for this IntensityData object."
                             "Append first, then assign channel names.  Or use replace_image instead")

        self.__current_shape = image.shape
        self.__data.append(image)

    def replace_image(self, image, value):
        """
        replace image in self.__data at supplied index.
        Index can be string (channel name), or int (position)

        Parameters
        ----------
        image : np.ndarray
        value : str or int

        Returns
        -------

        """

        # data checks
        if not self.check_dtype(image):
            raise TypeError("image is not ndarray")
        if not self.check_shape(image.shape):
            raise ValueError("image does not conform to current data dimensions")
        self.__current_shape = image.shape

        # find the position in the array, check that it exists
        if type(value) == int:
            if len(self.__data) <= value:
                raise IndexError("replacing Intensity Data image at position that does not exist")
            position = value
        elif type(value) == str:
            if value not in self.__channel_names:
                raise IndexError("replacing Intensity Data image at channel name that is not defined")
            position = self.__channel_names.index(value)
        else:
            raise ValueError("index or channel name in data does not exist or not defined")

        # replace image
        self.__data[position] = image

    def get_image(self, position):
        """
        enable image search by channel name or index

        Parameters
        ----------
        position : int or str
                if str, search for matching str in supplied channel_names
        Returns
        -------

        """
        if type(position) is str:
            if position in self.__channel_names:
                try:
                    dat = self.__data[self.__channel_names.index(position)]
                except TypeError:
                    raise TypeError("channel %s does not exist in data" % position)
                return dat
            else:
                raise ValueError("Intensity Data with channel name %s is not found")
        elif type(position) is int:
            return self.__data[position]
