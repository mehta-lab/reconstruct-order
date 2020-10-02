from ReconstructOrder.utils.mManagerIO import mManagerReader, PolAcquReader
from ..datastructures import IntensityDataCreator, IntensityData
from .ConfigReader import ConfigReader
from .imgProcessing import ImgMin
from .aux_utils import loop_pt
from typing import Union
import numpy as np
import cv2


class FlatFieldCorrector(object):
    """Compute illumination function of fluorescence channels
        for flat-field correction

     Parameters
    ----------
    img_io: object
        mManagerReader object that holds the image parameters
    config: object
        ConfigReader object that holds the user input config parameters
    img_reconstructor: ImgReconstructor
        ImgReconstructor object for image reconstruction
    background_data: StokesData
        object of type StokesData

    Attributes
    ----------
    ff_method : str
        flat-field correction method. Using morphological opening if 'open' and empty image if 'empty'
    img_fluor_min : array
        array of empty flourescence images for each channel
    img_fluor_sum : array
        array of sum of morphologically opened flourescence images for each channel
    kernel : obj
      kernel for image opening operation
    img_fluor_bg : array
        Fluorescence channel background for flat-field correction
    binning : int
        binning (or pooling) size for the images
    """

    def __init__(self,
                 img_io: Union[mManagerReader, PolAcquReader],
                 config: ConfigReader,
                 method='open'):
        self.img_io = img_io
        self.binning = config.processing.binning
        if config.dataset.ROI is None:
            self.height, self.width = img_io.height, img_io.width
        else:
            self.height, self.width = config.dataset.ROI[2], config.dataset.ROI[3]
        self.img_fluor_min = np.full((5, img_io.height // img_io.binning, img_io.width // img_io.binning),
                                     np.inf)  # set initial min array to to be Inf
        self.img_fluor_sum = np.zeros(
            (5, img_io.height // img_io.binning,
             img_io.width // img_io.binning))  # set the default background to be Ones (uniform field)
        assert method in ['open', 'empty'], "flat-field correction method must be 'open' or 'empty'"
        self.method = method
        self.img_int_creator = IntensityDataCreator(ROI=config.dataset.ROI, binning=self.binning)
        self._fluor_chan_names = ['405', '488', '568', '640', 'ex561em700']

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (100,
                                                 100))  # kernel for image opening operation, 100-200 is usually good
        self.img_fluor_bg = IntensityData(channel_names=self._fluor_chan_names)
        for chan_name in self.img_fluor_bg.channel_names:
            self.img_fluor_bg.replace_image(
                np.ones((self.height // self.binning, self.width // self.binning)), chan_name)

    def compute_flat_field(self):
        """
        Compute illumination function of fluorescence channels
        for flat-field correction. Computes the illumination function
        of fluorescence channels using image opening or looking for empty images,
        currently only process the first Z for speed

        """
        if self.method == 'open':
            for chan_name in self.img_fluor_bg.channel_names:
                self.img_fluor_bg.replace_image(
                    np.zeros(self.height // self.binning, self.width // self.binning), chan_name)
        elif self.method == 'empty':
            for chan_name in self.img_fluor_bg.channel_names:
                self.img_fluor_bg.replace_image(
                    np.full((self.height // self.binning, self.width // self.binning), np.inf), chan_name)

        print('Calculating illumination function for flatfield correction...')
        self._compute_ff_helper(img_io=self.img_io)

        for chan_name in self.img_fluor_bg.channel_names:
            img_bg_new = self.img_fluor_bg.get_image(chan_name)
            img_bg_new = img_bg_new - min(np.nanmin(img_bg_new), 0) + 1 #add 1 to avoid 0
            img_bg_new /= np.mean(img_bg_new)  # normalize the background to have mean = 1
            self.img_fluor_bg.replace_image(img_bg_new, chan_name)

    @loop_pt
    def _compute_ff_helper(self,
                           config: ConfigReader,
                           img_io: Union[mManagerReader, PolAcquReader]=None,
                           ):

        for z_idx in range(0, 1):  # only use the first z
            img_io.z_idx = z_idx
            img_int_raw = self.img_int_creator.get_data_object(config, img_io)
            for chan_idx, chan_name in enumerate(self._fluor_chan_names):
                img = img_int_raw.get_image(chan_name)
                if np.any(img):  # if the flour channel exists
                    img_bg_new = self.img_fluor_bg.get_image(chan_name)
                    if self.method == 'open':
                        img_bg_new += \
                            cv2.morphologyEx(img, cv2.MORPH_OPEN, self.kernel, borderType=cv2.BORDER_REPLICATE)
                    elif self.method == 'empty':
                        img_bg_new = \
                            ImgMin(img, img_bg_new)
                    self.img_fluor_bg.replace_image(img_bg_new, chan_name)

    def correct_flat_field(self, img_int_sm: IntensityData) -> IntensityData:
        """
        flat-field correction for fluorescence channels
        Parameters
        ----------
        img_int_sm : IntensityData
            stack of fluorescence images with with shape (channel, y, x)

        Returns
        -------
        img_int_sm : IntensityData
            flat-field corrected fluorescence images
        """

        for chan_name in self._fluor_chan_names:
            img = img_int_sm.get_image(chan_name)
            if np.any(img):  # if the flour channel exists
                img_int_sm.replace_image(img / self.img_fluor_bg.get_image(chan_name),
                                         chan_name)
        return img_int_sm
