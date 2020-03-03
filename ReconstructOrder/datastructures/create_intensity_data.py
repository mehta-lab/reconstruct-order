import numpy as np
import warnings

from ReconstructOrder.datastructures import IntensityData
from ..utils.mManagerIO import mManagerReader
from ..utils.imgProcessing import mean_pooling_2d


_fluor_chan_names = ['405', '488', '568', '640', 'ex561em700']

class IntensityDataCreator(object):
    """Create IntensityData objects from images in mManager/Polacquisition data format
    Parameters
    ----------
    ROI    : list
        region of interest for reconstruction in format of [n_start_y, n_start_x, Ny, Nx]
    input_chan : list
        list of input channel names, subset of same as the img_io.input_chans
    int_obj_chans : list
        Channels in the output intensity data object
    binning : int
        binning (or pooling) size for the images
    """

    def __init__(self, input_chans=None, int_obj_chans=None, ROI=None, binning=1):
        self.input_chans = input_chans
        self.roi = ROI
        self.binning = binning
        self.int_obj_chans = ['IExt', 'I90', 'I135', 'I45', 'I0', 'BF',
                         '405', '488', '568', '640', 'ex561em700']
        if int_obj_chans is not None:
            self.int_obj_chans = int_obj_chans
            
    def get_data_object(self, img_io: mManagerReader) -> IntensityData:
        """Parse tiff file name following mManager/Polacquisition output format
        return intensity data objects with images assigned to corresponding channels
        based on the file name
        Parameters
        ----------
        img_io : obj
            mManagerReader instance
        Returns
        -------
        imgs : IntensityData
            images from polarization, fluorescence, bright-field channels
        """

        imgs = IntensityData(channel_names=self.int_obj_chans)
        
        if self.roi is None:
            self.roi = [0, 0, img_io.height, img_io.width]
        
        for chan_name in _fluor_chan_names:
            imgs.replace_image(np.zeros((self.roi[2], self.roi[3])), chan_name)
        
        assert self.roi[0] + self.roi[2] <= img_io.height and self.roi[1] + self.roi[3] <= img_io.width, \
            "Region of interest is beyond the size of the actual image"

        if self.input_chans is None:
            self.input_chans = img_io.input_chans
        for chan_name in self.input_chans:
            img_io.chan_idx = img_io.input_chans.index(chan_name)
            img = img_io.read_img()
            if img is None:
                warnings.warn('image "{}" cannot be found. Skipped.'.format(chan_name))
            else:
                img = img[self.roi[0]:self.roi[0] + self.roi[2], self.roi[1]:self.roi[1] + self.roi[3]]
                img -= img_io.blackLevel
                img = mean_pooling_2d(img, self.binning)
                imgs = IntensityDataCreator.chan_name_parser(imgs, img, chan_name)
        return imgs
    
    
        


    @staticmethod
    def chan_name_parser(imgs, img, chan_name):
        """Parse the image channel name and assign the image to
        the channel in the intensity data object

        Parameters
        ----------
        imgs : IntensityData
            intensity data object
        img : image to assign
        chan_name :
            image channel name
        Returns
        -------
        imgs : IntensityData
            images from polarization, fluorescence, bright-field channels
        """
        if any(substring in chan_name for substring in ['State', 'state', 'Pol']):
            if '0' in chan_name:
                imgs.replace_image(img, 'IExt')
            elif '1' in chan_name:
                imgs.replace_image(img, 'I90')
            elif '2' in chan_name:
                imgs.replace_image(img, 'I135')
            elif '3' in chan_name:
                imgs.replace_image(img, 'I45')
            elif '4' in chan_name:
                imgs.replace_image(img, 'I0')
        elif any(substring in chan_name for substring in
                 ['Confocal40', 'Confocal_40', 'Widefield', 'widefield', 'Fluor']):
            if any(substring in chan_name for substring in ['DAPI', '405', '405nm']):
                imgs.replace_image(img, '405')
            elif any(substring in chan_name for substring in ['GFP', '488', '488nm']):
                imgs.replace_image(img, '488')
            elif any(substring in chan_name for substring in
                     ['TxR', 'TXR', 'TX', '568', '561', '560']):
                imgs.replace_image(img, '568')
            elif any(substring in chan_name for substring in ['Cy5', 'IFP', '640', '637']):
                imgs.replace_image(img, '640')
            elif any(substring in chan_name for substring in ['FM464', 'fm464']):
                imgs.replace_image(img, 'ex561em700')
        elif any(substring in chan_name for substring in ['BF', 'BrightField']):
            imgs.replace_image(img, 'BF')

        return imgs