import numpy as np
import cv2
from ..utils.background_estimator import BackgroundEstimator2D
from ..utils.ConfigReader import ConfigReader
from ..utils.imgProcessing import mean_pooling_2d_stack

from ..datastructures import IntensityData, StokesData, PhysicalData


from typing import Union


class ImgReconstructor:
    """
    ImgReconstructor contains methods to compute physical properties of birefringence
    given images collected with 4 or 5 polarization states

    Parameters
    ----------
    img_shape : tuple
        Shape of the input image (channel, y, x)
    bg_method : str
        "Global" or "Local". Type of background correction. "Global" will correct each image
         using the same background. "Local" will do correction with locally estimated
         background in addition to global background
    n_slice_local_bg : int
        Number of slices averaged for local background estimation
    swing : float
        swing of the elliptical polarization states in unit of fraction of wavelength
    wavelength : int
        wavelenhth of the illumination light (nm)
    kernel_size : int
        size of the Gaussian kernel for local background estimation
    poly_fit_order : int
        order of the polynomial used for 'Local_fit' background correction
    azimuth_offset : float
        offset of the orientation reference axis
    circularity : str
         ('lcp' or 'rcp') the circularity of the analyzer looking from the detector's point of view.
        Changing this flag will flip the slow axis horizontally.

    Attributes
    ----------
    img_shape : tuple
        Shape of the input image (channel, y, x)
    bg_method : str
        "Global" or "Local". Type of background correction. "Global" will correct each image
         using the same background. "Local" will do correction with locally estimated
         background in addition to global background
    n_slice_local_bg : int
        Number of slices averaged for local background estimation
    swing : float
        swing of the elliptical polarization states in unit of radian
    wavelength : int
        wavelenhth of the illumination light
    kernel_size : int
        size of the Gaussian kernel for local background estimation
    azimuth_offset : float
        offset of the orientation reference axis
    circularity : str
         ('lcp' or 'rcp') the circularity of the analyzer looking from the detector's point of view.
        Changing this flag will flip the slow axis horizontally.
    inst_mat_inv : 2d array
        inverse of the instrument matrix
    stokes_param_bg_tm :
        transformed global background Stokes parameters
    stokes_param_bg_local_tm :
        transformed local background Stokes parameters

    """

    def __init__(self,
                 int_obj: IntensityData,
                 config: ConfigReader,
                 bg_method        = 'Global',
                 n_slice_local_bg = 1,
                 swing            = None,
                 wavelength       = 532,
                 kernel_size      = 401,
                 poly_fit_order   = 2,
                 azimuth_offset   = 0,
                 circularity      = 'rcp',
                 use_gpu          = False,
                 gpu_id           = 0):

        # image params
        self._n_chann = 4
        if '5-State' in config.processing.calibration_scheme:
            self._n_chann = 5
        img_shape = int_obj.get_image('IExt').shape
        self.img_shape = [self._n_chann] + list(img_shape)
        self.bg_method        = bg_method
        self.n_slice_local_bg = n_slice_local_bg
        self.swing            = swing # covert swing from fraction of wavelength to radian
        self.wavelength       = wavelength
        self.kernel_size      = kernel_size
        self.poly_fit_order   = poly_fit_order
        self.use_gpu          = use_gpu
        self.gpu_id           = gpu_id
        
        if self.use_gpu:
            
            try:
                globals()['cp'] = __import__("cupy")
                cp.cuda.Device(self.gpu_id).use()
            except ModuleNotFoundError:
                print("cupy not installed, using CPU instead")
                self.use_gpu = False


        # compute instrument matrix only once

        if config.processing.calibration_scheme == '5-State' or config.processing.calibration_scheme == '4-State':
            chi = self.swing * 2 * np.pi
            if self._n_chann == 4:  # if the images were taken using 4-frame scheme (0, 45, 90, 135)
                inst_mat = np.array([[1, 0, 0, -1],
                                     [1, 0, np.sin(chi), -np.cos(chi)],
                                     [1, -np.sin(chi), 0, -np.cos(chi)],
                                     [1, 0, -np.sin(chi), -np.cos(chi)]])
            elif self._n_chann == 5:  # if the images were taken using 5-frame scheme
                inst_mat = np.array([[1, 0, 0, -1],
                                     [1, np.sin(chi), 0, -np.cos(chi)],
                                     [1, 0, np.sin(chi), -np.cos(chi)],
                                     [1, -np.sin(chi), 0, -np.cos(chi)],
                                     [1, 0, -np.sin(chi), -np.cos(chi)]])

        elif config.processing.calibration_scheme == '4-State Extinction': # if the images were taken using 4-frame scheme (Ext, 0, 60, 120)
            chi = self.swing
            inst_mat = np.array([[1, 0, 0, -1],
                                 [1, np.sin(2 * np.pi * chi), 0, -np.cos(2 * np.pi * chi)],
                                 [1, -0.5 * np.sin(2 * np.pi * chi), np.sqrt(3) * np.cos(np.pi * chi) * np.sin(np.pi * chi), -np.cos(2 * np.pi * chi)],
                                 [1, -0.5 * np.sin(2 * np.pi * chi), -np.sqrt(3) / 2 * np.sin(2 * np.pi * chi), -np.cos(2 * np.pi * chi)]])

        else:
            raise Exception('Expected image shape is (channel, y, x, z)...'
                            'The number of channels is {}, but allowed values are 4 or 5'.format(self._n_chann))

        self.inst_mat_inv = np.linalg.pinv(inst_mat)
        self.azimuth_offset = azimuth_offset/180*np.pi
        # self.stokes_param_bg_tm = []

        # self.stokes_param_bg_local_tm = []
        self.circularity = circularity

    @property
    def img_shape(self):
        return self._img_shape

    @img_shape.setter
    def img_shape(self, shape):
        assert len(shape) == 3 or 4, \
            'ImgReconstructor only supports 2D image or 3D stack'
        self._img_shape = shape
        if len(shape) == 3:
            [self._n_chann, self._height, self._width] = shape
            self._depth = 1
        else:
            [self._n_chann, self._height, self._width, self._depth] = shape

    def compute_stokes(self, config: ConfigReader, int_obj: IntensityData) -> StokesData:
        """
        Given raw polarization images, compute stokes images

        Parameters
        ----------
        int_obj : IntensityData
            input image with shape (channel, y, x) or (channel, y, x, z)

        Returns
        -------
        stokes parameters : list of nd array.
            [s0, s1, s2, s3]

        """
        if not isinstance(int_obj, IntensityData):
            raise TypeError("Incorrect Data Type: must be IntensityData")

        # check that dims match instrument matrix dims
        if self.img_shape[1:] != list(np.shape(int_obj.get_image('IExt'))):
            raise ValueError("Instrument matrix dimensions do not match supplied intensity dimensions")

        if self._n_chann == 4 and config.processing.calibration_scheme == '4-State':
            img_raw = np.stack((int_obj.get_image('IExt'),
                                int_obj.get_image('I45'),
                                int_obj.get_image('I90'),
                                int_obj.get_image('I135')))  # order the channel following stokes calculus convention

        elif self._n_chann == 4 and config.processing.calibration_scheme == '4-State Extinction':
            img_raw = np.stack((int_obj.get_image('IExt'),
                                int_obj.get_image('I0'),
                                int_obj.get_image('I60'),
                                int_obj.get_image('I120')))  # order the channel following stokes calculus convention
        elif self._n_chann == 5:
            img_raw = np.stack((int_obj.get_image('IExt'),
                                int_obj.get_image('I0'),
                                int_obj.get_image('I45'),
                                int_obj.get_image('I90'),
                                int_obj.get_image('I135')))  # order the channel following stokes calculus convention
        else:
            raise ValueError("Intensity data first dim must be # of channels.  Only n_chann = 4 or 5 implemented")

        # calculate stokes
        img_raw_flat = np.reshape(img_raw, (self._n_chann, -1))
        img_stokes_flat = np.dot(self.inst_mat_inv, img_raw_flat)
        img_stokes = np.reshape(img_stokes_flat, [4,] + self.img_shape[1:])

        out = StokesData()
        [out.s0,
         out.s1,
         out.s2,
         out.s3] = [img_stokes[i, ...] for i in range(4)]

        return out

    def stokes_normalization(self, stokes_param: StokesData) -> StokesData:
        """
        Computes S1 and S2 norms.  Computes normalized polarization.

        Parameters
        ----------
        stokes_param : Union[StokesData, StokesData]
            object of type StokesData or StokesData

        Returns
        -------
        StokesData :
            object of type StokesData

        """
        if not isinstance(stokes_param, StokesData):
            raise TypeError("stokes_param must be of type StokesData")

        norm_dat = StokesData()
        
        [s0, s1, s2, s3] = stokes_param.data
        
        if self.use_gpu:
            s0 = cp.array(s0)
            s1 = cp.array(s1)
            s2 = cp.array(s2)
            s3 = cp.array(s3)
            
            # set norm_dat's normalized data
            norm_dat.s1_norm      = cp.asnumpy(s1 / s3)
            norm_dat.s2_norm      = cp.asnumpy(s2 / s3)
            norm_dat.polarization = cp.asnumpy(cp.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2) / s0)
            
        else:
            
            # set norm_dat's normalized data
            norm_dat.s1_norm      = s1 / s3
            norm_dat.s2_norm      = s2 / s3
            norm_dat.polarization = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2) / s0


        # set norm_dat's stokes data
        [norm_dat.s0,
         norm_dat.s1,
         norm_dat.s2,
         norm_dat.s3] = stokes_param.data

        return norm_dat

    def correct_background_stokes(self, sample_norm_obj: StokesData, bg_norm_obj: StokesData) -> StokesData:
        """
        correct background of transformed Stokes parameters

        Parameters
        ----------
        sample_norm_obj : StokesData
            Object of type StokesData from normalized sample
        bg_norm_obj : StokesData
            Object of type StokesData from normalized background

        Returns
        -------
        StokesData
            Object of type StokesData with correction
        """
        # add a dummy z-dimension to background if sample image has xyz dimension
        if len(bg_norm_obj.s0.shape) < len(sample_norm_obj.s0.shape):
            # add blank axis to end of background images so it matches dim of input image
            bg_norm_obj.s0           = bg_norm_obj.s0[..., np.newaxis]
            bg_norm_obj.polarization = bg_norm_obj.polarization[..., np.newaxis]
            bg_norm_obj.s1_norm      = bg_norm_obj.s1_norm[..., np.newaxis]
            bg_norm_obj.s2_norm      = bg_norm_obj.s2_norm[..., np.newaxis]

        # perform the correction
        sample_norm_obj.s0           = sample_norm_obj.s0 / bg_norm_obj.s0
        sample_norm_obj.polarization = sample_norm_obj.polarization / bg_norm_obj.polarization
        sample_norm_obj.s1_norm      = sample_norm_obj.s1_norm - bg_norm_obj.s1_norm
        sample_norm_obj.s2_norm      = sample_norm_obj.s2_norm - bg_norm_obj.s2_norm


        return sample_norm_obj

    def correct_background(self, sample_data: StokesData, background_data: StokesData) -> StokesData:
        """
        Corrects background and (optionally) does local_fit or local_filter

        Parameters
        ----------
        sample_data : StokesData
            Object of type StokesData
        background_data : StokesData
            Object of type StokesData

        Returns
        -------
        StokesData
            Corrected data using local_fit, local_filter or global

        """

        if self.n_slice_local_bg > 1:
            assert len(np.shape(sample_data.data[0])) == 3, \
                'Input image has to have >1 z-slice for n_slice_local_bg > 1'

        # sample_stokes_norm_corrected has z-appended
        sample_stokes_norm_corrected = self.correct_background_stokes(sample_data, background_data)

        # if local BG correction
        if self.bg_method in ['Local_filter', 'Local_fit']:
            # average only along these 'correction' attributes
            correction = ['s0', 'polarization', 's1_norm', 's2_norm', 's3']

            sample_stokes_norm_local = StokesData()
            # if z-axis is present, average across all z for local BG correction
            ax = 2  # if len(sample_stokes_norm_corrected.s0.shape) > 2 else None
            [sample_stokes_norm_local.s0,
             sample_stokes_norm_local.polarization,
             sample_stokes_norm_local.s1_norm,
             sample_stokes_norm_local.s2_norm,
             sample_stokes_norm_local.s3] = [np.mean(img, axis=ax) if ax else img for img in
                                       [sample_stokes_norm_corrected.__getattribute__(corr) for corr in correction]]

            local_background = self.compute_local_background(sample_stokes_norm_local)


            sample_stokes_norm_corrected = self.correct_background_stokes(sample_stokes_norm_corrected, local_background)

        return sample_stokes_norm_corrected

    def compute_local_background(self, stokes_param_sm_local_tm: StokesData) -> StokesData:
        """
        Estimate local Stokes background using Guassian filter
        Parameters
        ----------
        stokes_param_sm_local_tm : StokesData
            Transformed sample Stokes parameters

        Returns
        -------
        StokesData
            local background Stokes parameters
        """

        stokes_param_bg_local_tm = StokesData()

        print('Estimating local background...')
        if self.bg_method == 'Local_filter':
            estimate_bg = self._gaussian_blur
        elif self.bg_method == 'Local_fit':
            estimate_bg = self._fit_background
        else:
            raise ValueError('background method has to be "Local_filter" or "Local_fit"')

        # estimate bg only on these stokes datasets
        correction = ['s0', 'polarization', 's1_norm', 's2_norm', 's3']

        [stokes_param_bg_local_tm.s0,
         stokes_param_bg_local_tm.polarization,
         stokes_param_bg_local_tm.s1_norm,
         stokes_param_bg_local_tm.s2_norm,
         stokes_param_bg_local_tm.s3] = [estimate_bg(img) for img in
                                         [stokes_param_sm_local_tm.__getattribute__(corr) for corr in correction]]


        return stokes_param_bg_local_tm

    def _gaussian_blur(self, img):
        background = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)
        return background

    def _fit_background(self, img):
        bg_estimator = BackgroundEstimator2D()
        background = bg_estimator.get_background(img, order=self.poly_fit_order, normalize=False)
        return background

    def reconstruct_birefringence(self, stokes_param_sm_tm: StokesData,
                                  img_crop_ref=None, extra=False) -> PhysicalData:
        """compute physical properties of birefringence

        Parameters
        ----------
        stokes_param_sm_tm: list of nd array.
            Transformed sample Stokes parameters

        Returns
        -------
        list of nd array.
              Brightfield_computed, Retardance, Orientation, Polarization, 'Stokes_1', 'Stokes_2', 'Stokes_3'
        """
        # for low birefringence sample that requires 0 background, set extra=True to manually offset the background
        # Correction based on Eq. 16 in reference using linear approximation assuming small retardance for both sample and background

        # ASmBg = 0
        # s2_normSmBg = 0
        # if extra: # extra background correction to set backgorund = 0
        #     imList = [s1_normSm, s2_normSm]
        #     imListCrop = imcrop(imList, I_ext) # manually select ROI with only background retardance
        #     s1_normSmCrop,s2_normSmCrop = imListCrop
        #     s1_normSmBg = np.nanmean(s1_normSmCrop)
        #     s2_normSmBg = np.nanmean(s2_normSmCrop)

        phys_data = PhysicalData()
        
        if self.use_gpu:
            s3 = cp.array(stokes_param_sm_tm.s3)
            s1 = cp.array(stokes_param_sm_tm.s1_norm) * s3
            s2 = cp.array(stokes_param_sm_tm.s2_norm) * s3
            
            retard = cp.arctan2(cp.sqrt(s1 ** 2 + s2 ** 2), s3)
            retard = cp.asnumpy(retard / (2 * np.pi) * self.wavelength)  # convert the unit to [nm]

            if self.circularity == 'lcp':
                azimuth = cp.asnumpy((0.5 * cp.arctan2(s1, -s2) + self.azimuth_offset) % (np.pi))  # make azimuth fall in [0,pi]
            elif self.circularity == 'rcp':
                azimuth = cp.asnumpy((0.5 * cp.arctan2(-s1, -s2) + self.azimuth_offset) % (np.pi))  # make azimuth fall in [0,pi]
            else:
                raise AttributeError("unable to compute azimuth, circularity parameter is not defined")
        else:
            s1 = stokes_param_sm_tm.s1_norm * stokes_param_sm_tm.s3
            s2 = stokes_param_sm_tm.s2_norm * stokes_param_sm_tm.s3

            retard = np.arctan2(np.sqrt(s1 ** 2 + s2 ** 2), stokes_param_sm_tm.s3)
            retard = retard / (2 * np.pi) * self.wavelength  # convert the unit to [nm]

            if self.circularity == 'lcp':
                azimuth = (0.5 * np.arctan2(s1, -s2) + self.azimuth_offset) % (np.pi)  # make azimuth fall in [0,pi]
            elif self.circularity == 'rcp':
                azimuth = (0.5 * np.arctan2(-s1, -s2) + self.azimuth_offset) % (np.pi)  # make azimuth fall in [0,pi]
            else:
                raise AttributeError("unable to compute azimuth, circularity parameter is not defined")

        phys_data.I_trans      = stokes_param_sm_tm.s0
        phys_data.retard       = retard
        phys_data.azimuth      = azimuth

        phys_data.polarization = stokes_param_sm_tm.polarization

        return phys_data

    def calibrate_inst_mat(self):
        raise NotImplementedError
        pass


