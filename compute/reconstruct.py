import numpy as np
import cv2

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

    def __init__(self, img_shape=None, bg_method='Global', n_slice_local_bg=1, swing=None, wavelength=532,
                 kernel_size=401, azimuth_offset=0, circularity='rcp'):

        self.img_shape = img_shape
        self.bg_method = bg_method
        self.n_slice_local_bg = n_slice_local_bg
        self.swing = swing * 2 * np.pi # covert swing from fraction of wavelength to radian
        self.wavelength = wavelength
        self.kernel_size = kernel_size
        chi = self.swing
        if self._n_chann == 4:  # if the images were taken using 4-frame scheme
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
        self.inst_mat_inv = np.linalg.pinv(inst_mat)
        self.azimuth_offset = azimuth_offset/180*np.pi
        self.stokes_param_bg_tm = []
        self.stokes_param_bg_local_tm = []
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

    def compute_stokes(self, img_raw):
        """
        Given raw polarization images, compute stokes images

        Parameters
        ----------
        img_raw : nd array.
            input image with shape (channel, y, x) or (channel, z, y, x)

        Returns
        -------
        stokes parameters : list of nd array.
            [s0, s1, s2, s3]

        """

        self.img_shape = np.shape(img_raw)
        img_raw_flat = np.reshape(img_raw, (self._n_chann, -1))
        img_stokes_flat = np.dot(self.inst_mat_inv, img_raw_flat)
        img_stokes = np.reshape(img_stokes_flat, (4,) + self.img_shape[1:])
        [s0, s1, s2, s3] = [img_stokes[i, ...] for i in range(4)]
        return [s0, s1, s2, s3]

    def stokes_transform(self, stokes_param):
        """
        Transform Stokes parameters for background correction

        Parameters
        ----------
        stokes_param : list of nd array.
            [s0, s1, s2, s3]

        Returns
        -------
        list of nd array
            Transformed Stokes parameters

        """
        [s0, s1, s2, s3] = stokes_param
        s1_norm = s1 / s3
        s2_norm = s2 / s3
        I_trans = s0
        polarization = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2) / s0
        return [I_trans, polarization, s1_norm, s2_norm, s3]

    def correct_background_stokes(self, stokes_param_sm_tm, stokes_param_bg_tm):
        """
        correct background of transformed Stokes parameters

        Parameters
        ----------
        stokes_param_sm_tm : list of nd array.
            Transformed sample Stokes parameters
        stokes_param_bg_tm
            Transformed background Stokes parameters

        Returns
        -------
        list of nd array.
            Background corrected transformed sample Stokes parameters
        """
        if len(stokes_param_bg_tm[0].shape) < len(self.img_shape):
            stokes_param_bg_tm = [img[..., np.newaxis] for img in stokes_param_bg_tm]
        [I_trans, polarization, s1_norm, s2_norm, s3] = stokes_param_sm_tm
        [I_trans_bg, polarization_bg, s1_norm_bg, s2_norm_bg, s3_bg] = stokes_param_bg_tm
        I_trans = I_trans / I_trans_bg
        polarization = polarization / polarization_bg
        s1_norm = s1_norm - s1_norm_bg
        s2_norm = s2_norm - s2_norm_bg
        # s3_norm = s3 / s3_bg
        return [I_trans, polarization, s1_norm, s2_norm, s3]

    def correct_background(self, stokes_param_sm_tm):
        """
        correct background of transformed Stokes parameters globally or locally

        Parameters
        ----------
        stokes_param_sm_tm : list of nd array.
            Transformed sample Stokes parameters

        Returns
        -------
        list of nd array.
            Background corrected transformed sample Stokes parameters

        """
        if self.n_slice_local_bg > 1:
            assert len(np.shape(stokes_param_sm_tm[0])) == 3, \
                'Input image has to have >1 z-slice for n_slice_local_bg > 1'
        stokes_param_sm_tm = self.correct_background_stokes(
            stokes_param_sm_tm, self.stokes_param_bg_tm)
        if self.bg_method == 'Local_filter':
            if self.n_slice_local_bg > 1:
                stokes_param_sm_local_tm = np.mean(stokes_param_sm_tm, -1)
            else:
                stokes_param_sm_local_tm = stokes_param_sm_tm
            self.compute_local_background(stokes_param_sm_local_tm)
            stokes_param_sm_tm = self.correct_background_stokes(
                stokes_param_sm_tm, self.stokes_param_bg_local_tm)
        return stokes_param_sm_tm

    def compute_local_background(self, stokes_param_sm_local_tm):
        """
        Estimate local Stokes background using Guassian filter
        Parameters
        ----------
        stokes_param_sm_local_tm : list of nd array.
            Transformed sample Stokes parameters

        Returns
        -------
        list of nd array
            local background Stokes parameters
        """
        stokes_param_bg_local_tm = []
        print('Estimating local background...')
        for img in stokes_param_sm_local_tm:
            img_filtered = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)
            stokes_param_bg_local_tm += [img_filtered]
        self.stokes_param_bg_local_tm = stokes_param_bg_local_tm

    def reconstruct_birefringence(self, stokes_param_sm_tm,
                           img_crop_ref=None, extra=False):
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
        [I_trans, polarization, s1_norm, s2_norm, s3] = stokes_param_sm_tm
        s1 = s1_norm * s3
        s2 = s2_norm * s3
        retard = np.arctan2(np.sqrt(s1 ** 2 + s2 ** 2), s3)
        retard = retard / (2 * np.pi) * self.wavelength  # convert the unit to [nm]
        if self.circularity == 'lcp':
            azimuth = (0.5 * np.arctan2(s1, -s2) + self.azimuth_offset) % (np.pi)  # make azimuth fall in [0,pi]
        elif self.circularity == 'rcp':
            azimuth = (0.5 * np.arctan2(-s1, -s2) + self.azimuth_offset) % (np.pi)  # make azimuth fall in [0,pi]
        return [I_trans, retard, azimuth, polarization, s1, s2, s3]

    def calibrate_inst_mat(self):
        raise NotImplementedError

        return


