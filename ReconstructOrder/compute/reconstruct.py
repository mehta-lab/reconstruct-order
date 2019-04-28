import numpy as np
import sys
import cv2
sys.path.append("..") # Add upper level directory to python modules path.


class ImgReconstructor:
    """
    ImgReconstructor contains methods to compute physical properties Birefringence /
    polarization and transmission given intensity data collected at 4 or 5 polarization orientations

    Parameters
    ----------
    img_pol_bg : list = [s0, s1, s2, s3]
        stokes images for background data.  Same as values returned by compute_stokes
    bg_method : str
        "Global", "Local".  type of background correction
    swing : float
    wavelength : int
    kernel : np.ndarray
    output_path : str
    azimuth_offset : int
    circularity : str

    Attributes
    ----------


    """

    def __init__(self, img_pol_bg=[], bg_method='Global', n_slice_local_bg=1, swing=None, wavelength=532,
                 kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100)),
                 output_path=None, azimuth_offset=0, circularity='rcp'):

        self.img_pol_bg = img_pol_bg
        self.bg_method = bg_method
        self.n_slice_local_bg = n_slice_local_bg
        self.swing = swing*2*np.pi # covert swing from fraction of wavelength to radian
        self.img_shape = np.shape(img_pol_bg)
        self.wavelength = wavelength
        self.kernel = kernel
        self.output_path = output_path
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
        Given raw image intensity, compute stokes images

        Parameters
        ----------
        img_raw : list of ndarray.  First element of output of utils.imgIO.parse_tiff_input

        Returns
        -------
        stokes images : list of ndarray.

        """

        self.img_shape = np.shape(img_raw)
        img_raw_flat = np.reshape(img_raw, (self._n_chann, -1))
        img_stokes_flat = np.dot(self.inst_mat_inv, img_raw_flat)
        img_stokes = np.reshape(img_stokes_flat, (4,) + self.img_shape[1:])
        [s0, s1, s2, s3] = [img_stokes[i, ...] for i in range(4)]
        return [s0, s1, s2, s3]

    def stokes_transform(self, stokes_param):
        [s0, s1, s2, s3] = stokes_param
        s1_norm = s1 / s3
        s2_norm = s2 / s3
        I_trans = s0
        polarization = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2) / s0
        return [I_trans, polarization, s1_norm, s2_norm, s3]

    def correct_background_stokes(self, stokes_param_sm_tm, stokes_param_bg_tm):
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
        if self.n_slice_local_bg > 1:
            assert len(np.shape(stokes_param_sm_tm[0])) == 3, \
                'Input image has to have >1 z-slice for n_slice_local_bg > 1'
        stokes_param_sm_tm = self.correct_background_stokes(stokes_param_sm_tm,
                                                            self.stokes_param_bg_tm)
        if self.bg_method == 'Local_filter':
            if self.n_slice_local_bg > 1:
                stokes_param_sm_local_tm = np.mean(stokes_param_sm_tm, -1)
            else:
                stokes_param_sm_local_tm = stokes_param_sm_tm
            self.compute_local_background(stokes_param_sm_local_tm)
            stokes_param_sm_tm = self.correct_background_stokes(stokes_param_sm_tm,
                                                                self.stokes_param_bg_local_tm)
        return stokes_param_sm_tm

    def compute_local_background(self, stokes_param_sm_local_tm):
        stokes_param_bg_local_tm = []
        print('Estimating local background...')
        for img in stokes_param_sm_local_tm:
            img_filtered = cv2.GaussianBlur(img, (401, 401), 0)
            stokes_param_bg_local_tm += [img_filtered]
        self.stokes_param_bg_local_tm = stokes_param_bg_local_tm

    def reconstruct_birefringence(self, stokes_param_sm_tm,
                           img_crop_ref=None, extra=False):
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
        return


