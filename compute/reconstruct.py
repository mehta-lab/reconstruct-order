import numpy as np
import sys
import cv2
# from utils.plotting import plot_sub_images
# from scipy.ndimage.filters import median_filter
from skimage.restoration import denoise_tv_chambolle
sys.path.append("..") # Add upper level directory to python modules path.
#from utils.imgCrop import imcrop
#%%
class ImgReconstructor:
    def __init__(self, img_raw_bg, bg_method='Global', swing=None, wavelength=532,
                 kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100)), output_path=None):
        self.img_raw_bg = img_raw_bg
        self.bg_method = bg_method
        self.swing = swing*2*np.pi # covert swing from fraction of wavelength to radian
        self.n_chann = np.shape(img_raw_bg)[0]
        self.height = np.shape(img_raw_bg)[1]
        self.width = np.shape(img_raw_bg)[2]
        self.wavelength = wavelength
        self.kernel = kernel
        self.output_path = output_path

    def reconstruct_birefringence(self, stokes_param_sm, stokes_param_bg=None, circularity='rcp', bg_method='Global',
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

        def stokes_transform(stokes_param):
            [s0, s1, s2, s3] = stokes_param
            s1_norm = s1 / s3
            s2_norm = -s2 / s3
            I_trans = s0
            polarization = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2) / s0
            return [I_trans, polarization, s1_norm, s2_norm, s3]

        def correct_background_stokes(stokes_param_sm, stokes_param_bg):
            [I_trans, polarization, s1_norm, s2_norm, s3] = stokes_param_sm
            [I_trans_bg, polarization_bg, s1_norm_bg, s2_norm_bg, s3_bg] = stokes_param_bg
            I_trans = I_trans / I_trans_bg
            polarization = polarization / polarization_bg
            s1_norm = s1_norm - s1_norm_bg
            s2_norm = s2_norm - s2_norm_bg
            return [I_trans, polarization, s1_norm, s2_norm, s3]

        def correct_background(stokes_param_sm, stokes_param_bg, bg_method='Global'):
            if stokes_param_bg:
                stokes_param_bg = stokes_transform(stokes_param_bg)
                stokes_param_bg = [denoise_tv_chambolle(img, weight=0.1) for img in stokes_param_bg]
                stokes_param_sm = correct_background_stokes(stokes_param_sm, stokes_param_bg)
                if bg_method == 'Local_filter':
                    stokes_param_bg_local = []
                    print('Estimating local background...')
                    for img in stokes_param_sm:
                        stokes_param_bg_local += [cv2.GaussianBlur(img, (401, 401), 0)]
                    stokes_param_sm = correct_background_stokes(stokes_param_sm, stokes_param_bg_local)
            return stokes_param_sm

        stokes_param_sm = stokes_transform(stokes_param_sm)
        stokes_param_sm = correct_background(stokes_param_sm, stokes_param_bg, bg_method=bg_method)
        [I_trans, polarization, s1_norm, s2_norm, s3] = stokes_param_sm
        s1 = s1_norm * s3
        s2 = s2_norm * s3
        retard = np.arctan2(np.sqrt(s1 ** 2 + s2 ** 2), s3)
        retard = retard / (2 * np.pi) * self.wavelength  # convert the unit to [nm]
        #    azimuth = 0.5*((np.arctan2(s1_norm,s2_norm)+2*np.pi)%(2*np.pi)) # make azimuth fall in [0,pi]
        if circularity == 'rcp':
            azimuth = (0.5 * np.arctan2(-s1, s2) + 0.5 * np.pi)  # make azimuth fall in [0,pi]
        elif circularity == 'lcp':
            azimuth = (0.5 * np.arctan2(s1, s2) + 0.5 * np.pi)  # make azimuth fall in [0,pi]
        return [I_trans, retard, azimuth, polarization]

    def calibrate_inst_mat(self):
        return

    def compute_stokes(self, img_raw):
        chi = self.swing
        I_ext = img_raw[0, :, :]  # Sigma0 in Fig.2
        I_90 = img_raw[1, :, :]  # Sigma2 in Fig.2
        I_135 = img_raw[2, :, :]  # Sigma4 in Fig.2
        I_45 = img_raw[3, :, :]  # Sigma3 in Fig.2
        # images = [I_ext, I_90, I_135, I_45]
        # titles = ['I_ext', 'I_90', 'I_135', 'I_45']
        # plot_sub_images(images, titles, self.output_path, 'raw')
        polarization = np.ones((self.height, self.width))  # polorization is always 1 for Jones calculus
        if img_raw.shape[0] == 4:  # if the images were taken using 4-frame scheme
            img_raw = np.stack((I_ext, I_45, I_90, I_135))  # order the channel following stokes calculus convention
            self.n_chann = np.shape(img_raw)[0]
            inst_mat = np.array([[1, 0, 0, -1],
                                 [1, 0, np.sin(chi), -np.cos(chi)],
                                 [1, -np.sin(chi), 0, -np.cos(chi)],
                                 [1, 0, -np.sin(chi), -np.cos(chi)]])
        elif img_raw.shape[0] == 5:  # if the images were taken using 5-frame scheme
            I_0 = img_raw[4, :, :]
            img_raw = np.stack((I_ext, I_0, I_45, I_90, I_135))  # order the channel following stokes calculus convention
            self.n_chann = np.shape(img_raw)[0]
            inst_mat = np.array([[1, 0, 0, -1],
                                 [1, np.sin(chi), 0, -np.cos(chi)],
                                 [1, 0, np.sin(chi), -np.cos(chi)],
                                 [1, -np.sin(chi), 0, -np.cos(chi)],
                                 [1, 0, -np.sin(chi), -np.cos(chi)]])
        inst_mat_inv = np.linalg.pinv(inst_mat)
        img_raw_flat = np.reshape(img_raw,(self.n_chann, self.height*self.width))
        img_stokes_flat = np.dot(inst_mat_inv, img_raw_flat)
        img_stokes = np.reshape(img_stokes_flat, (img_stokes_flat.shape[0], self.height, self.width))
        [s0, s1, s2, s3] = [img_stokes[i, :, :] for i in range(0, img_stokes.shape[0])]
        return [s0, s1, s2, s3]
