import numpy as np
import sys
import cv2
from utils.plotting import plot_sub_images
from scipy.ndimage.filters import median_filter

sys.path.append("..") # Add upper level directory to python modules path.
#from utils.imgCrop import imcrop
#%%
class ImgReconstructor:
    def __init__(self, img_raw_bg, method='Stokes', swing=None, wavelength=532,
                 kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100)), output_path=None):
        self.img_raw_bg = img_raw_bg
        self.method = method
        self.swing = swing*2*np.pi # covert swing from fraction of wavelength to radian
        self.n_chann = np.shape(img_raw_bg)[0]
        self.height = np.shape(img_raw_bg)[1]
        self.width = np.shape(img_raw_bg)[2]
        self.wavelength = wavelength
        self.kernel = kernel
        self.output_path = output_path

    def compute_param(self, img_raw):
        if self.method == 'Jones':
            img_param = self.compute_jones(img_raw)
        elif self.method == 'Stokes':
            img_param = self.compute_stokes(img_raw)
        return img_param

    def compute_jones(self, img_raw): # output numerators and denominators of A, B along with to retain the phase info
        assert self.n_chann in [4,5], \
            'reconstruction using Jones calculus only supports 4- or 5- frame algorithm'
        chi = self.swing
        I_ext = img_raw[0,:,:] # Sigma0 in Fig.2
        I_90 = img_raw[1,:,:] # Sigma2 in Fig.2
        I_135 = img_raw[2,:,:] # Sigma4 in Fig.2
        I_45 = img_raw[3,:,:] # Sigma3 in Fig.2
        polarization = np.ones((self.height, self.width)) # polorization is always 1 for Jones calculus
        if img_raw.shape[0]==4: # 4-frame algorithm
            nB = (I_135-I_45)*np.tan(chi/2)
            nA = (I_45+I_135 -2*I_90)*np.tan(chi/2)   # Eq. 10 in reference
            dAB = I_45+I_135-2*I_ext
            A = nA/dAB
            B = nB/dAB
            I_trans = I_45+I_135-2*np.cos(chi)*I_ext

        elif img_raw.shape[0]==5: # 5-frame algorithm
            I_0 = img_raw[4,:,:] # Sigma1 in Fig.2
            nB = (I_135-I_45)*np.tan(chi/2)
            dB = I_135+I_45-2*I_ext
            nA = (I_0-I_90)*np.tan(chi/2)
            dA = I_0+I_90-2*I_ext   # Eq. 10 in reference
            dB[dB==0] = np.spacing(1.0)
            dA[dA==0] = np.spacing(1.0)
            A = nA/dA
            B = nB/dB
            I_trans = I_45+I_135-2*np.cos(chi)*I_ext
            dAB = (dA+dB)/2

        return [I_trans, polarization, A, B, dAB]

    def correct_background(self, img_param_sm, img_param_bg, method='Global',
                           img_crop_ref=None, extra=False):
        # for low birefringence sample that requires 0 background, set extra=True to manually offset the background
        # Correction based on Eq. 16 in reference using linear approximation assuming small retardance for both sample and background

        # ASmBg = 0
        # BSmBg = 0
        # if extra: # extra background correction to set backgorund = 0
        #     imList = [ASm, BSm]
        #     imListCrop = imcrop(imList, I_ext) # manually select ROI with only background retardance
        #     ASmCrop,BSmCrop = imListCrop
        #     ASmBg = np.nanmean(ASmCrop)
        #     BSmBg = np.nanmean(BSmCrop)

        [I_trans_sm, polarization_sm, A_sm, B_sm, dAB_sm] = img_param_sm
        [I_trans_bg, polarization_bg, A_bg, B_bg, dAB_bg] = img_param_bg
        I_trans_sm = I_trans_sm/I_trans_bg
        polarization_sm = polarization_sm/polarization_bg
        A_sm = A_sm - A_bg
        B_sm = B_sm - B_bg
        img_param = [I_trans_sm, polarization_sm, A_sm, B_sm, dAB_sm]

        # img_param = [np.subtract(i, j) for i, j in zip(img_param_sm, img_param_bg)]  # correct contributions from background
        # img_param = np.subtract(img_param_sm, img_param_bg)  # correct contributions from backgroundimg_param = [i - j for i, j in zip(img_param_sm, img_param_bg)]  # correct contributions from background
        return img_param

    def reconstruct_img(self, img_param, flipPol=False):
        # if self.method == 'Jones':
        [I_trans, polarization, A, B, dAB] = img_param
        retard = np.arctan(np.sqrt(A ** 2 + B ** 2))
        retardNeg = np.pi + np.arctan(
            np.sqrt(A ** 2 + B ** 2))  # different from Eq. 10 due to the definition of arctan in numpy
        DeltaMask = dAB >= 0  # Mask term in Eq. 11
        retard[~DeltaMask] = retardNeg[~DeltaMask]  # Eq. 11
        retard = retard / (2 * np.pi) * self.wavelength  # convert the unit to [nm]
        #    azimuth = 0.5*((np.arctan2(A,B)+2*np.pi)%(2*np.pi)) # make azimuth fall in [0,pi]
        if flipPol:
            azimuth = (0.5 * np.arctan2(-A, B) + 0.5 * np.pi)  # make azimuth fall in [0,pi]
        else:
            azimuth = (0.5 * np.arctan2(A, B) + 0.5 * np.pi)  # make azimuth fall in [0,pi]

        # elif self.method == 'Stokes':
        #     [s0, s1, s2, s3] = img_param
        #     retard = np.arctan2(s3, np.sqrt(s1 ** 2 + s2 ** 2))
        #     retard = (retard + np.pi) % np.pi
        #     if flipPol:
        #         # azimuth = (0.5 * np.arctan2(-s1, s2) + 0.5 * np.pi)  # make azimuth fall in [0,pi]
        #         azimuth = 0.5 * ((np.arctan2(-s1, s2) + 2 * np.pi) % (2 * np.pi))
        #     else:
        #         azimuth = (0.5 * np.arctan2(s1, s2) + 0.5 * np.pi)  # make azimuth fall in [0,pi]
        #     polarization = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)/s0
        return retard, azimuth, polarization

    def calibrate_inst_mat(self):
        return

    def compute_stokes(self, img_raw):
        chi = self.swing
        I_ext = img_raw[0, :, :]  # Sigma0 in Fig.2
        I_90 = img_raw[1, :, :]  # Sigma2 in Fig.2
        I_135 = img_raw[2, :, :]  # Sigma4 in Fig.2
        I_45 = img_raw[3, :, :]  # Sigma3 in Fig.2
        images = [I_ext, I_90, I_135, I_45]
        titles = ['I_ext', 'I_90', 'I_135', 'I_45']
        plot_sub_images(images, titles, self.output_path, 'raw')
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
        images = [s0, s1, s2, s3]
        titles = ['s0', 's1', 's2', 's3']
        plot_sub_images(images, titles, self.output_path, 'stokes')
        A = s1/s3
        B = -s2/s3
        I_trans = s0
        polarization = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)/s0
        dAB = s3
        return [I_trans, polarization, A, B, dAB]
