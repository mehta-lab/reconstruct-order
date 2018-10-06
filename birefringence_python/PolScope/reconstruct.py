import numpy as np
import sys
sys.path.append("..") # Adds higher directory to python modules path.
#from utils.imgCrop import imcrop
#%%
class ImgReconstructor:
    def __init__(self, img_raw_bg, method='Stokes', swing=None):
        self.img_raw_bg = img_raw_bg
        self.method = method
        self.swing = swing*2*np.pi # covert swing from fraction of wavelength to radian

    def compute_background(self):
        if self.method == 'Jones':
            Abg, Bbg, IAbsBg, DeltaMaskBg = self.computeAB(self.img_raw_bg)
        elif self.method == 'Stokes':




    def computeAB(self, img_raw): # output numerators and denominators of A, B along with to retain the phase info
        chi = self.swing
        img_raw = self.img_raw
        I_ext = img_raw[0,:,:] # Sigma0 in Fig.2
        I_90 = img_raw[1,:,:] # Sigma2 in Fig.2
        I_135 = img_raw[2,:,:] # Sigma4 in Fig.2
        I_45 = img_raw[3,:,:] # Sigma3 in Fig.2
        if img_raw.shape[0]==4: # 4-frame algorithm
            nB = (I_135-I_45)*np.tan(chi/2)
            nA = (I_45+I_135 -2*I_90)*np.tan(chi/2)   # Eq. 10 in reference
            dAB = I_45+I_135-2*I_ext
            A = nA/dAB
            B = nB/dAB
            IAbs = I_45+I_135-2*np.cos(chi)*I_ext
            DeltaMask = dAB>=0 # Mask term in Eq. 11
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
            IAbs = I_45+I_135-2*np.cos(chi)*I_ext
            DeltaMask = dA>=0 # Mask term in Eq. 11
        return  A, B, IAbs, DeltaMask


    def correctBackground(img,ASm,BSm,img_raw,extra=False):
        # for low birefringence sample that requires 0 background, set extra=True to manually offset the background
        # Correction based on Eq. 16 in reference using linear approximation assuming small retardance for both sample and background
        I_ext = img_raw[:,:,0] # Sigma0 in Fig.2
        ASmBg = 0
        BSmBg = 0
        if extra: # extra background correction to set backgorund = 0
            imList = [ASm, BSm]
            imListCrop = imcrop(imList, I_ext) # manually select ROI with only background retardance
            ASmCrop,BSmCrop = imListCrop
            ASmBg = np.nanmean(ASmCrop)
            BSmBg = np.nanmean(BSmCrop)
        A = ASm-img.Abg-ASmBg # correct contributions from background
        B = BSm-img.Bbg-BSmBg
        return A, B

    def computeDeltaPhi(img, A, B, DeltaMask, flipPol=False):
        retard = np.arctan(np.sqrt(A**2+B**2))
        retardNeg = np.pi + np.arctan(np.sqrt(A**2+B**2)) # different from Eq. 10 due to the definition of arctan in numpy
        retard[~DeltaMask] = retardNeg[~DeltaMask] #Eq. 11
        retard = retard/(2*np.pi)*img.wavelength # convert the unit to [nm]
    #    azimuth = 0.5*((np.arctan2(A,B)+2*np.pi)%(2*np.pi)) # make azimuth fall in [0,pi]
        if flipPol:
            azimuth = (0.5*np.arctan2(-A, B)+0.5*np.pi) # make azimuth fall in [0,pi]
        else:
            azimuth = (0.5*np.arctan2(A, B)+0.5*np.pi) # make azimuth fall in [0,pi]
        return retard, azimuth

    def calibrate_inst_mat():

    def compute_stokes(img_io, img_raw):
        chi = img_io.swing * 2 * np.pi  # covert swing from fraction of wavelength to radian
        I_ext = img_raw[:, :, 0]
        I_90 = img_raw[:, :, 1]
        I_135 = img_raw[:, :, 2]
        I_45 = img_raw[:, :, 3]
        inst_mat = np.array([[1, 0, 0, -1],
                             [1, np.sin(chi), 0, -np.cos(chi)],
                             [1, 0, np.sin(chi), -np.cos(chi)],
                             [1, -np.sin(chi), 0, -np.cos(chi)],
                             [1, 0, -np.sin(chi), -np.cos(chi)]])
        inst_mat_inv = np.linalg.pinv(inst_mat)
        img_io.width
        stokes = np.dot(,