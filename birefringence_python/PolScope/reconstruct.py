import numpy as np
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from utils.imgCrop import imcrop
#%%               
def computeAB(ImgReader, ImgRaw): # output numerators and denominators of A, B along with to retain the phase info   
    Chi = ImgReader.swing
    Iext = ImgRaw[:,:,0] # Sigma0 in Fig.2
    I90 = ImgRaw[:,:,1] # Sigma2 in Fig.2
    I135 = ImgRaw[:,:,2] # Sigma4 in Fig.2
    I45 = ImgRaw[:,:,3] # Sigma3 in Fig.2        
    if ImgRaw.shape[2]==4: # 4-frame algorithm           
        nB = (I135-I45)*np.tan(Chi/2)
        nA = (I45+I135 -2*I90)*np.tan(Chi/2)   # Eq. 10 in reference
        dAB = I45+I135-2*Iext
        A = nA/dAB
        B = nB/dAB   
        IAbs = I45+I135-2*np.cos(Chi)*Iext
        DeltaMask = dAB>=0 # Mask term in Eq. 11
    elif ImgRaw.shape[2]==5: # 5-frame algorithm               
        I0 = ImgRaw[:,:,4] # Sigma1 in Fig.2          
        nB = (I135-I45)*np.tan(Chi/2)
        dB = I135+I45-2*Iext
        nA = (I0-I90)*np.tan(Chi/2)
        dA = I0+I90-2*Iext   # Eq. 10 in reference 
        dB[dB==0] = np.spacing(1.0)
        dA[dA==0] = np.spacing(1.0)
        A = nA/dA
        B = nB/dB   
        IAbs = I45+I135-2*np.cos(Chi)*Iext
        DeltaMask = dA>=0 # Mask term in Eq. 11
    else: # treat anything else as 5-frame for now      
        I0 = ImgRaw[:,:,4] # Sigma1 in Fig.2          
        nB = (I135-I45)*np.tan(Chi/2)
        dB = I135+I45-2*Iext
        nA = (I0-I90)*np.tan(Chi/2)
        dA = I0+I90-2*Iext   # Eq. 10 in reference 
        dB[dB==0] = np.spacing(1.0)
        dA[dA==0] = np.spacing(1.0)
        A = nA/dA
        B = nB/dB   
        IAbs = I45+I135-2*np.cos(Chi)*Iext
        DeltaMask = dA>=0 # Mask term in Eq. 11
    return  A, B, IAbs, DeltaMask

def correctBackground(img,ASm,BSm,ImgRaw,extra=False):
    # for low birefringence sample that requires 0 background, set extra=True to manually offset the background 
    # Correction based on Eq. 16 in reference using linear approximation assuming small retardance for both sample and background 
    Iext = ImgRaw[:,:,0] # Sigma0 in Fig.2
    ASmBg = 0
    BSmBg = 0
    if extra: # extra background correction to set backgorund = 0
        imList = [ASm, BSm]
        imListCrop = imcrop(imList, Iext) # manually select ROI with only background retardance    
        ASmCrop,BSmCrop = imListCrop 
        ASmBg = np.nanmean(ASmCrop)
        BSmBg = np.nanmean(BSmCrop)               
    A = ASm-img.Abg-ASmBg # correct contributions from background
    B = BSm-img.Bbg-BSmBg    
    return A, B        

def computeDeltaPhi(img, A,B,DeltaMask, flipPol=False):
    retard = np.arctan(np.sqrt(A**2+B**2))
    retardNeg = np.pi + np.arctan(np.sqrt(A**2+B**2)) # different from Eq. 10 due to the definition of arctan in numpy
    retard[~DeltaMask] = retardNeg[~DeltaMask] #Eq. 11
    retard = retard/(2*np.pi)*img.wavelength # convert the unit to [nm]    
#    azimuth = 0.5*((np.arctan2(A,B)+2*np.pi)%(2*np.pi)) # make azimuth fall in [0,pi]    
    if flipPol:
        azimuth = (0.5*np.arctan2(-A,B)+0.5*np.pi) # make azimuth fall in [0,pi]    
    else:
        azimuth = (0.5*np.arctan2(A,B)+0.5*np.pi) # make azimuth fall in [0,pi]    
    return retard, azimuth
