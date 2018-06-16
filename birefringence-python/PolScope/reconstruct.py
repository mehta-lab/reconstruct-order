import os
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import re
import cv2
import bisect
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets  import RectangleSelector
import warnings
sns.set_context("poster")
plt.close("all") # close all the figures from the last run
#%%               
def computeAB(ImgRaw,Chi): # output numerators and denominators of A, B along with to retain the phase info   
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
    elif ImgRaw.shape[2]==5: # 5-frame algorithm               
        I0 = ImgRaw[:,:,4] # Sigma1 in Fig.2          
        B = (I135-I45)*np.tan(Chi/2)/(I135+I45-2*Iext)
        A = (I0-I90)*np.tan(Chi/2)/(I0+I90-2*Iext)   # Eq. 10 in reference        
        IAbs = I45+I135-2*np.cos(Chi)*Iext        
    return  A, B, IAbs

def correctBackground(ASm,BSm,Abg,Bbg,ImgRaw,extra=False):
    # for low birefringence sample that requires 0 background, set extra=True to manually offset the background 
    Iext = ImgRaw[:,:,0] # Sigma0 in Fig.2
    ASmBg = 0
    BSmBg = 0
    if extra: # extra background correction to set backgorund = 0
        imList = [ASm, BSm]
        imListCrop = imcrop(imList, Iext) # manually select ROI with only background retardance    
        ASmCrop,BSmCrop = imListCrop 
        ASmBg = np.nanmean(ASmCrop)
        BSmBg = np.nanmean(BSmCrop)               
    A = ASm-Abg-ASmBg # correct contributions from background
    B = BSm-Bbg-BSmBg    
    return A, B        

def computeDeltaPhi(A,B):
    retardPos = np.arctan(np.sqrt(A**2+B**2))
    retardNeg = np.pi + np.arctan(np.sqrt(A**2+B**2)) # different from Eq. 10 due to the definition of arctan in numpy
    retard = retardNeg*(retardPos<0)+retardPos*(retardPos>=0)        
#    azimuth = 0.5*((np.arctan2(A,B)+2*np.pi)%(2*np.pi)) # make azimuth fall in [0,pi]    
    azimuth = (0.5*np.arctan2(A,B)+0.5*np.pi) # make azimuth fall in [0,pi]    
    return retard, azimuth
