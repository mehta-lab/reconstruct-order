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

def findBackground(ImgSmPath, ImgBgPath, Chi):
    subDirName = GetSubDirName(ImgBgPath)               
    acquDirPath = os.path.join(ImgBgPath, subDirName[0]) # only load the first acquisition for now        
    ImgRawBg, retardMMBg, azimuthMMBg, ImgFluor = ParseTiffInput(acquDirPath)
    Abg, Bbg, IAbsBg = computeAB(ImgRawBg,Chi)    
    subDirName = GetSubDirName(ImgSmPath)            
#    nDir = len(subDirName)   
    DAPIBg = np.inf
    TdTomatoBg = np.inf    
    for subDir in subDirName:
        plt.close("all") # close all the figures from the last run
        if re.match( r'(\d+)-Pos_(\d+)_(\d+)', subDir, re.M|re.I):
            acquDirPath = os.path.join(ImgSmPath, subDir) # only load the first acquisition for now            
            ImgRawSm, retardMMSm, azimuthMMSm, ImgFluor = ParseTiffInput(acquDirPath)
            if ImgFluor.size:            
                DAPI = ImgFluor[:,:,0] # Needs to be generalized in the future
                TdTomato = ImgFluor[:,:,1]  # Needs to be generalized in the future
                DAPIBg = ImgMin(DAPI, DAPIBg)
                TdTomatoBg = ImgMin(TdTomato, TdTomatoBg)
    return Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg        
        

    
def loopPos(ImgSmPath, Chi, Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg): # loop through each position in the acquistion folder       
    subDirName = GetSubDirName(ImgSmPath)            
#    nDir = len(subDirName)
    ind = 0
    for subDir in subDirName:
        plt.close("all") # close all the figures from the last run
        acquDirPath = os.path.join(ImgSmPath, subDir) # only load the first acquisition for now  
        if re.match( r'(\d?)-?Pos_?(\d+)_?(\d?)', subDir, re.M|re.I):          
            ImgRawSm, retardMMSm, azimuthMMSm, ImgFluor = ParseTiffInput(acquDirPath)    
            ASm, BSm, IAbsSm = computeAB(ImgRawSm, Chi)        
            A, B = correctBackground(ASm,BSm,Abg,Bbg, ImgRawSm, extra=False)
#            IAbsSm = IAbsSm/IAbsBg
            retard, azimuth = computeDeltaPhi(A,B)        
            #retard = removeBubbles(retard)        
            retardBg, azimuthBg = computeDeltaPhi(Abg, Bbg)
            if ImgFluor.size:            
                DAPI = ImgFluor[:,:,0]/DAPIBg # Needs to be generalized in the future
                TdTomato = ImgFluor[:,:,1]/TdTomatoBg  # Needs to be generalized in the future
            else:
                DAPI = np.array([])
                TdTomato = np.array([])
            titles = ['Retardance (MM)','Orientation (MM)','Retardance (Py)','Orientation (Py)']
            images = [retardMMSm, azimuthMMSm,retard, azimuth]
            plot_sub_images(images,titles)
            plt.savefig(os.path.join(acquDirPath,'compare_MM_Py.png'),dpi=200)    
            plot_birefringence(IAbsSm,retard, azimuth, acquDirPath, ind, DAPI=DAPI,
                               TdTomato=TdTomato, spacing=6, vectorScl=0.5, zoomin=False, dpi=300)
            ind+=1
        
