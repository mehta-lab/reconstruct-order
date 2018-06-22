import os
import numpy as np
import matplotlib.pyplot as plt
import re
#import sys
#sys.path.append("..") # Adds higher directory to python modules path.
from utils.imgIO import GetSubDirName, ParseTiffInput
from .reconstruct import computeAB, correctBackground, computeDeltaPhi
from utils.imgProcessing import ImgMin
from utils.plotting import plot_birefringence, plot_sub_images

def findBackground(ImgSmPath, ImgBgPath, Chi, flatField=False):
    subDirName = GetSubDirName(ImgBgPath)               
    acquDirPath = os.path.join(ImgBgPath, subDirName[0]) # only load the first acquisition for now        
    ImgRawBg, retardMMBg, azimuthMMBg, ImgFluor = ParseTiffInput(acquDirPath)
    Abg, Bbg, IAbsBg = computeAB(ImgRawBg,Chi)    
    subDirName = GetSubDirName(ImgSmPath)         
    DAPIBg = np.inf
    TdTomatoBg = np.inf
    if flatField: # find background flourescence for flatField corection  
        for subDir in subDirName:
            plt.close("all") # close all the figures from the last run
            if re.match( r'(\d?)-?Pos_?(\d+)_?(\d?)', subDir, re.M|re.I):
                acquDirPath = os.path.join(ImgSmPath, subDir) # only load the first acquisition for now            
                ImgRawSm, retardMMSm, azimuthMMSm, ImgFluor = ParseTiffInput(acquDirPath)
                if ImgFluor.size:            
                    DAPI = ImgFluor[:,:,0] # Needs to be generalized in the future
                    TdTomato = ImgFluor[:,:,1]  # Needs to be generalized in the future
                    DAPIBg = ImgMin(DAPI, DAPIBg)
                    TdTomatoBg = ImgMin(TdTomato, TdTomatoBg)
    return Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg               
    
def loopPos(ImgSmPath, Chi, Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg, flatField=False): 
#loop through each position in the acquistion folder, perform flat-field correction       
    subDirName = GetSubDirName(ImgSmPath)           
    ind = 0
    for subDir in subDirName:
        plt.close("all") # close all the figures from the last run
        acquDirPath = os.path.join(ImgSmPath, subDir) # only load the first acquisition for now  
        if re.match( r'(\d?)-?Pos_?(\d+)_?(\d?)', subDir, re.M|re.I):          
            DAPI = np.array([])
            TdTomato = np.array([])
            ImgRawSm, retardMMSm, azimuthMMSm, ImgFluor = ParseTiffInput(acquDirPath)    
            ASm, BSm, IAbsSm = computeAB(ImgRawSm, Chi)        
            A, B = correctBackground(ASm,BSm,Abg,Bbg, ImgRawSm, extra=False)            
            retard, azimuth = computeDeltaPhi(A,B)        
            #retard = removeBubbles(retard)     # remove bright speckles in mounted brain slice images       
            retardBg, azimuthBg = computeDeltaPhi(Abg, Bbg)
            if ImgFluor.size:
                DAPI = ImgFluor[:,:,0]
                TdTomato = ImgFluor[:,:,1]
            if flatField:
                IAbsSm = IAbsSm/IAbsBg #flat-field correction 
                DAPI = DAPI/DAPIBg # #flat-field correction 
                TdTomato = TdTomato/TdTomatoBg  # #flat-field correction         
                
            titles = ['Retardance (MM)','Orientation (MM)','Retardance (Py)','Orientation (Py)']
            images = [retardMMSm, azimuthMMSm,retard, azimuth]
            plot_sub_images(images,titles)
            plt.savefig(os.path.join(acquDirPath,'compare_MM_Py.png'),dpi=200)    
            plot_birefringence(IAbsSm,retard, azimuth, acquDirPath, ind, DAPI=DAPI,
                               TdTomato=TdTomato, spacing=6, vectorScl=0.5, zoomin=False, dpi=300)
            ind+=1
        
