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
    acquDirPath = os.path.join(ImgBgPath, subDirName[0]) # only load the first position for now        
    ImgRawBg, ImgProcBg, ImgFluor = ParseTiffInput(acquDirPath, 0) # 0 for z-index
    Abg, Bbg, IAbsBg = computeAB(ImgRawBg,Chi)    
    subDirName = GetSubDirName(ImgSmPath)         
    DAPIBg = np.inf
    TdTomatoBg = np.inf
    if flatField: # find background flourescence for flatField corection  
        for subDir in subDirName:
            plt.close("all") # close all the figures from the last run
            if re.match( r'(\d?)-?Pos_?(\d+)_?(\d?)', subDir, re.M|re.I):
                acquDirPath = os.path.join(ImgSmPath, subDir) # only load the first acquisition for now            
                ImgRawSm, ImgProcSm, ImgFluor = ParseTiffInput(acquDirPath, '000')
                if ImgFluor.size:            
                    DAPI = ImgFluor[:,:,0] # Needs to be generalized in the future
                    TdTomato = ImgFluor[:,:,1]  # Needs to be generalized in the future
                    DAPIBg = ImgMin(DAPI, DAPIBg)
                    TdTomatoBg = ImgMin(TdTomato, TdTomatoBg)
    return Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg               
    

        
