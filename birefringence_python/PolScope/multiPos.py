import os
import numpy as np
import matplotlib.pyplot as plt
import re
import cv2
#import sys
#sys.path.append("..") # Adds higher directory to python modules path.
from utils.imgIO import GetSubDirName, ParseTiffInput
from .reconstruct import computeAB, correctBackground, computeDeltaPhi
from utils.imgProcessing import ImgMin
from utils.plotting import plot_sub_images

def findBackground(ImgSmPath, ImgBgPath, Chi, flatField=False, method='open'):
    subDirName = GetSubDirName(ImgBgPath)               
    acquDirPath = os.path.join(ImgBgPath, subDirName[0]) # only load the first position for now        
    ImgRawBg, ImgProcBg, ImgFluor = ParseTiffInput(acquDirPath, 0) # 0 for z-index
    Abg, Bbg, IAbsBg = computeAB(ImgRawBg,Chi)    
    subDirName = GetSubDirName(ImgSmPath)         
    DAPIMin = np.inf
    TdTomatoMin = np.inf
    DAPIBg = np.ones(Abg.shape)
    TdTomatoBg = np.ones(Abg.shape)    
    DAPISum = np.zeros(Abg.shape)
    TdTomatoSum = np.zeros(Abg.shape)
    ind=0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))    
    if flatField: # find background flourescence for flatField corection  
        for subDir in subDirName:
            plt.close("all") # close all the figures from the last run
            if re.match( r'(\d?)-?Pos_?(\d+)_?(\d?)', subDir, re.M|re.I):
                acquDirPath = os.path.join(ImgSmPath, subDir) # only load the first acquisition for now            
                ImgRawSm, ImgProcSm, ImgFluor = ParseTiffInput(acquDirPath, 0)
                if ImgFluor.size:                    
                    DAPI = ImgFluor[:,:,0] # Needs to be generalized in the future
                    TdTomato = ImgFluor[:,:,1]  # Needs to be generalized in the future
                    DAPISum += cv2.morphologyEx(DAPI, cv2.MORPH_OPEN, kernel, borderType = cv2.BORDER_REPLICATE )
                    TdTomatoSum += cv2.morphologyEx(TdTomato, cv2.MORPH_OPEN, kernel, borderType = cv2.BORDER_REPLICATE )
                    DAPIMin = ImgMin(DAPI, DAPIMin)
                    TdTomatoMin = ImgMin(TdTomato, TdTomatoMin)                    
                    ind += 1
        DAPISum = DAPISum/ind
        
#        DAPIOpen = cv2.morphologyEx(DAPISum, cv2.MORPH_OPEN, kernel, borderType = cv2.BORDER_REPLICATE )
        TdTomatoSum = TdTomatoSum/ind        
#        TdTomatoOpen = cv2.morphologyEx(TdTomatoSum, cv2.MORPH_OPEN, kernel,borderType = cv2.BORDER_REPLICATE )
        DAPIOpen = DAPISum
        TdTomatoOpen = TdTomatoSum
        if method=='open':            
            DAPIBg = DAPIOpen
            TdTomatoBg = TdTomatoOpen
        elif method=='empty':
            DAPIBg = DAPIMin
            TdTomatoBg = TdTomatoMin
        ## compare empty v.s. open method#####
        titles = ['Ch1 (Open)','Ch2 (Open)','Ch1 (Empty)','ch2 (Empty)']
        images = [DAPIOpen, TdTomatoOpen, DAPIMin, TdTomatoMin]
        plot_sub_images(images,titles)
        plt.savefig(os.path.join(ImgSmPath,'compare_flat_field.png'),dpi=150)
        ##################################################################            
    return Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg               
    

        
