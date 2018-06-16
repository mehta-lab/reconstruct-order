# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:21:38 2018

@author: Sam Guo
"""

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
import imageio
sns.set_context("poster")
plt.close("all") # close all the figures from the last run                        
           
def loadTiffStk(ImgOutPath): # Load the TIFF stack format output by the acquisition software created by Shalin (software name?)
    acquFiles = os.listdir(ImgOutPath)    
    ImgAniso = np.array([])
    ImgOrient = np.array([])
    ImgAvg = np.array([])
    for i in range(len(acquFiles)): # load Aniso images with Sigma0, 1, 2, 3 states, and processed images
#        matchObjAniso = re.match( r'img_000000000_State(\d+) - Acquired Image_000.tif', acquFiles[i], re.M|re.I) # read images with "state" string in the filename
        matchObjAniso = re.match( r'I1-aniso_Z(\d+)_T(\d+)_Ch1.tif', acquFiles[i], re.M|re.I) # read images with "state" string in the filename
        matchObjOrient = re.match( r'I2-orient_Z(\d+)_T(\d+)_Ch1.tif', acquFiles[i], re.M|re.I) # read computed images 
        matchObjAvg = re.match( r'I3-avg_Z(\d+)_T(\d+)_Ch1.tif', acquFiles[i], re.M|re.I) # read computed images 
        if matchObjAniso:            
            img = loadTiff(ImgOutPath, acquFiles[i])
            if ImgAniso.size:            
                ImgAniso = np.concatenate((ImgAniso, img), axis=0)
            else:
                ImgAniso = img
        elif matchObjOrient:
            img = loadTiff(ImgOutPath, acquFiles[i])
            
            if ImgOrient.size:            
                ImgOrient = np.concatenate((ImgOrient, img), axis=0)
            else:
                ImgOrient = img
        elif matchObjAvg:
            img = loadTiff(ImgOutPath, acquFiles[i])
            if ImgAvg.size:            
                ImgAvg = np.concatenate((ImgAvg, img), axis=0)
            else:
                ImgAvg = img 
    
    return ImgAniso, ImgOrient, ImgAvg 

def loadTiff(ImgOutPath, acquFiles):   
    TiffFile = os.path.join(ImgOutPath, acquFiles)
    img = cv2.imread(TiffFile,-1) # flag -1 to perserve the bit dept of the raw image
    img = img.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    img = img.reshape(1,img.shape[0], img.shape[1])
    return img
#%%
def pol2cart(ImgOrient, ImgAniso, anisoCeiling=0.45):            
    # Convert polarization factor and orientation into vector components    
    # scale images in the analysis folder to compute vector
    # components that can be plotted with quiver3d.
    ImgVx=np.cos((np.pi/18000)*ImgOrient);
    ImgVy=np.sin((np.pi/18000)*ImgOrient);
    ImgVz=1-(ImgAniso/(2**16-1))/anisoCeiling;
    ImgR = np.sqrt(ImgVx**2+ImgVy**2+ImgVz**2)
    ImgVx=ImgVx/ImgR;
    ImgVy=ImgVy/ImgR;
    ImgVz=ImgVz/ImgR;
    return ImgVx,ImgVy,ImgVz

def exporTiffStk(ImgAniso, ImgOrient, ImgAvg, ImgVx, ImgVy, ImgVz, figPath,  dpi=300):               
    images = [ImgAniso, ImgOrient, ImgAvg, ImgVx, ImgVy, ImgVz]
    tiffNames = ['ImgAniso', 'ImgOrient', 'ImgAvg', 'ImgVx', 'ImgVy', 'ImgVz']        
    for im, tiffName in zip(images, tiffNames):
        fileName = tiffName+'.tif'
#        cv2.imwrite(os.path.join(figPath, fileName), im)
        imageio.mimwrite(os.path.join(figPath, fileName),im)

#%%
ImgOutPath = 'C:/Google Drive/20170327EpithelialCells_Caco2/FOV1_AU1/Pos1/analysis' # MATLAB output image folder path
ImgAniso, ImgOrient, ImgAvg  = loadTiffStk(ImgOutPath)    
ImgVx,ImgVy,ImgVz = pol2cart(ImgOrient, ImgAniso, anisoCeiling=0.45)
exporTiffStk(ImgAniso, ImgOrient, ImgAvg, ImgVx, ImgVy, ImgVz,os.path.join(ImgOutPath,'Chimera'), dpi=300)
         