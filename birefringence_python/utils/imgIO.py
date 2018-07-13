import os
import numpy as np
import glob
#import seaborn as sns
#import matplotlib.pyplot as plt
import re
import cv2
#import sys
#sys.path.append("..") # Adds higher directory to python modules path.
#from mpl_toolkits.axes_grid1 import make_axes_locatable

def GetSubDirName(ImgPath):
    subDirPath = glob.glob(os.path.join(ImgPath, '*/'))    
    subDirName = [os.path.split(subdir[:-1])[1] for subdir in subDirPath]            
    return subDirName

def loadTiff(acquDirPath, acquFiles):   
    TiffFile = os.path.join(acquDirPath, acquFiles)
    img = cv2.imread(TiffFile,-1) # flag -1 to perserve the bit dept of the raw image
    img = img.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    img = img.reshape(img.shape[0], img.shape[1],1)
    return img

def ParseFileList(acquDirPath):
    acquFiles = os.listdir(acquDirPath) 
    PolChan = []
    PolZ = []
    FluorChan = []
    FluorZ =[]
    for fileName in acquFiles:
        matchObjRaw = re.match( r'img_000000000_(State|PolAcquisition|Zyla_PolState)(\d+)( - Acquired Image|_Confocal40|)_(\d+).tif', fileName, re.M|re.I) # read images with "state" string in the filename
#        matchObjProc = re.match( r'img_000000000_(.*) - Computed Image_000.tif', fileName, re.M|re.I) # read computed images 
        matchObjFluor = re.match( r'img_000000000_Zyla_Confocal40_(.*)_(\d+).tif', fileName, re.M|re.I) # read computed images 
        if matchObjRaw:                   
            PolChan += [matchObjRaw.group(2)]
            PolZ += [matchObjRaw.group(4)]        
        elif matchObjFluor:
            FluorChan += [matchObjFluor.group(1)]
            FluorZ += [matchObjFluor.group(2)]
    PolChan = list(set(PolChan))
    PolZ = list(set(PolZ))
    PolZ = [int(z) for z in PolZ]
    FluorChan = list(set(FluorChan))
    FluorZ = list(set(FluorZ))    
    return PolChan, PolZ, FluorChan, FluorZ
            
        
def ParseTiffInput(acquDirPath,z): # Load the TIFF stack format output by the acquisition software created by Shalin (software name?)
    acquFiles = os.listdir(acquDirPath)      
    ImgRaw = np.array([])
    ImgProc = np.array([])
    ImgFluor = np.array([])
    
    for fileName in acquFiles: # load raw images with Sigma0, 1, 2, 3 states, and processed images        
        matchObjRaw = re.match( r'img_000000000_(State|PolAcquisition|Zyla_PolState)(\d+)( - Acquired Image|_Confocal40|)_%03d.tif'%z, fileName, re.M|re.I) # read images with "state" string in the filename
        matchObjProc = re.match( r'img_000000000_(.*) - Computed Image_%03d.tif'%z, fileName, re.M|re.I) # read computed images 
        matchObjFluor = re.match( r'img_000000000_Zyla_Confocal40_(.*)_%03d.tif'%z, fileName, re.M|re.I) # read computed images 
        if matchObjRaw:            
            img = loadTiff(acquDirPath, fileName)
            if ImgRaw.size:            
                ImgRaw = np.concatenate((ImgRaw, img), axis=2)
            else:
                ImgRaw = img
        elif matchObjProc:
            img = loadTiff(acquDirPath, fileName)
            if ImgProc.size:            
                ImgProc = np.concatenate((ImgProc, img), axis=2)
            else:
                ImgProc = img
        elif matchObjFluor:
            img = loadTiff(acquDirPath, fileName)
            if ImgFluor.size:            
                ImgFluor = np.concatenate((ImgFluor, img), axis=2)
            else:
                ImgFluor = img    
    
    return ImgRaw, ImgProc, ImgFluor 

def exportImg(images, tiffNames, ind, z, figPath):
    for im, tiffName in zip(images, tiffNames):
        fileName = tiffName+'_%03d_%03d.tif'%(ind,z)
        if len(im.shape)<3:
            cv2.imwrite(os.path.join(figPath, fileName), im)
        else:
            cv2.imwrite(os.path.join(figPath, fileName), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))