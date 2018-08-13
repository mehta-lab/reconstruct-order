"""
Read and write Tiff in mManager format. Will be replaced by mManagerIO.py 
"""
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
    assert os.path.exists(ImgPath), 'Input folder does not exist!' 
    subDirPath = glob.glob(os.path.join(ImgPath, '*/'))    
    subDirName = [os.path.split(subdir[:-1])[1] for subdir in subDirPath]
#    assert subDirName, 'No sub directories found'            
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
        matchObjRaw = re.match( r'img_000000000_(State|PolAcquisition|Zyla_PolState)(\d+)( - Acquired Image|_Confocal40|_Widefield|)_(\d+).tif', fileName, re.M|re.I) # read images with "state" string in the filename
#        matchObjProc = re.match( r'img_000000000_(.*) - Computed Image_000.tif', fileName, re.M|re.I) # read computed images 
        matchObjFluor = re.match( r'img_000000000_Zyla_(Confocal40|Widefield|widefield)_(.*)_(\d+).tif', fileName, re.M|re.I) # read computed images 
        
        if matchObjRaw:                   
            PolChan += [matchObjRaw.group(2)]
            PolZ += [matchObjRaw.group(4)]        
        elif matchObjFluor:
            FluorChan += [matchObjFluor.group(1)]
            FluorZ += [matchObjFluor.group(2)]
        
            
    PolChan = list(set(PolChan))
    PolZ = list(set(PolZ))
    PolZ = [int(zIdx) for zIdx in PolZ]
    FluorChan = list(set(FluorChan))
    FluorZ = list(set(FluorZ))    
    return PolChan, PolZ, FluorChan, FluorZ
            
        
def ParseTiffInput(imgInput): # Load the TIFF stack format output by the acquisition software created by Shalin (software name?)
    acquDirPath = imgInput.ImgPosPath
    acquFiles = os.listdir(acquDirPath)      
    ImgRaw = np.array([])
    ImgProc = np.array([])
    ImgFluor = np.zeros((imgInput.height,imgInput.width,4))
    ImgBF = np.array([])
    tIdx = imgInput.tIdx 
    zIdx = imgInput.zIdx
    for fileName in acquFiles: # load raw images with Sigma0, 1, 2, 3 states, and processed images        
        matchObjRaw = re.match( r'img_000000%03d_(State|PolAcquisition|Zyla_PolState)(\d+)( - Acquired Image|_Confocal40|_Widefield|)_%03d.tif'%(tIdx,zIdx), fileName, re.M|re.I) # read images with "state" string in the filename
        matchObjProc = re.match( r'img_000000%03d_(.*) - Computed Image_%03d.tif'%(tIdx,zIdx), fileName, re.M|re.I) # read computed images 
        matchObjFluor = re.match( r'img_000000%03d_Zyla_(Confocal40|Widefield|widefield|BF)_(.*)_%03d.tif'%(tIdx,zIdx), fileName, re.M|re.I) # read computed images 
        matchObjBF = re.match( r'img_000000%03d_Zyla_(BF)_%03d.tif'%(tIdx,zIdx), fileName, re.M|re.I) # read computed images 
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
            img = img.reshape(img.shape[0], img.shape[1])
            FluorChannName = matchObjFluor.group(2)
            if FluorChannName in ['DAPI']:
                ImgFluor[:,:,0] = img
            elif FluorChannName in ['GFP']:
                ImgFluor[:,:,1] = img
            elif FluorChannName in ['TxR']:
                ImgFluor[:,:,2] = img
            elif FluorChannName in ['Cy5']:
                ImgFluor[:,:,3] = img                            
        elif matchObjBF:
            img = loadTiff(acquDirPath, fileName)
            if ImgBF.size:            
                ImgBF = np.concatenate((ImgBF, img), axis=2)
            else:
                ImgBF = img  
    return ImgRaw, ImgProc, ImgFluor, ImgBF 

def exportImg(imgInput,images):    
    tIdx = imgInput.tIdx 
    zIdx = imgInput.zIdx
    posIdx = imgInput.posIdx
    for im, tiffName in zip(images, imgInput.chNames):
        fileName = 'img_'+tiffName+'_t%03d_p%03d_z%03d.tif'%(tIdx, posIdx, zIdx)
        if len(im.shape)<3:
            cv2.imwrite(os.path.join(imgInput.ImgOutPath, fileName), im)
        else:
            cv2.imwrite(os.path.join(imgInput.ImgOutPath, fileName), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))