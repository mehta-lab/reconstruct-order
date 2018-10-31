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

def FindDirContainPos(ImgPath):
    """
    Recursively find the parent directory of "Pos#" directory
    """
    subDirName = GetSubDirName(ImgPath)
    assert subDirName, 'No "Pos" directories found. Check if the input folder contains "Pos"'
    subDir = subDirName[0]  # get pos0 if it exists
    ImgSubPath = os.path.join(ImgPath, subDir)
    if 'Pos' not in subDir:
        ImgPath = FindDirContainPos(ImgSubPath)
        return ImgPath
    else:
        return ImgPath


def loadTiff(acquDirPath, acquFiles):
    """
    Load single tiff file
    :param acquDirPath str: directory of the tiff
    :param acquFiles str: file name of the tiff
    :return 2D float32 array: image
    """
    TiffFile = os.path.join(acquDirPath, acquFiles)
    img = cv2.imread(TiffFile,-1) # flag -1 to preserve the bit dept of the raw image
    img = img.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    # img = img.reshape(img.shape[0], img.shape[1],1)
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
            
        
def ParseTiffInput_old(img_io):
    """
    Parse tiff file name following mManager/Polacquisition output format
    :param img_io instance: instance of mManagerIO class holding imaging metadata
    :return 3D float32 arrays: stack of images parsed based on their imaging modalities with axis order (channel, row,
    column)
    """
    acquDirPath = img_io.ImgPosPath
    acquFiles = os.listdir(acquDirPath)
    ImgRaw = []
    ImgProc = []
    ImgBF = []
    ImgFluor = np.zeros((4, img_io.height,img_io.width)) # assuming 4 flour channels for now
    tIdx = img_io.tIdx
    zIdx = img_io.zIdx
    for fileName in acquFiles: # load raw images with Sigma0, 1, 2, 3 states, and processed images        
        matchObjRaw = re.match( r'img_000000%03d_(State|PolAcquisition|Zyla_PolState|EMCCD_PolState)(\d+)( - Acquired Image|_Confocal40|_Widefield|)_%03d.tif'%(tIdx,zIdx), fileName, re.M|re.I) # read images with "state" string in the filename
        matchObjProc = re.match( r'img_000000%03d_(.*) - Computed Image_%03d.tif'%(tIdx,zIdx), fileName, re.M|re.I) # read computed images
        matchObjFluor1 = re.match(
            r'img_000000%03d_(Zyla|EMCCD)_(Confocal40|Widefield|widefield|BF)_(.*)_%03d.tif'%(tIdx,zIdx), fileName, re.M|re.I)
        matchObjFluor2 = re.match(
            r'img_000000%03d_(Zyla|EMCCD)_(.*)_(Confocal40|Widefield|widefield|BF)_%03d.tif' % (tIdx, zIdx), fileName,
            re.M | re.I)  # read computed images
        matchObjBF = re.match( r'img_000000%03d_(Zyla|EMCCD)_(BF)_%03d.tif'%(tIdx,zIdx), fileName, re.M|re.I) # read computed images
        if any([matchObjRaw, matchObjProc, matchObjFluor1, matchObjFluor2, matchObjBF]):
            img = loadTiff(acquDirPath, fileName)
            img -= img_io.blackLevel
            if matchObjRaw:
                ImgRaw += [img]
            elif matchObjProc:
                ImgProc += [img]
            elif matchObjFluor1 or matchObjFluor2:
                if matchObjFluor1:
                    FluorChannName = matchObjFluor1.group(3)
                elif matchObjFluor2:
                    FluorChannName = matchObjFluor2.group(2)
                if FluorChannName in ['DAPI','405', '405nm']:
                    ImgFluor[0,:,:] = img
                elif FluorChannName in ['GFP','488', '488nm']:
                    ImgFluor[1,:,:] = img
                elif FluorChannName in ['TxR', 'TXR', '568', '568nm', '560']:
                    ImgFluor[2,:,:] = img
                elif FluorChannName in ['Cy5', 'IFP', '640', '640nm']:
                    ImgFluor[3,:,:] = img
            elif matchObjBF:
                ImgBF += [img]
    if ImgRaw:
        ImgRaw = np.stack(ImgRaw)
    if ImgProc:
        ImgProc = np.stack(ImgProc)
    if ImgBF:
        ImgBF = np.stack(ImgBF)
    return ImgRaw, ImgProc, ImgFluor, ImgBF 

def parse_tiff_input(img_io):
    """
    Parse tiff file name following mManager/Polacquisition output format
    :param img_io instance: instance of mManagerIO class holding imaging metadata
    :return 3D float32 arrays: stack of images parsed based on their imaging modalities with axis order (channel, row,
    column)
    """
    acquDirPath = img_io.ImgPosPath
    acquFiles = os.listdir(acquDirPath)
    ImgRaw = []
    ImgProc = []
    ImgBF = []
    ImgFluor = np.zeros((4, img_io.height,img_io.width)) # assuming 4 flour channels for now
    tIdx = img_io.tIdx
    zIdx = img_io.zIdx
    for fileName in acquFiles: # load raw images with Sigma0, 1, 2, 3 states, and processed images
        matchObj = re.match( r'img_000000%03d_(.*)_%03d.tif'%(tIdx,zIdx), fileName, re.M|re.I) # read images with "state" string in the filename
        if matchObj:
            img = loadTiff(acquDirPath, fileName)
            img -= img_io.blackLevel
            if any(substring in matchObj.group(1) for substring in ['State', 'Pol']):
                ImgRaw += [img]
            elif any(substring in matchObj.group(1) for substring in ['Computed Image']):
                ImgProc += [img]
            elif any(substring in matchObj.group(1) for substring in ['Confocal40','Confocal_40', 'Widefield', 'widefield']):
                if any(substring in matchObj.group(1) for substring in ['DAPI', '405', '405nm']):
                    ImgFluor[0,:,:] = img
                elif any(substring in matchObj.group(1) for substring in ['GFP', '488', '488nm']):
                    ImgFluor[1,:,:] = img
                elif any(substring in matchObj.group(1) for substring in ['TxR', 'TXR', '568', '568nm', '560']):
                    ImgFluor[2,:,:] = img
                elif any(substring in matchObj.group(1) for substring in ['Cy5', 'IFP', '640', '640nm']):
                    ImgFluor[3,:,:] = img
            elif any(substring in matchObj.group(1) for substring in ['BF']):
                ImgBF += [img]
    if ImgRaw:
        ImgRaw = np.stack(ImgRaw)
    if ImgProc:
        ImgProc = np.stack(ImgProc)
    if ImgBF:
        ImgBF = np.stack(ImgBF)
    return ImgRaw, ImgProc, ImgFluor, ImgBF

def exportImg(img_io,imgDict):
    tIdx = img_io.tIdx
    zIdx = img_io.zIdx
    posIdx = img_io.posIdx
    for tiffName in img_io.chNames:
        fileName = 'img_'+tiffName+'_t%03d_p%03d_z%03d.tif'%(tIdx, posIdx, zIdx)
        if len(imgDict[tiffName].shape)<3:
            cv2.imwrite(os.path.join(img_io.ImgOutPath, fileName), imgDict[tiffName])
        else:
            cv2.imwrite(os.path.join(img_io.ImgOutPath, fileName), cv2.cvtColor(imgDict[tiffName], cv2.COLOR_RGB2BGR))