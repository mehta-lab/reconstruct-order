import os
import numpy as np
import matplotlib.pyplot as plt
import re
import cv2
#import sys
#sys.path.append("..") # Adds higher directory to python modules path.
from utils.imgIO import ParseTiffInput, exportImg
from .reconstruct import computeAB, correctBackground, computeDeltaPhi
from utils.imgProcessing import ImgMin
from utils.plotting import plot_sub_images
from utils.mManagerIO import mManagerReader, PolAcquReader


from utils.plotting import plot_birefringence, plot_sub_images
from utils.imgProcessing import ImgLimit


def findBackground(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann, flatField=False, bgCorrect='Auto', method='open'):
    """
    Estimate background for each channel to perform background substraction for
    birefringence and flat-field correction (division) for bright-field and 
    fluorescence channels
        
    """
    
    ImgSmPath = os.path.join(RawDataPath, ImgDir, SmDir) # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'    
    OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir) 
    try:
        imgSm = PolAcquReader(ImgSmPath, OutputPath)
    except:
        imgSm = mManagerReader(ImgSmPath,OutputPath)
    if bgCorrect=='None':
        print('No background correction is performed...')
        BgDir = SmDir # need smarter way to deal with different backgroud options           
    elif bgCorrect=='Input':
        OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir+'_'+BgDir)
        imgSm.ImgOutPath = OutputPath
    else: #'Auto'        
        if hasattr(imgSm, 'bg'):
            BgDir = imgSm.bg
        else:
            print('Background not specified in metadata. Use user input background directory')   
        OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir+'_'+BgDir)
        imgSm.ImgOutPath = OutputPath
        

    
    ImgBgPath = os.path.join(RawDataPath, ImgDir, BgDir) # Background image folder path, of form 'BG_yyyy_mmdd_hhmm_X'
    imgBg = PolAcquReader(ImgBgPath, OutputPath)    
    imgBg.posIdx = 0 # assuming only single image for background 
    imgBg.tIdx = 0
    imgBg.zIdx = 0
    ImgRawBg, ImgProcBg, ImgFluor, ImgBF = ParseTiffInput(imgBg) # 0 for z-index
    Abg, Bbg, IAbsBg, DeltaMaskBg  = computeAB(imgBg, ImgRawBg) 
    
    imgSm.Abg = Abg
    imgSm.Bbg = Bbg
    imgSm.swing = imgBg.swing
    imgSm.wavelength = imgBg.wavelength
    imgSm.ImgFluorMin  = np.full((imgBg.height,imgBg.width,4), np.inf) # set initial min array to to be Inf     
    imgSm.ImgFluorSum  = np.zeros((imgBg.height,imgBg.width,4)) # set the default background to be Ones (uniform field)
    imgSm.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))  # kernel for image opening operation, 100-200 is usually good 
    imgSm.loopZ ='background'
    ImgFluorBg = np.ones((imgBg.height,imgBg.width,4))
    
    if flatField: # find background flourescence for flatField corection 
        print('Calculating illumination function for flatfield correction...')
        imgSm = loopPos(imgSm, outputChann, flatField=flatField)                                    
        imgSm.ImgFluorSum = imgSm.ImgFluorSum/imgSm.nPos # normalize the sum                     
        if method=='open':            
            ImgFluorBg = imgSm.ImgFluorSum            
        elif method=='empty':
            ImgFluorBg = imgSm.ImgFluorMin   
        
        ## compare empty v.s. open method#####
#        titles = ['Ch1 (Open)','Ch2 (Open)','Ch1 (Empty)','ch2 (Empty)']
#        images = [imgSm.ImgFluorSum[:,:,0], imgSm.ImgFluorSum[:,:,1], 
#                  imgSm.ImgFluorMin[:,:,0], imgSm.ImgFluorMin[:,:,1]]
#        plot_sub_images(images,titles)
        print('Exporting illumination function...')
#        plt.savefig(os.path.join(OutputPath,'compare_flat_field.png'),dpi=150)
        ##################################################################
    else:        
        IAbsBg = np.ones((imgBg.height,imgBg.width))  # use uniform field if no correction
    imgSm.IAbsBg = IAbsBg
    imgSm.ImgFluorBg = ImgFluorBg        
    return imgSm               
    
def loopPos(imgSm, outputChann, flatField=False, bgCorrect=True, flipPol=False): 
    """
    Loops through each position in the acquisition folder, and performs flat-field correction.
    
    @param ImgSmPath: Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'
    @param OutputPath: Output folder path
    @param Chi: Swing
    @param Lambda: Wavelength of polarized light.
    @param Abg: A term in background
    @param Bbg: B term in background
    @param IAbsBg: another background term.
    @param DAPIBg: another backgruond term.
    @param TdTomatoBg: another background term.
    @param flatField: boolean - whether flatField correction is applied.
    @param bgCorrect: boolean - whether or not background correction is applied.
    @param flipPol: whether or not to flip the sign of polarization.
    @return: None
    """       
    
    
    for posIdx in range(0,imgSm.nPos):
        print('Processing position %03d ...' %posIdx)
        plt.close("all") # close all the figures from the last run
        if imgSm.metaFile['Summary']['InitialPositionList']: # PolAcquisition doens't save position list
            subDir = imgSm.metaFile['Summary']['InitialPositionList'][posIdx]['Label']   
        else:
            subDir = 'Pos0'
        imgSm.ImgPosPath = os.path.join(imgSm.ImgSmPath, subDir)
        imgSm.posIdx = posIdx
        img = loopT(imgSm, outputChann, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol)
    return img                
        
def loopT(img, outputChann, flatField=False, bgCorrect=True, flipPol=False):
    for tIdx in range(0,img.nTime):        
        img.tIdx = tIdx
        if img.loopZ =='sample':
            img = loopZSm(img, outputChann, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol)
        else:
            img = loopZBg(img, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol)
    return img        

def loopZSm(img, outputChann, flatField=False, bgCorrect=True, flipPol=False):
    """
    Loops through Z.
    
    @param PolZ: Polarization Z
    @param ind:
    @param acquDirPath
    @param OutputPath: Output folder path
    @param Chi: Swing
    @param Lambda: Wavelength of polarized light.
    @param Abg: A term in background
    @param Bbg: B term in background
    @param IAbsBg: another background term.
    @param DAPIBg: another backgruond term.
    @param TdTomatoBg: another background term.
    @param imgLimits:
    @param flatField: boolean - whether flatField correction is applied.
    @param bgCorrect: boolean - whether or not background correction is applied.
    @param flipPol: whether or not to flip the sign of polarization.
    @return: None
    """  
   
    
    for zIdx in range(0,img.nZ):
        plt.close("all") # close all the figures from the last run
        img.zIdx = zIdx      
        retardMMSm = np.array([])
        azimuthMMSm = np.array([])     
        ImgRawSm, ImgProcSm, ImgFluor, ImgBF = ParseTiffInput(img)            
        ASm, BSm, IAbsSm, DeltaMaskSM = computeAB(img,ImgRawSm)
        if bgCorrect == 'None':                    
            A, B = ASm, BSm
        else:
            A, B = correctBackground(img,ASm,BSm,ImgRawSm, extra=False) # background subtraction             
        retard, azimuth = computeDeltaPhi(img,A,B,DeltaMaskSM,flipPol=flipPol)        
        #retard = removeBubbles(retard)     # remove bright speckles in mounted brain slice images       
#        retardBg, azimuthBg = computeDeltaPhi(Abg, Bbg,flipPol=flipPol)
        if not ImgBF.size: # use brightfield calculated from pol-images if there is no brighfield data
            ImgBF = IAbsSm
        else:
            ImgBF = ImgBF[:,:,0]
            
        for i in range(ImgFluor.shape[2]):
            if np.any(ImgFluor[:,:,i]):  # if the flour channel exists   
                ImgFluor[:,:,i] = ImgFluor[:,:,i]/img.ImgFluorBg[:,:,i]
            
        if ImgProcSm.size:
            retardMMSm =  ImgProcSm[:,:,0]
            azimuthMMSm = ImgProcSm[:,:,1]
        if flatField:
            ImgBF = ImgBF/img.IAbsBg #flat-field correction 
                    
            ## compare python v.s. Polacquisition output#####
#            titles = ['Retardance (MM)','Orientation (MM)','Retardance (Py)','Orientation (Py)']
#            images = [retardMMSm, azimuthMMSm,retard, azimuth]
#            plot_sub_images(images,titles)
#            plt.savefig(os.path.join(acquDirPath,'compare_MM_Py.png'),dpi=200)
            ##################################################################

        imgs = [ImgBF,retard, azimuth, ImgFluor]
#        imgLimits = ImgLimit(imgs,imgLimits)
        if not os.path.exists(img.ImgOutPath): # create folder for processed images
            os.makedirs(img.ImgOutPath)
        img, imgs = plot_birefringence(img, imgs,outputChann, spacing=10, vectorScl=10, zoomin=False, dpi=300)
        
        
        
        
        ##To do: add 'Fluor+Retardance' channel## 
        
        img.writeMetaData()
        exportImg(img, imgs)
    return img     

def loopZBg(img, flatField=False, bgCorrect=True, flipPol=False):           
    for zIdx in range(0,1): # only use the first z 
        img.zIdx = zIdx              
        ImgRawSm, ImgProcSm, ImgFluor, ImgBF = ParseTiffInput(img)            
        for i in range(ImgFluor.shape[2]):
            if np.any(ImgFluor[:,:,i]):  # if the flour channel exists                                      
                img.ImgFluorSum[:,:,i] += cv2.morphologyEx(ImgFluor[:,:,i], cv2.MORPH_OPEN, img.kernel, borderType = cv2.BORDER_REPLICATE )                    
                img.ImgFluorMin[:,:,i] = ImgMin(ImgFluor[:,:,i], img.ImgFluorMin[:,:,i]) 
    return img                                       

