import os
import numpy as np
import matplotlib.pyplot as plt
import re
import cv2
#import sys
#sys.path.append("..") # Adds higher directory to python modules path.
from utils.imgIO import ParseTiffInput, exportImg, GetSubDirName, FindDirContainPos
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
    ImgSmPath = FindDirContainPos(ImgSmPath)
    try:
        imgIOSm = PolAcquReader(ImgSmPath, OutputPath, outputChann)
    except:
        imgIOSm = mManagerReader(ImgSmPath,OutputPath, outputChann)
    if bgCorrect=='None':
        print('No background correction is performed...')
        BgDir = SmDir # need a smarter way to deal with different backgroud options           
    elif bgCorrect=='Input':
        OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir+'_'+BgDir)
        imgIOSm.ImgOutPath = OutputPath
    else: #'Auto'        
        if hasattr(imgIOSm, 'bg'):
            if imgIOSm.bg == 'No Background':
                bgCorrect=='None' # need to pass the flag down     
                print('No background correction is performed for background measurments...')
            else:
                BgDir = imgIOSm.bg
        else:
            print('Background not specified in metadata. Use user input background directory')   
        OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir+'_'+BgDir)
        imgIOSm.ImgOutPath = OutputPath

    ImgBgPath = os.path.join(RawDataPath, ImgDir, BgDir) # Background image folder path, of form 'BG_yyyy_mmdd_hhmm_X'
    imgIOBg = PolAcquReader(ImgBgPath, OutputPath)    
    imgIOBg.posIdx = 0 # assuming only single image for background 
    imgIOBg.tIdx = 0
    imgIOBg.zIdx = 0
    ImgRawBg, ImgProcBg, ImgFluor, ImgBF = ParseTiffInput(imgIOBg) # 0 for z-index
    Abg, Bbg, IAbsBg, DeltaMaskBg  = computeAB(imgIOBg, ImgRawBg) 
    
    imgIOSm.Abg = Abg
    imgIOSm.Bbg = Bbg
    imgIOSm.swing = imgIOBg.swing
    imgIOSm.wavelength = imgIOBg.wavelength
    imgIOSm.ImgFluorMin  = np.full((4,imgIOBg.height,imgIOBg.width), np.inf) # set initial min array to to be Inf
    imgIOSm.ImgFluorSum  = np.zeros((4,imgIOBg.height,imgIOBg.width)) # set the default background to be Ones (uniform field)
    imgIOSm.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))  # kernel for image opening operation, 100-200 is usually good 
    imgIOSm.loopZ ='background'
    ImgFluorBg = np.ones((4,imgIOBg.height,imgIOBg.width))
    
    if flatField: # find background flourescence for flatField corection 
        print('Calculating illumination function for flatfield correction...')
        imgIOSm = loopPos(imgIOSm, outputChann, flatField=flatField)                                    
        imgIOSm.ImgFluorSum = imgIOSm.ImgFluorSum/imgIOSm.nPos # normalize the sum                     
        if method=='open':            
            ImgFluorBg = imgIOSm.ImgFluorSum            
        elif method=='empty':
            ImgFluorBg = imgIOSm.ImgFluorMin   
        
        ## compare empty v.s. open method#####
#        titles = ['Ch1 (Open)','Ch2 (Open)','Ch1 (Empty)','ch2 (Empty)']
#        images = [imgIOSm.ImgFluorSum[:,:,0], imgIOSm.ImgFluorSum[:,:,1], 
#                  imgIOSm.ImgFluorMin[:,:,0], imgIOSm.ImgFluorMin[:,:,1]]
#        plot_sub_images(images,titles)
        print('Exporting illumination function...')
#        plt.savefig(os.path.join(OutputPath,'compare_flat_field.png'),dpi=150)
        ##################################################################
    else:        
        IAbsBg = np.ones((imgIOBg.height,imgIOBg.width))  # use uniform field if no correction
    imgIOSm.IAbsBg = IAbsBg
    imgIOSm.ImgFluorBg = ImgFluorBg        
    return imgIOSm               
    
def loopPos(imgIOSm, outputChann, flatField=False, bgCorrect=True, flipPol=False): 
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
    
    
    for posIdx in range(0,imgIOSm.nPos):
        print('Processing position %03d ...' %posIdx)
        plt.close("all") # close all the figures from the last run
        if imgIOSm.metaFile['Summary']['InitialPositionList']: # PolAcquisition doens't save position list
            subDir = imgIOSm.metaFile['Summary']['InitialPositionList'][posIdx]['Label']   
        else:
            subDir = 'Pos0'
        imgIOSm.ImgPosPath = os.path.join(imgIOSm.ImgSmPath, subDir)
        imgIOSm.posIdx = posIdx
        imgIO = loopT(imgIOSm, outputChann, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol)
    return imgIO                
        
def loopT(imgIO, outputChann, flatField=False, bgCorrect=True, flipPol=False):
    for tIdx in range(0,imgIO.nTime):        
        imgIO.tIdx = tIdx
        if imgIO.loopZ =='sample':
            imgIO = loopZSm(imgIO, outputChann, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol)
        else:
            imgIO = loopZBg(imgIO, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol)
    return imgIO        

def loopZSm(imgIO, outputChann, flatField=False, bgCorrect=True, flipPol=False):
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
   
    
    for zIdx in range(0,imgIO.nZ):
        plt.close("all") # close all the figures from the last run
        imgIO.zIdx = zIdx      
        retardMMSm = np.array([])
        azimuthMMSm = np.array([])     
        ImgRawSm, ImgProcSm, ImgFluor, ImgBF = ParseTiffInput(imgIO)            
        ASm, BSm, IAbsSm, DeltaMaskSM = computeAB(imgIO,ImgRawSm)
        if bgCorrect == 'None':                    
            A, B = ASm, BSm
        else:
            A, B = correctBackground(imgIO,ASm,BSm,ImgRawSm, extra=False) # background subtraction             
        retard, azimuth = computeDeltaPhi(imgIO,A,B,DeltaMaskSM,flipPol=flipPol)        
        #retard = removeBubbles(retard)     # remove bright speckles in mounted brain slice images       
#        retardBg, azimuthBg = computeDeltaPhi(Abg, Bbg,flipPol=flipPol)
        if not ImgBF.size: # use brightfield calculated from pol-images if there is no brighfield data
            ImgBF = IAbsSm
        else:
            ImgBF = ImgBF[0,:,:]
            
        for i in range(ImgFluor.shape[0]):
            if np.any(ImgFluor[:,:,i]):  # if the flour channel exists   
                ImgFluor[:,:,i] = ImgFluor[:,:,i]/imgIO.ImgFluorBg[:,:,i]
            
        if ImgProcSm.size:
            retardMMSm =  ImgProcSm[0,:,:]
            azimuthMMSm = ImgProcSm[1,:,:]
        if flatField:
            ImgBF = ImgBF/imgIO.IAbsBg #flat-field correction 
                    
            ## compare python v.s. Polacquisition output#####
#            titles = ['Retardance (MM)','Orientation (MM)','Retardance (Py)','Orientation (Py)']
#            images = [retardMMSm, azimuthMMSm,retard, azimuth]
#            plot_sub_images(images,titles)
#            plt.savefig(os.path.join(acquDirPath,'compare_MM_Py.png'),dpi=200)
            ##################################################################

        imgs = [ImgBF,retard, azimuth, ImgFluor]        
        if not os.path.exists(imgIO.ImgOutPath): # create folder for processed images
            os.makedirs(imgIO.ImgOutPath)
        imgIO, imgs = plot_birefringence(imgIO, imgs,outputChann, spacing=10, vectorScl=2, zoomin=False, dpi=200)
        # imgIO.imgLimits = ImgLimit(imgs,imgIO.imgLimits)
        
        
        
        ##To do: add 'Fluor+Retardance' channel## 
        
        imgIO.writeMetaData()
        exportImg(imgIO, imgs)
    return imgIO     

def loopZBg(imgIO, flatField=False, bgCorrect=True, flipPol=False):           
    for zIdx in range(0,1): # only use the first z 
        imgIO.zIdx = zIdx              
        ImgRawSm, ImgProcSm, ImgFluor, ImgBF = ParseTiffInput(imgIO)            
        for i in range(ImgFluor.shape[0]):
            if np.any(ImgFluor[i,:,:]):  # if the flour channel exists
                imgIO.ImgFluorSum[i,:,:] += cv2.morphologyEx(ImgFluor[i,:,:], cv2.MORPH_OPEN, imgIO.kernel, borderType = cv2.BORDER_REPLICATE )
                imgIO.ImgFluorMin[i,:,:] = ImgMin(ImgFluor[i,:,:], imgIO.ImgFluorMin[i,:,:])
    return imgIO                                       

