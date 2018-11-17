import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import cv2
#import sys
#sys.path.append("..") # Adds higher directory to python modules path.
from utils.imgIO import parse_tiff_input, exportImg, GetSubDirName, FindDirContainPos
from .reconstruct import ImgReconstructor
from utils.imgProcessing import ImgMin
from utils.plotting import plot_sub_images
from utils.mManagerIO import mManagerReader, PolAcquReader


from utils.plotting import plot_birefringence, plot_sub_images
from utils.imgProcessing import ImgLimit



def findBackground(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann, flatField=False, bgCorrect='Auto',
                   recon_method='Stokes', ff_method='open'):
    """
    Estimate background for each channel to perform background substraction for
    birefringence and flat-field correction (division) for bright-field and 
    fluorescence channels
        
    """
    
    ImgSmPath = os.path.join(RawDataPath, ImgDir, SmDir) # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'    
    OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir)
    ImgSmPath = FindDirContainPos(ImgSmPath)
    try:
        img_ioSm = PolAcquReader(ImgSmPath, OutputPath, outputChann)
    except:
        img_ioSm = mManagerReader(ImgSmPath,OutputPath, outputChann)
    img_ioSm.bg_method = 'Global'
    if bgCorrect=='None':
        print('No background correction is performed...')
        img_ioSm.bg_correct = False
    elif bgCorrect=='Input':
        print('Background correction mode set as "Input". Use user input background directory')
        OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir+'_'+BgDir)
        img_ioSm.ImgOutPath = OutputPath
        img_ioSm.bg_correct = True
    elif bgCorrect=='Local':
        print('Background correction mode set as "Local". Use local background estimated from sample images')
        OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + SmDir)
        img_ioSm.ImgOutPath = OutputPath
        img_ioSm.bg_method = 'Local'
        img_ioSm.bg_correct = True
    elif bgCorrect=='Auto':
        if hasattr(img_ioSm, 'bg'):
            if img_ioSm.bg == 'No Background':
                bgCorrect=='None' # need to pass the flag down
                BgDir = SmDir  # need a smarter way to deal with different backgroud options
                img_ioSm.bg_correct = False
                print('No background correction is performed for background measurements...')
            else:
                print('Background info found in metadata. Use background specified in metadata')
                BgDir = img_ioSm.bg
                OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + BgDir)
                img_ioSm.bg_correct = True
        else:
            print('Background not specified in metadata. Use user input background directory')   
            OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir+'_'+BgDir)
            img_ioSm.bg_correct = True
        img_ioSm.ImgOutPath = OutputPath

    ImgBgPath = os.path.join(RawDataPath, ImgDir, BgDir) # Background image folder path, of form 'BG_yyyy_mmdd_hhmm_X'
    img_ioBg = PolAcquReader(ImgBgPath, OutputPath)
    img_ioBg.posIdx = 0 # assuming only single image for background
    img_ioBg.tIdx = 0
    img_ioBg.zIdx = 0
    img_ioBg.recon_method = recon_method
    ImgRawBg, ImgProcBg, ImgFluor, ImgBF = parse_tiff_input(img_ioBg) # 0 for z-index

    img_reconstructor = ImgReconstructor(ImgRawBg, method=recon_method, swing=img_ioBg.swing,
                                         wavelength=img_ioBg.wavelength)
    img_param_bg = img_reconstructor.compute_param(ImgRawBg)
    
    img_ioSm.param_bg = img_param_bg
    img_ioSm.swing = img_ioBg.swing
    img_ioSm.wavelength = img_ioBg.wavelength
    img_ioSm.blackLevel = img_ioBg.blackLevel
    img_ioSm.recon_method = recon_method
    img_ioSm.ImgFluorMin = np.full((4,img_ioBg.height,img_ioBg.width), np.inf) # set initial min array to to be Inf
    img_ioSm.ImgFluorSum = np.zeros((4,img_ioBg.height,img_ioBg.width)) # set the default background to be Ones (uniform field)
    img_ioSm.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))  # kernel for image opening operation, 100-200 is usually good
    img_ioSm.loopZ ='background'
    img_ioSm.ff_method = ff_method
    ImgFluorBg = np.ones((4,img_ioBg.height,img_ioBg.width))
    
    if flatField: # find background flourescence for flatField corection 
        print('Calculating illumination function for flatfield correction...')
        img_ioSm = loopPos(img_ioSm, outputChann, flatField=flatField)
        img_ioSm.ImgFluorSum = img_ioSm.ImgFluorSum/img_ioSm.nPos # normalize the sum
        if ff_method=='open':
            ImgFluorBg = img_ioSm.ImgFluorSum
        elif ff_method=='empty':
            ImgFluorBg = img_ioSm.ImgFluorMin
        
        ## compare empty v.s. open method#####
#        titles = ['Ch1 (Open)','Ch2 (Open)','Ch1 (Empty)','ch2 (Empty)']
#        images = [img_ioSm.ImgFluorSum[:,:,0], img_ioSm.ImgFluorSum[:,:,1],
#                  img_ioSm.ImgFluorMin[:,:,0], img_ioSm.ImgFluorMin[:,:,1]]
#        plot_sub_images(images,titles)
#         print('Exporting illumination function...')
#        plt.savefig(os.path.join(OutputPath,'compare_flat_field.png'),dpi=150)
        ##################################################################
    else:        
        I_trans_Bg = np.ones((img_ioBg.height,img_ioBg.width))  # use uniform field if no correction
    img_ioSm.I_trans_Bg = img_param_bg[0]
    img_ioSm.ImgFluorBg = ImgFluorBg
    return img_ioSm
    
def loopPos(img_ioSm, outputChann, flatField=False, bgCorrect=True, flipPol=False, norm=True):
    """
    Loops through each position in the acquisition folder, and performs flat-field correction.
    
    @param ImgSmPath: Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'
    @param OutputPath: Output folder path
    @param Chi: Swing
    @param Lambda: Wavelength of polarized light.
    @param Abg: A term in background
    @param Bbg: B term in background
    @param I_trans_Bg: another background term.
    @param DAPIBg: another backgruond term.
    @param TdTomatoBg: another background term.
    @param flatField: boolean - whether flatField correction is applied.
    @param bgCorrect: boolean - whether or not background correction is applied.
    @param flipPol: whether or not to flip the sign of polarization.
    @return: None
    """       
    
    
    for posIdx in range(0,img_ioSm.nPos):
    # for posIdx in range(0, 25):
        plt.close("all") # close all the figures from the last run
        if img_ioSm.metaFile['Summary']['InitialPositionList']: # PolAcquisition doens't save position list
            subDir = img_ioSm.metaFile['Summary']['InitialPositionList'][posIdx]['Label']
        else:
            subDir = 'Pos0'
        img_ioSm.ImgPosPath = os.path.join(img_ioSm.ImgSmPath, subDir)
        img_ioSm.posIdx = posIdx
        img_io = loopT(img_ioSm, outputChann, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol, norm=norm)
    return img_io
        
def loopT(img_io, outputChann, flatField=False, bgCorrect=True, flipPol=False, norm=True):
    for tIdx in range(0,img_io.nTime):
        img_io.tIdx = tIdx
        if img_io.loopZ =='sample':
            img_io = loopZSm(img_io, outputChann, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol, norm=norm)
        else:
            img_io = loopZBg(img_io, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol)
    return img_io

def loopZSm(img_io, outputChann, flatField=False, bgCorrect=True, flipPol=False, norm=True):
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
    @param I_trans_Bg: another background term.
    @param DAPIBg: another backgruond term.
    @param TdTomatoBg: another background term.
    @param imgLimits:
    @param flatField: boolean - whether flatField correction is applied.
    @param bgCorrect: boolean - whether or not background correction is applied.
    @param flipPol: whether or not to flip the sign of polarization.
    @return: None
    """
    if not os.path.exists(img_io.ImgOutPath):  # create folder for processed images
        os.makedirs(img_io.ImgOutPath)
    tIdx = img_io.tIdx
    posIdx = img_io.posIdx
    for zIdx in range(0,img_io.nZ):
        print('Processing position %03d, time %03d, z %03d ...' % (posIdx, tIdx, zIdx))
        plt.close("all") # close all the figures from the last run
        img_io.zIdx = zIdx
        retardMMSm = np.array([])
        azimuthMMSm = np.array([])     
        ImgRawSm, ImgProcSm, ImgFluor, ImgBF = parse_tiff_input(img_io)
        img_reconstructor = ImgReconstructor(ImgRawSm, method=img_io.recon_method, swing=img_io.swing,
                                             wavelength=img_io.wavelength, kernel=img_io.kernel)
        img_param_sm = img_reconstructor.compute_param(ImgRawSm)

        if img_io.bg_correct:
            img_param_sm = img_reconstructor.correct_background(img_param_sm, img_io.param_bg, method=img_io.bg_method,
                                                                extra=False) # background subtraction
        if img_io.bg_method == 'Local':
            img_param_bg_local = []
            print('Estimating local background...')
            for img in img_param_sm:
                img_param_bg_local += [cv2.GaussianBlur(img, (401, 401), 0)]
            img_param_bg = img_param_bg_local
            img_param_sm = img_reconstructor.correct_background(img_param_sm, img_param_bg, method=img_io.bg_method,
                                                                extra=False)  # background subtraction

        # titles = ['polarization', 'A', 'B', 'dAB']
        # plot_sub_images(img_param_sm[1:], titles, img_io)
        I_trans_Sm = img_param_sm[0]
        retard, azimuth, polarization = img_reconstructor.reconstruct_img(img_param_sm,flipPol=flipPol)
        #retard = removeBubbles(retard)     # remove bright speckles in mounted brain slice images
        if isinstance(ImgBF, np.ndarray):
            if flatField:
                ImgBF = ImgBF[0,:,:] / img_io.param_bg[0]  # flat-field correction
            else:
                ImgBF = ImgBF[0, :, :]
        else:   # use brightfield calculated from pol-images if there is no brightfield data
            ImgBF = I_trans_Sm

        for i in range(ImgFluor.shape[0]):
            if np.any(ImgFluor[:,:,i]):  # if the flour channel exists   
                ImgFluor[:,:,i] = ImgFluor[:,:,i]/img_io.ImgFluorBg[:,:,i]
            
        if isinstance(ImgProcSm, np.ndarray):
            retardMMSm =  ImgProcSm[0,:,:]
            azimuthMMSm = ImgProcSm[1,:,:]

                    
            ## compare python v.s. Polacquisition output#####
#            titles = ['Retardance (MM)','Orientation (MM)','Retardance (Py)','Orientation (Py)']
#            images = [retardMMSm, azimuthMMSm,retard, azimuth]
#            plot_sub_images(images,titles)
#            plt.savefig(os.path.join(acquDirPath,'compare_MM_Py.png'),dpi=200)
            ##################################################################

        imgs = [ImgBF,retard, azimuth, polarization, ImgFluor]

        img_io, imgs = plot_birefringence(img_io, imgs, outputChann, spacing=20, vectorScl=2, zoomin=False, dpi=200,
                                         norm=norm, plot=True)
        # img_io.imgLimits = ImgLimit(imgs,img_io.imgLimits)
        
        
        
        ##To do: add 'Fluor+Retardance' channel## 
        
        img_io.writeMetaData()
        exportImg(img_io, imgs)
    return img_io

def loopZBg(img_io, flatField=False, bgCorrect=True, flipPol=False):
    for zIdx in range(0,1): # only use the first z 
        img_io.zIdx = zIdx
        ImgRawSm, ImgProcSm, ImgFluor, ImgBF = parse_tiff_input(img_io)
        for i in range(ImgFluor.shape[0]):
            if np.any(ImgFluor[i,:,:]):  # if the flour channel exists
                if img_io.ff_method == 'open':
                    img_io.ImgFluorSum[i,:,:] += cv2.morphologyEx(ImgFluor[i,:,:], cv2.MORPH_OPEN, img_io.kernel, borderType = cv2.BORDER_REPLICATE)
                elif img_io.ff_method == 'empty':
                    img_io.ImgFluorMin[i,:,:] = ImgMin(ImgFluor[i,:,:], img_io.ImgFluorMin[i,:,:])
    return img_io

