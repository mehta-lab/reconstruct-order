import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import cv2
import time
import copy
from utils.imgIO import parse_tiff_input, exportImg, GetSubDirName, FindDirContainPos
from .reconstruct import ImgReconstructor
from utils.imgProcessing import ImgMin
from utils.plotting import plot_birefringence, plot_stokes, plot_pol_imgs, plot_Polacquisition_imgs
from utils.mManagerIO import mManagerReader, PolAcquReader
from utils.imgProcessing import ImgLimit, imBitConvert
from skimage.restoration import denoise_tv_chambolle

def findBackground(RawDataPath, ProcessedPath, ImgDir, SmDir, PosList, BgDir, outputChann,
                   BgDir_local=None, flatField=False, bgCorrect='Auto',
                   ff_method='open', azimuth_offset = 0):
    """
    Estimate background for each channel to perform background substraction for
    birefringence and flat-field correction (division) for bright-field and
    fluorescence channels

    """

    ImgSmPath = os.path.join(RawDataPath, ImgDir, SmDir)  # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'
    OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir)
    ImgSmPath = FindDirContainPos(ImgSmPath)
    try:
        img_io = PolAcquReader(ImgSmPath, OutputPath)
    except:
        img_io = mManagerReader(ImgSmPath, OutputPath, outputChann=outputChann)
    ImgBgPath = os.path.join(RawDataPath, ImgDir, BgDir)  # Background image folder path, of form 'BG_yyyy_mmdd_hhmm_X'
    img_io_bg = PolAcquReader(ImgBgPath, OutputPath)
    img_io_bg.posIdx = 0  # assuming only single image for background
    img_io_bg.tIdx = 0
    img_io_bg.zIdx = 0
    img_io.bg_method = 'Global'

    if bgCorrect == 'None':
        print('No background correction is performed...')
        img_io.bg_correct = False
    else:
        if bgCorrect == 'Input':
            print('Background correction mode set as "Input". Use user input background directory')
            OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + BgDir)
            img_io.bg_correct = True
        elif bgCorrect == 'Local_filter':
            print('Background correction mode set as "Local_filter". Additional background correction using local '
                  'background estimated from sample images will be performed')
            OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + SmDir)
            img_io.bg_method = 'Local_filter'
            img_io.bg_correct = True
        elif bgCorrect == 'Local_defocus':
            print('Background correction mode set as "Local_defocus". Use images from' + BgDir_local +
                  'at the same position as background')
            img_bg_path = os.path.join(RawDataPath, ImgDir,
                                       BgDir_local)
            OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + BgDir_local)
            img_io.bg_method = 'Local_defocus'
            img_io.bg_correct = True
            img_io_bg_local = mManagerReader(img_bg_path, OutputPath)
            img_io_bg_local.blackLevel = img_io_bg.blackLevel
            img_io.bg_local = img_io_bg_local

        elif bgCorrect == 'Auto':
            if hasattr(img_io, 'bg'):
                if img_io.bg == 'No Background':
                    BgDir = SmDir
                    img_io.bg_correct = False
                    print('No background correction is performed for background measurements...')
                else:
                    print('Background info found in metadata. Use background specified in metadata')
                    BgDir = img_io.bg
                    OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + BgDir)
                    img_io.bg_correct = True
            else:
                print('Background not specified in metadata. Use user input background directory')
                OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + BgDir)
                img_io.bg_correct = True
    # OutputPath = OutputPath + '_pol'
    img_io.ImgOutPath = OutputPath
    os.makedirs(OutputPath, exist_ok=True)  # create folder for processed images
    ImgRawBg, ImgProcBg, ImgFluor, ImgBF = parse_tiff_input(img_io_bg)  # 0 for z-index
    img_io.img_raw_bg = ImgRawBg
    img_reconstructor = ImgReconstructor(ImgRawBg, bg_method=img_io.bg_method, swing=img_io_bg.swing,
                                         wavelength=img_io_bg.wavelength, output_path=img_io_bg.ImgOutPath,
                                         azimuth_offset=azimuth_offset)
    if img_io.bg_correct:
        img_stokes_bg = img_reconstructor.compute_stokes(ImgRawBg)        
        # print('denoising the background...')
        # img_stokes_bg = [denoise_tv_chambolle(img, weight=10**6) for img in img_stokes_bg]
        # img_stokes_bg = [cv2.GaussianBlur(img, (5, 5), 0) for img in img_stokes_bg]
        # img_stokes_bg = [cv2.medianBlur(img, 5) for img in img_stokes_bg]
    else:
        img_stokes_bg = None

    img_io.param_bg = img_stokes_bg
    img_io.swing = img_io_bg.swing
    img_io.wavelength = img_io_bg.wavelength
    img_io.blackLevel = img_io_bg.blackLevel
    img_io.ImgFluorMin = np.full((4, img_io_bg.height, img_io_bg.width), np.inf)  # set initial min array to to be Inf
    img_io.ImgFluorSum = np.zeros(
        (4, img_io_bg.height, img_io_bg.width))  # set the default background to be Ones (uniform field)
    img_io.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (100, 100))  # kernel for image opening operation, 100-200 is usually good
    img_io.loopZ = 'background'
    img_io.PosList = PosList
    img_io.ff_method = ff_method
    ImgFluorBg = np.ones((4, img_io_bg.height, img_io_bg.width))

    if flatField:  # find background flourescence for flatField corection
        print('Calculating illumination function for flatfield correction...')
        img_io = loopPos(img_io, outputChann, flatField=flatField)
        img_io.ImgFluorSum = img_io.ImgFluorSum / img_io.nPos  # normalize the sum. Add 1 to avoid 0
        if ff_method == 'open':
            ImgFluorBg = img_io.ImgFluorSum
        elif ff_method == 'empty':
            ImgFluorBg = img_io.ImgFluorMin
        ImgFluorBg = ImgFluorBg - min(np.nanmin(ImgFluorBg[:]), 0) + 1
        ## compare empty v.s. open method#####
    #        titles = ['Ch1 (Open)','Ch2 (Open)','Ch1 (Empty)','ch2 (Empty)']
    #        images = [img_io.ImgFluorSum[:,:,0], img_io.ImgFluorSum[:,:,1],
    #                  img_io.ImgFluorMin[:,:,0], img_io.ImgFluorMin[:,:,1]]
    #        plot_sub_images(images,titles)
    #         print('Exporting illumination function...')
    #        plt.savefig(os.path.join(OutputPath,'compare_flat_field.png'),dpi=150)
    ##################################################################
    else:
        I_trans_Bg = np.ones((img_io_bg.height, img_io_bg.width))  # use uniform field if no correction
    img_io.I_trans_Bg = img_stokes_bg[0]
    img_io.ImgFluorBg = ImgFluorBg
    return img_io, img_reconstructor


def loopPos(img_io, img_reconstructor, plot_config, flatField=False, bgCorrect=True,
            circularity='rcp', separate_pos=True):
    """
    Loops through each position in the acquisition folder, and performs flat-field correction.
    @param flatField: boolean - whether flatField correction is applied.
    @param bgCorrect: boolean - whether or not background correction is applied.
    @param circularity: whether or not to flip the sign of polarization.
    @return: None
    """
    
    try:
        posDict = {idx: img_io.metaFile['Summary']['InitialPositionList'][idx]['Label'] for idx in range(img_io.nPos)}
    except:
        # PolAcquisition doens't save position list
        posDict = {0:'Pos0'}

    for posIdx, pos_name in posDict.items():
        if pos_name in img_io.PosList or img_io.PosList == 'all':
            plt.close("all")  # close all the figures from the last run            
            img_io.img_in_pos_path = os.path.join(img_io.ImgSmPath, pos_name)
            img_io.pos_name = pos_name
            if separate_pos:
                img_io.img_out_pos_path = os.path.join(img_io.ImgOutPath, pos_name)
                os.makedirs(img_io.img_out_pos_path, exist_ok=True)  # create folder for processed images
            else:
                img_io.img_out_pos_path = img_io.ImgOutPath
    
            if img_io.bg_method == 'Local_defocus':
                img_io_bg = img_io.bg_local
                img_io_bg.pos_name = os.path.join(img_io_bg.ImgSmPath, pos_name)
                img_io_bg.posIdx = posIdx
            img_io.posIdx = posIdx
    
            img_io = loopT(img_io, img_reconstructor, plot_config, circularity=circularity)
    return img_io


def loopT(img_io, img_reconstructor, plot_config, circularity='rcp'):
    for tIdx in range(0, img_io.nTime):
        img_io.tIdx = tIdx
        if img_io.loopZ == 'sample':
            if img_io.bg_method == 'Local_defocus':
                img_io_bg = img_io.bg_local
                print('compute defocused backgorund at pos{} ...'.format(img_io_bg.posIdx))
                img_io_bg.tIdx = tIdx
                img_io_bg.zIdx = 0
                ImgRawBg, ImgProcBg, ImgFluor, ImgBF = parse_tiff_input(img_io_bg)  # 0 for z-index
                img_stokes_bg = img_reconstructor.compute_stokes(ImgRawBg)
                img_io.param_bg = img_stokes_bg
            img_io = loopZSm(img_io, img_reconstructor, plot_config, circularity=circularity)

        else:
            img_io = loopZBg(img_io)
    return img_io


def loopZSm(img_io, img_reconstructor, plot_config, circularity='rcp'):
    """
    Loops through Z.

    @param flatField: boolean - whether flatField correction is applied.
    @param bgCorrect: boolean - whether or not background correction is applied.
    @param circularity: whether or not to flip the sign of polarization.
    @return: None
    """

    tIdx = img_io.tIdx
    posIdx = img_io.posIdx

    for zIdx in range(0, img_io.nZ):
        # for zIdx in range(0, 1):
        print('Processing position %03d, time %03d, z %03d ...' % (posIdx, tIdx, zIdx))
        plt.close("all")  # close all the figures from the last run
        img_io.zIdx = zIdx
        if 'norm' in plot_config:
            norm = plot_config['norm']
        save_fig = False
        if 'save_fig' in plot_config:
            save_fig = plot_config['save_fig']
        save_stokes_fig = False
        if 'save_stokes_fig' in plot_config:
            save_stokes_fig = plot_config['save_stokes_fig']
        save_pol_fig = False
        if 'save_pol_fig' in plot_config:
            save_pol_fig = plot_config['save_pol_fig']
        save_mm_fig = False
        if 'save_mm_fig' in plot_config:
            save_mm_fig = plot_config['save_mm_fig']

        ImgRawSm, ImgProcSm, ImgFluor, ImgBF = parse_tiff_input(img_io)

        for i in range(ImgFluor.shape[0]):
            # print(np.nanmax(ImgFluor[i,:,:][:]))
            # print(np.nanmin(img_io.ImgFluorBg[i,:,:][:]))
            if np.any(ImgFluor[i, :, :]):  # if the flour channel exists
                ImgFluor[i, :, :] = ImgFluor[i, :, :] / img_io.ImgFluorBg[i, :, :]

        pol_names = ['Pol_State_0', 'Pol_State_1', 'Pol_State_2', 'Pol_State_3', 'Pol_State_4']
        stokes_names = ['Stokes_0', 'Stokes_1', 'Stokes_2', 'Stokes_3']
        stokes_names_sm = [x + '_sm' for x in stokes_names]
        birefring_names = ['Transmission', 'Retardance','+Orientation',
            'Retardance+Orientation', 'Scattering+Orientation', 'Transmission+Retardance+Orientation',
         'Retardance+Fluorescence', 'Retardance+Fluorescence_all']

        img_dict = {}
        if any(chan in stokes_names + stokes_names_sm + birefring_names
               for chan in img_io.chNamesOut) or any([save_fig, save_stokes_fig, save_mm_fig]):
            img_stokes_sm = img_reconstructor.compute_stokes(ImgRawSm)
            [s0, retard, azimuth, polarization, s1, s2, s3] = \
                img_reconstructor.reconstruct_birefringence(img_stokes_sm, img_io.param_bg,
                                                                          circularity=circularity,
                                                                          extra=False)  # background subtraction
            img_stokes = [s0, s1, s2, s3]
            # retard = removeBubbles(retard)     # remove bright speckles in mounted brain slice images
            if isinstance(ImgBF, np.ndarray):
                ImgBF = ImgBF[0, :, :] / img_io.param_bg[0]  # flat-field correction

            else:  # use brightfield calculated from pol-images if there is no brightfield data
                ImgBF = s0
            if isinstance(ImgProcSm, np.ndarray):
                retardMMSm = ImgProcSm[0, :, :]
                azimuthMMSm = ImgProcSm[1, :, :]
                if save_mm_fig:
                    imgs_mm_py = [retardMMSm, azimuthMMSm, retard, azimuth]
                    plot_Polacquisition_imgs(img_io, imgs_mm_py)
            if any(chan in birefring_names for chan in img_io.chNamesOut) or save_fig:
                imgs = [ImgBF, retard, azimuth, polarization, ImgFluor]
                img_io, img_dict = plot_birefringence(img_io, imgs, spacing=20, vectorScl=2, zoomin=False,
                                                  dpi=200,
                                                  norm=norm, plot=save_fig)
            if any(chan in stokes_names + stokes_names_sm for chan in img_io.chNamesOut) or save_stokes_fig:
                if save_stokes_fig:
                    plot_stokes(img_io, img_stokes, img_stokes_sm)
                img_stokes = [x.astype(np.float32, copy=False) for x in img_stokes]
                img_stokes_sm = [x.astype(np.float32, copy=False) for x in img_stokes_sm]
                img_stokes_dict = dict(zip(stokes_names, img_stokes))
                img_stokes_sm_dict = dict(zip(stokes_names_sm, img_stokes_sm))
                img_dict.update(img_stokes_dict)
                img_dict.update(img_stokes_sm_dict)

        if any(chan in pol_names for chan in img_io.chNamesOut) or save_pol_fig:
            imgs_pol = []
            for i in range(ImgRawSm.shape[0]):
                imgs_pol += [ImgRawSm[i, ...] / img_io.img_raw_bg[i, ...]]
            if save_pol_fig:
                plot_pol_imgs(img_io, imgs_pol, pol_names)
            imgs_pol = [imBitConvert(img * 10 ** 4, bit=16) for img in imgs_pol]
            img_dict.update(dict(zip(pol_names, imgs_pol)))
        exportImg(img_io, img_dict)
    return img_io

def loopZBg(img_io):
    for zIdx in range(0, 1):  # only use the first z
        img_io.zIdx = zIdx
        ImgRawSm, ImgProcSm, ImgFluor, ImgBF = parse_tiff_input(img_io)
        for i in range(ImgFluor.shape[0]):
            if np.any(ImgFluor[i, :, :]):  # if the flour channel exists
                if img_io.ff_method == 'open':
                    img_io.ImgFluorSum[i, :, :] += cv2.morphologyEx(ImgFluor[i, :, :], cv2.MORPH_OPEN, img_io.kernel,
                                                                    borderType=cv2.BORDER_REPLICATE)
                elif img_io.ff_method == 'empty':
                    img_io.ImgFluorMin[i, :, :] = ImgMin(ImgFluor[i, :, :], img_io.ImgFluorMin[i, :, :])
    return img_io

