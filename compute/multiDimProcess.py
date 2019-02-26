import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import cv2
import time
from utils.imgIO import parse_tiff_input, exportImg, GetSubDirName, FindDirContainPos
from .reconstruct import ImgReconstructor
from utils.imgProcessing import ImgMin
from utils.plotting import plot_sub_images
from utils.mManagerIO import mManagerReader, PolAcquReader
from utils.plotting import plot_birefringence, plot_sub_images
from utils.imgProcessing import ImgLimit
from skimage.restoration import denoise_tv_chambolle




def findBackground(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann,
                   BgDir_local=None, flatField=False, bgCorrect='Auto',
                   ff_method='open'):
    """
    Estimate background for each channel to perform background substraction for
    birefringence and flat-field correction (division) for bright-field and
    fluorescence channels

    """

    ImgSmPath = os.path.join(RawDataPath, ImgDir, SmDir)  # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'
    OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir)
    ImgSmPath = FindDirContainPos(ImgSmPath)
    try:
        img_io = PolAcquReader(ImgSmPath, OutputPath, outputChann=outputChann)
    except:
        img_io = mManagerReader(ImgSmPath, OutputPath, outputChann=outputChann)
    ImgBgPath = os.path.join(RawDataPath, ImgDir, BgDir)  # Background image folder path, of form 'BG_yyyy_mmdd_hhmm_X'
    img_ioBg = PolAcquReader(ImgBgPath, OutputPath)
    img_ioBg.posIdx = 0  # assuming only single image for background
    img_ioBg.tIdx = 0
    img_ioBg.zIdx = 0
    img_io.bg_method = 'Global'

    if bgCorrect == 'None':
        print('No background correction is performed...')
        img_io.bg_correct = False
    else:
        if bgCorrect == 'Input':
            print('Background correction mode set as "Input". Use user input background directory')
            OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + BgDir)
            img_io.ImgOutPath = OutputPath
            img_io.bg_correct = True
        elif bgCorrect == 'Local_filter':
            print('Background correction mode set as "Local_filter". Additional background correction using local '
                  'background estimated from sample images will be performed')
            OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + SmDir)
            img_io.ImgOutPath = OutputPath
            img_io.bg_method = 'Local_filter'
            img_io.bg_correct = True
        elif bgCorrect == 'Local_defocus':
            print('Background correction mode set as "Local_defocus". Use images from' + BgDir_local +
                  'at the same position as background')
            img_bg_path = os.path.join(RawDataPath, ImgDir,
                                       BgDir_local)
            OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir + '_' + BgDir_local)
            img_io.ImgOutPath = OutputPath
            img_io.bg_method = 'Local_defocus'
            img_io.bg_correct = True
            img_io_bg_local = mManagerReader(img_bg_path, OutputPath)
            img_io_bg_local.blackLevel = img_ioBg.blackLevel
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
            img_io.ImgOutPath = OutputPath

    ImgRawBg, ImgProcBg, ImgFluor, ImgBF = parse_tiff_input(img_ioBg)  # 0 for z-index
    img_reconstructor = ImgReconstructor(ImgRawBg, bg_method=img_io.bg_method, swing=img_ioBg.swing,
                                         wavelength=img_ioBg.wavelength, output_path=img_ioBg.ImgOutPath)
    if img_io.bg_correct:
        img_stokes_bg = img_reconstructor.compute_stokes(ImgRawBg)
        # print('denoising the background...')
        # img_stokes_bg = [denoise_tv_chambolle(img, weight=10**6) for img in img_stokes_bg]
        # img_stokes_bg = [cv2.GaussianBlur(img, (5, 5), 0) for img in img_stokes_bg]
        # img_stokes_bg = [cv2.medianBlur(img, 5) for img in img_stokes_bg]
    else:
        img_stokes_bg = None

    img_io.param_bg = img_stokes_bg
    img_io.swing = img_ioBg.swing
    img_io.wavelength = img_ioBg.wavelength
    img_io.blackLevel = img_ioBg.blackLevel
    img_io.ImgFluorMin = np.full((4, img_ioBg.height, img_ioBg.width), np.inf)  # set initial min array to to be Inf
    img_io.ImgFluorSum = np.zeros(
        (4, img_ioBg.height, img_ioBg.width))  # set the default background to be Ones (uniform field)
    img_io.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (100, 100))  # kernel for image opening operation, 100-200 is usually good
    img_io.loopZ = 'background'
    img_io.ff_method = ff_method
    ImgFluorBg = np.ones((4, img_ioBg.height, img_ioBg.width))

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
        I_trans_Bg = np.ones((img_ioBg.height, img_ioBg.width))  # use uniform field if no correction
    img_io.I_trans_Bg = img_stokes_bg[0]
    img_io.ImgFluorBg = ImgFluorBg
    return img_io, img_reconstructor


def loopPos(img_io, img_reconstructor, flatField=False, bgCorrect=True, circularity='rcp', norm=True):
    """
    Loops through each position in the acquisition folder, and performs flat-field correction.
    @param flatField: boolean - whether flatField correction is applied.
    @param bgCorrect: boolean - whether or not background correction is applied.
    @param circularity: whether or not to flip the sign of polarization.
    @return: None
    """

    for posIdx in range(0, img_io.nPos):
        plt.close("all")  # close all the figures from the last run
        if img_io.metaFile['Summary']['InitialPositionList']:  # PolAcquisition doens't save position list
            subDir = img_io.metaFile['Summary']['InitialPositionList'][posIdx]['Label']
        else:
            subDir = 'Pos0'
        img_io.ImgPosPath = os.path.join(img_io.ImgSmPath, subDir)
        if img_io.bg_method == 'Local_defocus':
            img_io_bg = img_io.bg_local
            img_io_bg.ImgPosPath = os.path.join(img_io_bg.ImgSmPath, subDir)
            img_io_bg.posIdx = posIdx
        img_io.posIdx = posIdx
        img_io = loopT(img_io, img_reconstructor, flatField=flatField, bgCorrect=bgCorrect, circularity=circularity,
                       norm=norm)
    return img_io


def loopT(img_io, img_reconstructor, flatField=False, bgCorrect=True, circularity='rcp', norm=True):
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
            img_io = loopZSm(img_io, img_reconstructor, circularity=circularity, norm=norm)

        else:
            img_io = loopZBg(img_io)
    return img_io


def loopZSm(img_io, img_reconstructor, circularity='rcp', norm=True):
    """
    Loops through Z.

    @param flatField: boolean - whether flatField correction is applied.
    @param bgCorrect: boolean - whether or not background correction is applied.
    @param circularity: whether or not to flip the sign of polarization.
    @return: None
    """
    if not os.path.exists(img_io.ImgOutPath):  # create folder for processed images
        os.makedirs(img_io.ImgOutPath)
    tIdx = img_io.tIdx
    posIdx = img_io.posIdx
    img_stokes_bg = img_io.param_bg
    for zIdx in range(0, img_io.nZ):
        # for zIdx in range(0, 1):
        print('Processing position %03d, time %03d, z %03d ...' % (posIdx, tIdx, zIdx))
        plt.close("all")  # close all the figures from the last run
        img_io.zIdx = zIdx
        retardMMSm = np.array([])
        azimuthMMSm = np.array([])
        start = time.time()
        ImgRawSm, ImgProcSm, ImgFluor, ImgBF = parse_tiff_input(img_io)
        stop = time.time()
        # print('parse_tiff_input takes {:.1f} ms ...'.format((stop - start) * 1000))
        start = time.time()
        img_stokes_sm = img_reconstructor.compute_stokes(ImgRawSm)
        stop = time.time()
        # print('compute_stokes takes {:.1f} ms ...'.format((stop - start) * 1000))
        start = time.time()
        img_computed_sm = img_reconstructor.reconstruct_birefringence(img_stokes_sm, img_stokes_bg,
                                                                      circularity=circularity,
                                                                      extra=False)  # background subtraction
        stop = time.time()
        # print('reconstruct_birefringence takes {:.1f} ms ...'.format((stop - start) * 1000))
        [I_trans, retard, azimuth, polarization] = img_computed_sm

        # titles = ['polarization', 'A', 'B', 'dAB']
        # plot_sub_images(img_param_sm[1:], titles, img_io)

        # retard = removeBubbles(retard)     # remove bright speckles in mounted brain slice images
        if isinstance(ImgBF, np.ndarray):
            ImgBF = ImgBF[0, :, :] / img_io.param_bg[0]  # flat-field correction

        else:  # use brightfield calculated from pol-images if there is no brightfield data
            ImgBF = I_trans

        for i in range(ImgFluor.shape[0]):
            # print(np.nanmax(ImgFluor[i,:,:][:]))
            # print(np.nanmin(img_io.ImgFluorBg[i,:,:][:]))
            if np.any(ImgFluor[i, :, :]):  # if the flour channel exists
                ImgFluor[i, :, :] = ImgFluor[i, :, :] / img_io.ImgFluorBg[i, :, :]

        if isinstance(ImgProcSm, np.ndarray):
            retardMMSm = ImgProcSm[0, :, :]
            azimuthMMSm = ImgProcSm[1, :, :]

            ## compare python v.s. Polacquisition output#####
        #            titles = ['Retardance (MM)','Orientation (MM)','Retardance (Py)','Orientation (Py)']
        #            images = [retardMMSm, azimuthMMSm,retard, azimuth]
        #            plot_sub_images(images,titles)
        #            plt.savefig(os.path.join(acquDirPath,'compare_MM_Py.png'),dpi=200)
        ##################################################################

        imgs = [ImgBF, retard, azimuth, polarization, ImgFluor]
        start = time.time()
        img_io, imgs = plot_birefringence(img_io, imgs, img_io.chNamesOut, spacing=20, vectorScl=2, zoomin=False,
                                          dpi=200,
                                          norm=norm, plot=True)
        stop = time.time()
        # print('plot_birefringence takes {:.1f} ms ...'.format((stop - start) * 1000))
        # img_io.imgLimits = ImgLimit(imgs,img_io.imgLimits)
        ##To do: add 'Fluor+Retardance' channel##
        start = time.time()
        exportImg(img_io, imgs)
        stop = time.time()
        # print('exportImg takes {:.1f} ms ...'.format((stop - start) * 1000))

        # titles = ['s0', 's1', 's2', 's3']
        # fig_name = 'stokes_sm_t%03d_p%03d_z%03d.jpg' % (tIdx, posIdx, zIdx)
        # plot_sub_images(img_stokes_sm, titles, img_io.ImgOutPath, fig_name, colorbar=True)
        # fig_name = 'stokes_bg_t%03d_p%03d_z%03d.jpg' % (tIdx, posIdx, zIdx)
        # plot_sub_images(img_stokes_bg, titles, img_io.ImgOutPath, fig_name, colorbar=True)
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

