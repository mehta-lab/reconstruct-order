# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:12:20 2018

@author: Sam Guo
"""
import os
from utils.mManagerIO import mManagerReader


RawDataPath = '/data/sguo/Data'
ProcessedPath = '/data/sguo/Processed'

ImgDir = '2018_11_01_kidney_slice'
SmDir = 'SMS_2018_1101_1906_1_BG_2018_1101_1906_1'
outputChann = ['Transmission', 'Retardance', 'Orientation', 'Scattering', '405', '488', '568']

ImgSmPath = os.path.join(ProcessedPath, ImgDir, SmDir) # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'

OutputPath = os.path.join(ImgSmPath,'split_images')
imgSm = mManagerReader(ImgSmPath,OutputPath, outputChann)
imgSm.save_images()
