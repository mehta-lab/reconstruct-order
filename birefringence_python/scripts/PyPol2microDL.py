# -*- coding: utf-8 -*-
"""
Script to covert PyPolScope output for to microDL input format 
"""
import os
from utils.mManagerIO import mManagerReader

RawDataPath = 'C:/Data'

ProcessedPath = 'C:/Processed'

ImgDir = '2018_07_03_KidneyTissueSection'
SmDir = 'SMS_2018_0703_1835_1_BG_2018_0703_1829_1'


ImgSmPath = os.path.join(ProcessedPath, ImgDir, SmDir) # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'

OutputPath = os.path.join(ImgSmPath,'split_images')
imgSm = mManagerReader(ImgSmPath,OutputPath)
imgSm.save_images()
