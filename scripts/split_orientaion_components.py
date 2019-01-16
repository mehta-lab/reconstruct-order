import sys
sys.path.append(".") # Adds current directory to python search path.
sys.path.append("..") # Adds parent directory to python search path.

import os
import numpy as np
from utils.mManagerIO import mManagerReader
from utils.imgProcessing import imBitConvert

if __name__ == '__main__':
    # RawDataPath = r'D:/Box Sync/Data'
    # ProcessedPath = r'D:/Box Sync/Processed/'
    RawDataPath = '/data/sguo/Processed'
    ProcessedPath = '/data/sguo/Processed'
    ImgDir = '2018_11_01_kidney_slice'
    SmDir = 'SMS_2018_1126_1625_1_BG_2018_1126_1621_1'
    outputChann = ['Orientation']

    ImgSmPath = os.path.join(ProcessedPath, ImgDir, SmDir)  # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'
    OutputPath = ImgSmPath
    img_io = mManagerReader(ImgSmPath, OutputPath, outputChann)

    for t_idx in range(img_io.nTime):
        img_io.tIdx = t_idx
        for pos_idx in range(img_io.nPos):  # nXY
            img_io.posIdx = pos_idx
            for z_idx in range(img_io.nZ):
                img_io.chanIdx=0
                azimuth_degree = img_io.read_img()
                azimuth = azimuth_degree/18000*np.pi
                azimuth_x = np.cos(2 * azimuth)
                azimuth_y = np.sin(2 * azimuth)
                azimuth_x = imBitConvert((azimuth_x + 1) * 1000, bit=16)  # scale to [0, 1000]
                azimuth_y = imBitConvert((azimuth_y + 1) * 1000, bit=16)  # scale to [0, 1000]
                img_io.write_img(azimuth_x, 'Orientation_x', pos_idx, z_idx, t_idx)
                img_io.write_img(azimuth_y, 'Orientation_y', pos_idx, z_idx, t_idx)


