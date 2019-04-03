import os
import numpy as np
from utils.mManagerIO import mManagerReader
from utils.imgProcessing import imBitConvert

if __name__ == '__main__':
    RawDataPath = '/flexo/ComputationalMicroscopy/Projects/brainarchitecture'
    ProcessedPath = RawDataPath
    ImgDir = '2019_01_04_david_594CTIP2_647SATB2_20X'
    SmDir = 'SMS_2019_0104_1257_2_SMS_2019_0104_1257_2'
    input_chan = ['Orientation']
    output_chan = ['Orientation_x', 'Orientation_y']
    ImgSmPath = os.path.join(RawDataPath, ImgDir, SmDir)  # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'
    OutputPath = ImgSmPath
    img_io = mManagerReader(ImgSmPath, OutputPath, inputChann=input_chan, outputChann=output_chan)

    for t_idx in range(img_io.nTime):
        img_io.tIdx = t_idx
        for pos_idx in range(img_io.nPos):  # nXY
            img_io.posIdx = pos_idx
            for z_idx in range(img_io.nZ):
                print('Processing position %03d, time %03d z %03d ...' % (pos_idx, t_idx, z_idx))
                img_io.zIdx = z_idx
                img_io.chanIdx = 0
                azimuth_degree = img_io.read_img()
                azimuth = azimuth_degree/18000*np.pi
                azimuth_imgs = [np.cos(2 * azimuth), np.sin(2 * azimuth)]
                azimuth_imgs = [imBitConvert((img + 1) * 1000, bit=16) for img in azimuth_imgs]  # scale to [0, 1000]
                for chan_idx, image in enumerate(azimuth_imgs):
                    img_io.chanIdx = chan_idx
                    img_io.write_img(image)



