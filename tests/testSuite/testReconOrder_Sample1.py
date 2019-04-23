#!/usr/bin/env python
# title           : testReconOrder_Sample1.py
# description     :This will create a header for a python script.
# author          :bryant.chhun
# date            :12/12/18
# version         :0.0
# usage           :python this_python_file.py -flags
# notes           :
# python_version  :3.6

import unittest
import os
import cv2
import numpy as np

from workflow.multiDimProcess import process_background, create_metadata_object, parse_bg_options
from utils.ConfigReader import ConfigReader
from utils.imgIO import parse_tiff_input
from utils.imgProcessing import imBitConvert
from tests.testMetrics import mse


'''
Methods to check that reconstruction procedure correctly constructs test data.
'''

class TestImageReconstruction(unittest.TestCase):

    targetData = "../testData/reconData/2018_10_02_MouseBrainSlice/"
    condition = "SM_2018_1002_1633_1_BG_2018_1002_1625_1"

    target_ITrans = targetData + \
                         condition + \
                         "/img_Transmission_t000_p000_z000.tif"
    target_retard = targetData + \
                         condition + \
                         "/img_Retardance_t000_p000_z000.tif"
    target_Orientation = targetData + \
                         condition + \
                         "/img_Orientation_t000_p000_z000.tif"
    target_Scattering = targetData + \
                         condition + \
                         "/img_Scattering_t000_p000_z000.tif"

    source_config_file = '../../config/config_MouseBrainSlice1_workflow_test.yml'
    def __init__(self, *args, **kwargs):
        '''
        Loads source/raw data configuration file.
        '''
        super(TestImageReconstruction, self).__init__(*args, **kwargs)

        self.config = ConfigReader()
        self.config.read_config(self.source_config_file)

        split_data_dir = os.path.split(self.config.dataset.data_dir)
        self.RawDataPath = split_data_dir[0]
        self.ProcessedPath = self.config.dataset.processed_dir
        self.ImgDir = split_data_dir[1]

        self.SmDirList = self.config.dataset.samples
        self.BgDirList = self.config.dataset.background
        self.PosList = self.config.dataset.positions
        #this test data has one element per list

        self.SmDir = self.SmDirList[0]
        self.BgDir = self.BgDirList[0]
        self.outputChann = self.config.processing.output_channels
        self.circularity = self.config.processing.circularity
        self.bgCorrect = self.config.processing.background_correction
        self.flatField = self.config.processing.flatfield_correction
        self.norm = self.config.plotting.normalize_color_images

    def construct_all(self):
        '''
        Reconstruct raw data for comparison with target (Recon) data.  Follows procedures outlined in "runReconstruction.py"

        :return: None
        '''
        img_io, img_io_bg = create_metadata_object(self.config, self.RawDataPath,
                                                   self.ImgDir, self.SmDir, self.BgDir)
        img_io, img_io_bg = parse_bg_options(img_io, img_io_bg, self.config,
                                             self.RawDataPath, self.ProcessedPath,
                                             self.ImgDir, self.SmDir, self.BgDir)

        self.img_io, img_reconstructor = process_background(img_io, img_io_bg, self.config)
        self.img_io.posIdx = 0
        self.img_io.tIdx = 0
        self.img_io.zIdx = 0
        ImgRawSm, ImgProcSm, ImgFluor, ImgBF = parse_tiff_input(self.img_io)
        img_stokes_sm = img_reconstructor.compute_stokes(ImgRawSm)
        img_stokes_sm = img_reconstructor.correct_background(img_stokes_sm)
        img_computed_sm = img_reconstructor.reconstruct_birefringence(img_stokes_sm)
        [I_trans, retard, azimuth, polarization, _, _, _] = img_computed_sm

        self.scattering = 1 - polarization
        self.azimuth_degree = azimuth / np.pi * 180

        self.I_trans = imBitConvert(I_trans * 10 ** 3, bit=16, norm=False)  # AU, set norm to False for tiling images
        self.retard = imBitConvert(retard * 10 ** 3, bit=16)  # scale to pm
        self.scattering = imBitConvert(self.scattering * 10 ** 4, bit=16)
        self.azimuth_degree = imBitConvert(self.azimuth_degree * 100, bit=16)  # scale to [0, 18000], 100*degree



    def test_mse_Itrans(self):
        self.construct_all()
        self.assertLessEqual(mse(self.I_trans, cv2.imread(self.target_ITrans, -1)), 50000)

    def test_mse_retard(self):
        self.construct_all()
        self.assertLessEqual(mse(self.retard, cv2.imread(self.target_retard, -1)), 100)

    def test_mse_orientation(self):
        self.construct_all()
        self.assertLessEqual(mse(self.azimuth_degree, cv2.imread(self.target_Orientation, -1)), 65000)

    def test_mse_scattering(self):
        self.construct_all()
        self.assertLessEqual(mse(self.scattering, cv2.imread(self.target_Scattering, -1)), 100)


if __name__ == '__main__':
    unittest.main()
