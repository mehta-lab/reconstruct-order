#!/usr/bin/env python
# title           : testReconOrder_mse.py
# description     :This will create a header for a python script.
# author          :bryant.chhun
# date            :12/12/18
# version         :0.0
# usage           :python this_python_file.py -flags
# notes           :
# python_version  :3.6

import unittest
import yaml
import cv2

from compute.multiDimProcess import findBackground
from compute.reconstruct import ImgReconstructor
from utils.imgIO import parse_tiff_input
from tests.testMetrics import mse


'''
Methods to check whether all IMAGE data in both raw and npy format are trainable
'''

class TestImageReconstruction(unittest.TestCase):

    targetData = "./testData/reconData/2018_10_02_MouseBrainSlice/"
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

    source_config_file = './TestData/CONFIG_SM_2018_1002_1633_1.yml'
    sourceData = "./testData/rawData/2018_10_02_MouseBrainSlice/"
    sourceSample = "SM_2018_1002_1633_1/"
    sourceBackground = "BG_2018_1002_1625_1/"

    RawDataPath = sourceData+sourceSample

    def __init__(self, *args, **kwargs):
        super(TestImageReconstruction, self).__init__(*args, **kwargs)

        with open('./TestData/CONFIG_SM_2018_1002_1633_1.yml', 'r') as f:
            config = yaml.load(f)
        self.RawDataPath = config['dataset']['RawDataPath']
        self.ProcessedPath = config['dataset']['ProcessedPath']
        self.ImgDir = config['dataset']['ImgDir']
        self.SmDir = config['dataset']['SmDir']
        self.BgDir = config['dataset']['BgDir']
        self.BgDir_local = config['dataset']['BgDir_local']
        self.outputChann = config['processing']['outputChann']
        self.circularity = config['processing']['circularity']
        self.bgCorrect = config['processing']['bgCorrect']
        self.flatField = config['processing']['flatField']
        self.batchProc = config['processing']['batchProc']
        self.norm = config['plotting']['norm']

    def construct_all(self):
        self.img_io, img_reconstructor = findBackground(self.RawDataPath, self.ProcessedPath, self.ImgDir, self.SmDir, self.BgDir, self.outputChann,
                           BgDir_local=self.BgDir_local, flatField=self.flatField,bgCorrect=self.bgCorrect,
                           ff_method='open')
        self.img_io.posIdx = 0
        self.img_io.tIdx = 0
        self.img_io.zIdx = 0
        ImgRawSm, ImgProcSm, ImgFluor, ImgBF = parse_tiff_input(self.img_io)
        img_stokes_sm = img_reconstructor.compute_stokes(ImgRawSm)
        img_computed_sm = img_reconstructor.reconstruct_birefringence(img_stokes_sm, self.img_io.param_bg,
                                                                      circularity=self.circularity,
                                                                      bg_method=self.img_io.bg_method,
                                                                      extra=False)  # background subtraction
        [self.I_trans, self.retard, self.azimuth, self.polarization] = img_computed_sm


    def test_mse_Itrans(self):
        self.construct_all()
        self.assertLessEqual(mse(self.I_trans, cv2.imread(self.target_ITrans, -1)), 100000)

    def test_mse_retard(self):
        self.construct_all()
        self.assertLessEqual(mse(self.retard, cv2.imread(self.target_retard, -1)), 100000)

    def test_mse_orientation(self):
        self.construct_all()
        self.assertLessEqual(mse(self.azimuth, cv2.imread(self.target_Orientation, -1)), 100000)

    def test_mse_scattering(self):
        self.construct_all()
        self.assertLessEqual(mse(1-self.polarization, cv2.imread(self.target_Scattering, -1)), 100000)


if __name__ == '__main__':
    unittest.main()
