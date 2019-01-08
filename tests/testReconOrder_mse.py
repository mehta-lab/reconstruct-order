#!/usr/bin/env python
# title           : this_python_file.py
# description     :This will create a header for a python script.
# author          :bryant.chhun
# date            :12/12/18
# version         :0.0
# usage           :python this_python_file.py -flags
# notes           :
# python_version  :3.6

import unittest

import cv2

from birefringence_python.PolScope import multiDimProcess, reconstruct
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

    def construct_all(self):
        # create file loaders
        datapipe = PipeToReconOrder(type="Test", sample_type="Sample1")
        datapipe_bg = PipeToReconOrder(type="Test", sample_type='BG')

        # initialize processors
        self.processor = ReconOrder()
        self.processor_bg = ReconOrder()
        self.processor.frames = 5
        self.processor_bg.frames = 5

        datapipe.set_processor(self.processor)
        datapipe_bg.set_processor(self.processor_bg)

        datapipe.compute_inst_matrix()
        datapipe_bg.compute_inst_matrix()

        # BGprocess first
        datapipe_bg.run_reconstruction()
        datapipe.run_reconstruction_BG_correction(datapipe_bg.get_processor())

    def construct_BG_only(self):
        datapipe_bg = PipeToReconOrder(type="Test", sample_type='BG')
        self.processor_bg = ReconOrder()
        self.processor_bg.frames = 5
        self.processor_bg.compute_inst_matrix()
        datapipe_bg.set_processor(self.processor_bg)
        # datapipe_bg.compute_inst_matrix()
        datapipe_bg.run_reconstruction()
        return datapipe_bg

    def test_mse_Itrans(self):
        self.construct_all()
        self.assertLessEqual(mse(self.processor.I_trans, cv2.imread(self.target_ITrans, -1)), 100000)

    def test_mse_retard(self):
        self.construct_all()
        self.assertLessEqual(mse(self.processor.retard, cv2.imread(self.target_retard, -1)), 100000)

    def test_mse_orientation(self):
        self.construct_all()
        self.assertLessEqual(mse(self.processor.azimuth_degree, cv2.imread(self.target_Orientation, -1)), 100000)

    def test_mse_scattering(self):
        self.construct_all()
        self.assertLessEqual(mse(self.processor.scattering, cv2.imread(self.target_Scattering, -1)), 100000)

    def test_mse_ReuseBackground(self):
        bg_pipe = self.construct_BG_only()
        datapipe1 = PipeToReconOrder(type="Test", sample_type="Sample1")
        datapipe2 = PipeToReconOrder(type="Test", sample_type="Sample2")
        processor1 = ReconOrder()
        processor2 = ReconOrder()
        processor1.frames = 5
        processor2.frames = 5

        processor1.compute_inst_matrix()

        datapipe1.set_processor(processor1)
        datapipe2.set_processor(processor2)
        datapipe1.compute_inst_matrix()
        datapipe2.compute_inst_matrix()

        datapipe1.run_reconstruction_BG_correction(bg_pipe.get_processor())
        datapipe2.run_reconstruction_BG_correction(bg_pipe.get_processor())


if __name__ == '__main__':
    unittest.main()
