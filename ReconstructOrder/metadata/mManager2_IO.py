import numpy as np
import pandas as pd
import cv2
import warnings
import os
from aicsimageio import AICSImage
from ..utils.imgIO import get_sub_dirs, get_sorted_names


def create_stack_object(img_sample_path):

    img_names = get_sorted_names(img_sample_path)
    img_paths = [os.path.join(img_sample_path,img) for img in img_names]

    stack_obj = [AICSImage(path) for path in img_paths]

    return stack_obj

class mManager2Reader(object):
    """General mManager metadata and image reader for data saved as separate 2D tiff files

    Parameters
    ----------
    img_sample_path : str
        full path of the acquisition folder (parent folder of pos folder)
    img_output_path : str
        full path of the output folder
    input_chan : list
        list of input channel names
    output_chan : list
        list of output channel names
    """

    def __init__(self, img_sample_path, img_output_path=None, input_chans=[], output_chans=[], binning=1):

        pos_path = img_sample_path # mManager 2.0 single position format
        sub_dirs = get_sub_dirs(img_sample_path)
        if sub_dirs:
            sub_dir = sub_dirs[0] # assume all the folders in the sample folder are position folders
            pos_path = os.path.join(img_sample_path, sub_dir)
            ##TODO: check the behavior of 2.0 gamma
        metadata_path = os.path.join(pos_path, 'metadata.txt')
        with open(metadata_path, 'r') as f:
            input_meta_file = json.load(f)

        self.input_meta_file = input_meta_file
        self.mm_version = input_meta_file['Summary']['MicroManagerVersion']
        if self.mm_version == '1.4.22':
            self.meta_parser = self._mm1_meta_parser
        elif '2.0' in self.mm_version:
            self.meta_parser = self._mm2_meta_parser
        else:
            raise ValueError(
                'Current MicroManager reader only supports version 1.4.22 and 2.0 but {} was detected'.
                    format(self.mm_version))

        self.img_sm_path = img_sample_path
        self.img_in_pos_path = pos_path
        self.img_names = get_sorted_names(pos_path)
        self.img_name_format = None
        self._detect_img_name_format()
        self.img_output_path = img_output_path
        self.input_chans = self.channels = input_meta_file['Summary']['ChNames']
        if input_chans:
            self.input_chans = input_chans
        self.n_input_chans = len(input_chans)
        self.output_chans = output_chans  # output channel names
        self.n_output_chans = len(output_chans)
        self.output_meta_file = []
        self.binning = binning
        self.name = input_meta_file["Summary"]["Prefix"]
        self.n_pos = input_meta_file['Summary']['Positions']
        self.n_time = input_meta_file['Summary']['Frames']
        self.n_z = input_meta_file['Summary']['Slices']
        self._t_list = self._meta_t_list = list(range(0, self.n_time))
        self._z_list = self._meta_z_list = list(range(0, self.n_z))
        self.size_z_um = input_meta_file['Summary']['z-step_um']
        self.pos_idx = 0  # assuming only single image for background
        self.t_idx = 0
        self.z_idx = 0
        self.chan_idx = 0
        self.bg = 'No Background'
        self.bg_method = 'Global'
        self.bg_correct = True
        self.meta_parser()

