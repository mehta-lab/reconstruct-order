"""
Class to read mManager format images saved separately and their metadata (JSON) .
"""
import json, os
import numpy as np
import pandas as pd
import cv2
from ..utils.imgIO import GetSubDirName


class mManagerReader:
    """General mManager Reader"""

    def __init__(self, ImgSmPath, ImgOutPath=None, input_chan=[], output_chan=[]):
        """
        :param str ImgSmPath: full path of the acquisition folder
        (1 level up of pos folder)
        :param str ImgOutPath: full path of the output folder
        :param list input_chan: list of input channel names
        :param list output_chan: list of output channel names
        """
        subDirName = GetSubDirName(ImgSmPath)

        img_in_pos_path = ImgSmPath # data structure doesn't have position folders
        if subDirName:
            subDir = subDirName[0] # pos0
            if 'Pos' in subDir: # mManager format            
                img_in_pos_path = os.path.join(ImgSmPath, subDir)
        metaFileName = os.path.join(img_in_pos_path, 'metadata.txt')
        with open(metaFileName, 'r') as f:
            input_meta_file = json.load(f)
        self.input_meta_file = input_meta_file
        self._meta_pos_list = ['Pos0']
        self._pos_list = self.meta_pos_list
        self.name = input_meta_file["Summary"]["Prefix"]
        self.output_meta_file = []
        self.ImgSmPath = ImgSmPath
        self.img_in_pos_path = img_in_pos_path # input pos path
        self.ImgOutPath = ImgOutPath
        self.width = input_meta_file['Summary']['Width']
        self.height = input_meta_file['Summary']['Height']
        self.chNames = input_meta_file['Summary']['ChNames'] # channels in metadata
        self.chNamesIn = input_chan  # channels to read
        self.nChannIn = len(input_chan)
        self.chNamesOut = output_chan #output channel names
        self.nChannOut = len(output_chan)
        self.imgLimits = [[np.Inf,0]]*self.nChannOut
        self.nPos = input_meta_file['Summary']['Positions']
        self.nTime = input_meta_file['Summary']['Frames']
        self.nZ = input_meta_file['Summary']['Slices']
        self.size_x_um = 6.5/63 # (um) for zyla at 63X. mManager metafile currently does not log the correct pixel size
        self.size_y_um = 6.5/63 # (um) for zyla at 63X. Manager metafile currently does not log the correct pixel size
        self.size_z_um = input_meta_file['Summary']['z-step_um']
        self.time_stamp = input_meta_file['Summary']['Time']
        self.img_fluor_bg = np.ones((4, self.height, self.width))
        self.posIdx = 0  # assuming only single image for background
        self.tIdx = 0
        self.zIdx = 0
        self.bg = None

    @property
    def meta_pos_list(self):
        pos_dict_list = self.input_meta_file['Summary']['InitialPositionList']
        self._meta_pos_list = [pos_dict['Label'] for pos_dict in pos_dict_list]
        return self._meta_pos_list
    @property
    def pos_list(self):
        return self._pos_list

    @pos_list.setter
    def pos_list(self, value):
        assert set(value).issubset(self._meta_pos_list), \
            'some positions cannot be found in metadata'
        self._pos_list = value

    def read_img(self):
        """read a single image at (c,t,p,z)"""
        fileName = 'img_'+self.chNamesIn[self.chanIdx]+'_t%03d_p%03d_z%03d.tif'%(self.tIdx, self.posIdx, self.zIdx)
        TiffFile = os.path.join(self.ImgSmPath, fileName)
        img = cv2.imread(TiffFile,-1) # flag -1 to preserve the bit dept of the raw image
        return img

    def read_multi_chan_img_stack(self, z_range=None):
        """read multi-channel image stack at a given (t,p)"""
        if not os.path.exists(self.ImgSmPath):
            raise FileNotFoundError(
                "image file doesn't exist at:", self.ImgSmPath
            )
        if not z_range:
            z_range = [0, self.nZ]
        img_chann = []  # list of 2D or 3D images from different channels
        for chanIdx in range(self.nChannIn):
            img_stack = []
            self.chanIdx = chanIdx
            for zIdx in range(z_range[0], z_range[1]):
                self.zIdx = zIdx
                img = self.read_img()
                img_stack += [img]
            img_stack = np.stack(img_stack)  # follow zyx order
            img_stack = np.squeeze(img_stack)
            img_chann += [img_stack]
        return img_chann
    
    def write_img(self, img):
        if not os.path.exists(self.ImgOutPath): # create folder for processed images
            os.makedirs(self.ImgOutPath)
        fileName = 'img_'+self.chNamesOut[self.chanIdx]+'_t%03d_p%03d_z%03d.tif'%(self.tIdx, self.posIdx, self.zIdx)
        if len(img.shape)<3:
            cv2.imwrite(os.path.join(self.ImgOutPath, fileName), img)
        else:
            cv2.imwrite(os.path.join(self.ImgOutPath, fileName), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
    def writeMetaData(self):
        if not os.path.exists(self.ImgOutPath): # create folder for processed images
            os.makedirs(self.ImgOutPath)
        self.input_meta_file['Summary']['ChNames'] = self.chNamesIn
        self.input_meta_file['Summary']['Channels'] = self.nChannIn
        metaFileName = os.path.join(self.ImgOutPath, 'metadata.txt')
        with open(metaFileName, 'w') as f:  
            json.dump(self.input_meta_file, f)
        df_pos_path = os.path.join(self.ImgOutPath, 'pos_table.csv')
        df_pos = pd.DataFrame(list(enumerate(self.pos_list)),
                          columns=['pos idx', 'pos dir'])
        df_pos.to_csv(df_pos_path, sep=',')
        
    def save_microDL_format_old(self):
        """
        Save the images in old microDL (https://github.com/czbiohub/microDL) input format
        microDL input structure:
        ImgOutPath
         |-split_images, split_images_info.csv
            |-tp0
                |-channel0
         |-img_512_512_8_.., cropped_images_info.csv
            |-tp-0
                |-channel0: contains all npy files for cropped images from channel0
                |-channel1: contains all npy files for cropped images from channel1..
                and so on
        
        Saves the individual images as a npy file
        """

        if not os.path.exists(self.ImgSmPath):
            raise FileNotFoundError(
                "image file doesn't exist at:", self.ImgSmPath
            )
        os.makedirs(self.ImgOutPath, exist_ok=True)
        self.logger = self._init_logger()        
        records = []
        for tIdx in range(self.nTime):
            self.tIdx = tIdx
            timepoint_dir = os.path.join(self.ImgOutPath,
                                         'timepoint_{}'.format(tIdx))
            os.makedirs(timepoint_dir, exist_ok=True)

            for chanIdx in range(self.nChannOut):
                self.chanIdx = chanIdx
                self.channel_dir = os.path.join(timepoint_dir,
                                           'channel_{}'.format(chanIdx))
                os.makedirs(self.channel_dir, exist_ok=True)

                # for posIdx in range(0, 37):  # nXY
                for posIdx in range(self.nPos):  # nXY

                    self.posIdx = posIdx
                    cur_records = self.save_npy_2D()
                    records.extend(cur_records)
                msg = 'Wrote files for tp:{}, channel:{}'.format(
                    tIdx, chanIdx
                )
                self._log_info(msg)
        df = pd.DataFrame.from_records(
            records,
            columns=['timepoint', 'channel_num', 'sample_num', 'slice_num',
                     'fname', 'size_x_microns', 'size_y_microns',
                     'size_z_microns','mean', 'std']
        )
        metadata_fname = os.path.join(self.ImgOutPath,
                                      'split_images_info.csv')
        df.to_csv(metadata_fname, sep=',')

    def save_microDL_format_new(self):
        """
        Save the images in new microDL (https://github.com/czbiohub/microDL) input format
        microDL input structure:
        dir_name
            |
            |- frames_meta.csv
            |- global_metadata.csv
            |- im_c***_z***_t***_p***.png
            |- im_c***_z***_t***_p***.png
            |- ...

        Saves the individual images as a png file
        """

        if not os.path.exists(self.ImgSmPath):
            raise FileNotFoundError(
                "image file doesn't exist at:", self.ImgSmPath
            )
        os.makedirs(self.ImgOutPath, exist_ok=True)
        self.logger = self._init_logger()
        records = []
        for tIdx in range(self.nTime):
            self.tIdx = tIdx
            for chanIdx in range(self.nChannOut):
                self.chanIdx = chanIdx
                # for posIdx in range(0, 37):  # nXY
                for posIdx in range(self.nPos):  # nXY
                    self.posIdx = posIdx
                    for zIdx in range(self.nZ):
                        self.zIdx = zIdx
                        cur_fname = os.path.join(
                            self.ImgOutPath, 'im_c{}_z{}_t{}_p{}.png'.format(chanIdx, zIdx, tIdx, posIdx)
                        )
                        # image voxels are 16 bits

                        img = self.read_img()
                        self.mean = np.nanmean(img)
                        self.std = np.nanstd(img)
                        cv2.imwrite(cur_fname, img)
                        msg = 'Generated file:{}'.format(cur_fname)
                        self._log_info(msg)
                msg = 'Wrote files for tp:{}, channel:{}'.format(
                    tIdx, chanIdx
                )
                self._log_info(msg)


class PolAcquReader(mManagerReader):
    """PolAcquistion Plugin output format reader"""
    def __init__(self, ImgSmPath, ImgOutPath=None, verbose=0, input_chan=[], output_chan=[]):
        """
        Extract PolAcquistion specific params from the metafile
        """
        mManagerReader.__init__(self, ImgSmPath, ImgOutPath, input_chan, output_chan)
        metaFile = self.input_meta_file
        self.acquScheme = metaFile['Summary']['~ Acquired Using']
        self.bg = metaFile['Summary']['~ Background']
        self.blackLevel = metaFile['Summary']['~ BlackLevel']
        self.mirror = metaFile['Summary']['~ Mirror']
        self.swing = metaFile['Summary']['~ Swing (fraction)']
        self.wavelength = metaFile['Summary']['~ Wavelength (nm)']

    @property
    def meta_pos_list(self):
        return self._meta_pos_list
        # PolAcquisition doens't save position list



    
