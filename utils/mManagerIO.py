"""
Class to read and write Tiff and the metafile (JSON) in mManager format.
"""
import json
import numpy as np
import os
import pandas as pd
import cv2
from utils.imgIO import GetSubDirName

class mManagerReader:
    """General mManager Reader"""

    def __init__(self, ImgSmPath, ImgOutPath, inputChann=[], outputChann=[]):
        """
        :param str ImgSmPath: fname with full path of the Lif file
        :param str ImgOutPath: base folder for storing the individual
         image and cropped volumes
        """
        subDirName = GetSubDirName(ImgSmPath)          
        
        ## TO DO: track global image limits
        ImgPosPath = ImgSmPath #PyPol format
        if subDirName:
            subDir = subDirName[0] # pos0
            if 'Pos' in subDir: # mManager format            
                ImgPosPath = os.path.join(ImgSmPath, subDir)             
        metaFileName = os.path.join(ImgPosPath, 'metadata.txt')
        with open(metaFileName, 'r') as f:
            metaFile = json.load(f)
        self.metaFile = metaFile
        self.ImgSmPath = ImgSmPath
        self.ImgPosPath = ImgPosPath
        self.ImgOutPath = ImgOutPath
        self.width = metaFile['Summary']['Width']
        self.height = metaFile['Summary']['Height']
        self.chNames = metaFile['Summary']['ChNames'] # channels in metadata
        self.chNamesIn = inputChann  # channels to read
        self.nChannIn = len(inputChann)
        self.chNamesOut = outputChann #output channel names
        self.nChannOut = len(outputChann)
        self.imgLimits = [[np.Inf,0]]*self.nChannOut
        self.nPos = metaFile['Summary']['Positions']
        self.nTime = metaFile['Summary']['Frames']
        self.nZ = metaFile['Summary']['Slices']
        self.size_x_um = 6.5/63 # (um) for zyla at 63X. mManager metafile currently does not log the correct pixel size
        self.size_y_um = 6.5/63 # (um) for zyla at 63X. Manager metafile currently does not log the correct pixel size
        self.size_z_um = metaFile['Summary']['z-step_um']
        self.time_stamp = metaFile['Summary']['Time']
            
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
        for chanIdx in range(self.nChannOut):
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
    
    def write_img(self, img, channel):
        if not os.path.exists(self.ImgOutPath): # create folder for processed images
            os.makedirs(self.ImgOutPath)
        fileName = 'img_'+channel+'_t%03d_p%03d_z%03d.tif'%(self.tIdx, self.posIdx, self.zIdx)
        if len(img.shape)<3:
            cv2.imwrite(os.path.join(self.ImgOutPath, fileName), img)
        else:
            cv2.imwrite(os.path.join(self.ImgOutPath, fileName), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
    def writeMetaData(self):
        if not os.path.exists(self.ImgOutPath): # create folder for processed images
            os.makedirs(self.ImgOutPath)
        self.metaFile['Summary']['ChNames'] = self.chNamesIn
        self.metaFile['Summary']['Channels'] = self.nChannIn 
        metaFileName = os.path.join(self.ImgOutPath, 'metadata.txt')
        with open(metaFileName, 'w') as f:  
            json.dump(self.metaFile, f)        
        
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
    def __init__(self, ImgSmPath, ImgOutPath, verbose=0):
        """
        Extract PolAcquistion specific params from the metafile
        """
        mManagerReader.__init__(self, ImgSmPath, ImgOutPath)
        metaFile = self.metaFile
        self.acquScheme = metaFile['Summary']['~ Acquired Using']
        self.bg = metaFile['Summary']['~ Background']
        self.blackLevel = metaFile['Summary']['~ BlackLevel']
        self.mirror = metaFile['Summary']['~ Mirror']
        self.swing = metaFile['Summary']['~ Swing (fraction)']
        self.wavelength = metaFile['Summary']['~ Wavelength (nm)']
        
    
