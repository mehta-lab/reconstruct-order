"""
Class to read and write Tiff and the metafile (JSON) in mManager format.  
"""


from abc import ABCMeta, abstractmethod
import json
import numpy as np
import os
import pandas as pd
import cv2
from utils.aux_utils import init_logger
from utils.imgIO import GetSubDirName

class mManagerReader(metaclass=ABCMeta):
    """General mManager Reader"""

    def __init__(self, ImgSmPath, ImgOutPath, verbose=0):
        """Init.

        :param str ImgPosPath: fname with full path of the Lif file
        :param str ImgOutPath: base folder for storing the individual
         image and cropped volumes
        :param int verbose: specifies the logging level: NOTSET:0, DEBUG:10,
         INFO:20, WARNING:30, ERROR:40, CRITICAL:50
        """
        subDirName = GetSubDirName(ImgSmPath)          
        imgLimits = [[np.Inf,0]]*5
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
        
        log_levels = [0, 10, 20, 30, 40, 50]
        if verbose in log_levels:
            self.verbose = verbose
        else:
            self.verbose = 10            
        self.width = metaFile['Summary']['Width']
        self.height = metaFile['Summary']['Height']
        self.chNames = metaFile['Summary']['ChNames']
        self.nChann = metaFile['Summary']['Channels']
        self.nPos = metaFile['Summary']['Positions']
        self.nTime = metaFile['Summary']['Frames']
        self.nZ = metaFile['Summary']['Slices']
        self.size_x_um = 6.5/63 # (um) for zyla at 63X
        self.size_y_um = 6.5/63 # (um) for zyla at 63X
        self.size_z_um = metaFile['Summary']['z-step_um']
#        if not os.path.exists(self.ImgOutPath): # create folder for processed images
#            os.makedirs(self.ImgOutPath)
        

    def _init_logger(self):
        """Initialize logger for pre-processing.

        Logger outputs to console and log_file
        """

        logger_fname = os.path.join(self.ImgOutPath, 'mManager_splitter.log')
        logger = init_logger('lif_splitter', logger_fname, self.verbose)
        return logger

#    @abstractmethod
#    def save_each_image(self, reader, nZ, channel_dir, tIdx,
#                        chanIdx, posIdx, size_x_um, size_y_um,
#                        size_z_um):
#        """Save each image as a numpy array."""
#
#        raise NotImplementedError

    def _log_info(self, msg):
        """Log info.

        :param str msg: message to be logged
        """

        if self.verbose > 0:
            self.logger.info(msg)
            
    def readmManager(self):
#        fileName = 'img_000000%03d'+self.chNames[chanIdx]+'_%03d.tif'%(timeIdx,zIdx)
        
        fileName = 'img_'+self.chNames[self.chanIdx]+'_t%03d_p%03d_z%03d.tif'%(self.tIdx, self.posIdx, self.zIdx)
        TiffFile = os.path.join(self.ImgSmPath, fileName)
        img = cv2.imread(TiffFile,-1) # flag -1 to perserve the bit dept of the raw image
#        img = img.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
#        img = img.reshape(img.shape[0], img.shape[1],1)        
        return img
    
    def writeImgPyPol(self, img, chanIdx, posIdx, zIdx, timeIdx):            
        if not os.path.exists(self.ImgOutPath): # create folder for processed images
            os.makedirs(self.ImgOutPath)
        fileName = 'img_'+self.chNames[chanIdx]+'_t%03d_p%03d_z%03d.tif'%(timeIdx,posIdx,zIdx)
        if len(img.shape)<3:
            cv2.imwrite(os.path.join(self.ImgOutPath, fileName), img)
        else:
            cv2.imwrite(os.path.join(self.ImgOutPath, fileName), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
    def writeMetaData(self):
        if not os.path.exists(self.ImgOutPath): # create folder for processed images
            os.makedirs(self.ImgOutPath)
        self.metaFile['Summary']['ChNames'] = self.chNames
        self.metaFile['Summary']['Channels'] = self.nChann 
        metaFileName = os.path.join(self.ImgOutPath, 'metadata.txt')
        with open(metaFileName, 'w') as f:  
            json.dump(self.metaFile, f)        
        
    def save_images(self):
        """
        Save the images in microDL (https://github.com/czbiohub/microDL) input format 
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

        2D might have more acquisitions +/- focal plane, (usually 3 images).
        focal_plane_idx corresponds to the plane to consider. Mid-plane is the
        one in focus and the +/- on either side would be blurred. For 2D
        acquisitions, this is stored along the Z dimension. How is this handled
        for 3D acquisitions?
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

            for chanIdx in range(self.nChann):
                self.chanIdx = chanIdx
                self.channel_dir = os.path.join(timepoint_dir,
                                           'channel_{}'.format(chanIdx))
                os.makedirs(self.channel_dir, exist_ok=True)
                for posIdx in range(0, 37):  # nXY
                    self.posIdx = posIdx
                    cur_records = self.save_each_image2D()                                        
                    records.extend(cur_records)
                msg = 'Wrote files for tp:{}, channel:{}'.format(
                    tIdx, chanIdx
                )
                self._log_info(msg)
        df = pd.DataFrame.from_records(
            records,
            columns=['timepoint', 'channel_num', 'sample_num', 'slice_num',
                     'fname', 'size_x_microns', 'size_y_microns',
                     'size_z_microns']
        )
        metadata_fname = os.path.join(self.ImgOutPath,
                                      'split_images_info.csv')
        df.to_csv(metadata_fname, sep=',')



    """Saves the individual images as a npy file

    In some acquisitions there are 3 z images corresponding to different focal
    planes (focal plane might not be the correct term here!). Using z=0 for the
    recent experiment
    """

    def save_each_image2D(self):
        """Saves the each individual image as a npy file.

        Have to decide when to reprocess the file and when not to. Currently
        doesn't check if the file has already been processed.
        :param bf.ImageReader reader: fname with full path of the lif image
        :param int nZ: number of focal_plane acquisitions
        :param str channel_dir: dir to save the split images
        :param int tIdx: timepoint to split
        :param int chann: channel to split
        :param int posIdx: sample to split
        :param float size_x_um: voxel resolution along x in microns
        :param float size_y_um: voxel resolution along y in microns
        :param float size_z_um: voxel resolution along focal_plane in microns
        :return: list of tuples of metadata
        """
        records = []
        # exclude the first 14 due to artifacts and some have one z
        # (series 15, 412) instead of 3
        for zIdx in range(self.nZ):
            self.zIdx = zIdx
            cur_fname = os.path.join(
                self.channel_dir, 'image_n{}_z{}.npy'.format(self.posIdx, zIdx)
            )
            # image voxels are 16 bits
            
            img = self.readmManager()
            np.save(cur_fname, img, allow_pickle=True, fix_imports=True)
            msg = 'Generated file:{}'.format(cur_fname)
            self._log_info(msg)
            # add wavelength info perhaps?
            records.append((self.tIdx, self.chanIdx, self.posIdx, zIdx,
                            cur_fname, self.size_x_um, self.size_y_um, self.size_z_um))
        return records



    """Class for splitting and cropping lif images."""

    def save_each_image3D(self, reader, nZ, channel_dir, tIdx,
                        chanIdx, posIdx, size_x_um, size_y_um,
                        size_z_um):
        """Saves the individual image volumes as a npy file.

        :param bf.ImageReader reader: fname with full path of the lif image
        :param int nZ: number of focal_plane acquisitions
        :param str channel_dir: dir to save the split images
        :param int tIdx: timepoint to split
        :param int chann: channel to split
        :param int posIdx: sample to split
        :param float size_x_um: voxel resolution along x in microns
        :param float size_y_um: voxel resolution along y in microns
        :param float size_z_um: voxel resolution along z in microns
        :return: list of tuples of metadata
        """

        records = []
        cur_vol_fname = os.path.join(channel_dir,
                                     'image_n{}.npy'.format(posIdx))
        for zIdx in range(nZ):
            img[:, :, zIdx] = reader.read(c=chanIdx, z=zIdx,
                                           t=tIdx, series=posIdx,
                                           rescale=False)
        np.save(cur_vol_fname, img, allow_pickle=True, fix_imports=True)
        # add wavelength info perhaps?
        records.append((chanIdx, posIdx, tIdx, cur_vol_fname,
                        size_x_um, size_y_um, size_z_um))
        msg = 'Wrote files for channel:{}'.format(chann)
        self._log_info(msg)
        return records

class PolAcquReader(mManagerReader):
    """PolAcquistion Plugin output reader""" 
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
        
    