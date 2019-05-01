"""
Class to read mManager format images saved separately and their metadata (JSON) .
"""
import json, os
import numpy as np
import pandas as pd
import cv2
from ..utils.imgIO import GetSubDirName


class mManagerReader:
    """General mManager metadata and image reader for data saved as separate 2D tiff files

    Parameters
    ----------
    img_sample_path : str
        full path of the acquisition folder (parent folder of pos folder)
    ImgOutPath : str
        full path of the output folder
    input_chan : list
        list of input channel names
    output_chan : list
        list of output channel names

    Attributes
    ----------
    input_meta_file : dict
        input mManager meta file of the acquistion
    _meta_pos_list : list
        position list in the meta file
    _pos_list : list
        position list to process
    name : str
        acquisition folder name
    output_meta_file : dict
        output meta file
    ImgSmPath : str
        path of the acquisition folder
    img_in_pos_path : str
        path of the current position folder
    ImgOutPath : str
        full path of the output folder
    width : int
        width of the input image
    height : int
        height of the input image
    chNames : list
        channels in the meta file
    chNamesIn : list
        channels to read
    nChannIn : int
        number of channels to read
    chNamesOut : list
        output channels
    nChannOut : int
        number of output channels
    imgLimits : list
        min and max intensity values of all the images
    nPos : int
        number of positions in the meta file
    nTime : int
        number of time points in the meta file
    nZ :
        number of time slices in the meta file
    size_x_um : float
        pixel size in x
    size_y_um : float
        pixel size in y
    size_z_um : float
        z step
    time_stamp : list
        time points in the meta file
    img_fluor_bg : array
        Fluorescence channel background for flat-field correction
    posIdx : int
        current postion index to process
    tIdx : int
        current time index to process
    zIdx : int
        current z index to process
    bg : str
        background folder name
    bg_method : str
        "Global" or "Local". Type of background correction. "Global" will correct each image
         using the same background. "Local" will do correction with locally estimated
         background in addition to global background
    bg_correct : bool
        Perform background correct (True) or not (False)
    ff_method : str
        flat-field correction method. Using morphological opening if 'open' and empty image if 'empty'
    ImgFluorMin : array
        array of empty flourescence images for each channel
    ImgFluorSum : array
        array of sum of morphologically opened flourescence images for each channel
    kernel : obj
      kernel for image opening operation
    loopZ : str
        flag determine which loopZ function to call for processing.
        'reconstruct' calls LoopZSm and 'flat-field' calls loopZBg

    """

    def __init__(self, img_sample_path, ImgOutPath=None, input_chan=[], output_chan=[]):

        img_in_pos_path = img_sample_path
        subDirName = GetSubDirName(img_sample_path)
        if subDirName:
            subDir = subDirName[0] # pos0
            if 'Pos' in subDir: # mManager format            
                img_in_pos_path = os.path.join(img_sample_path, subDir)
        metaFileName = os.path.join(img_in_pos_path, 'metadata.txt')
        with open(metaFileName, 'r') as f:
            input_meta_file = json.load(f)
        self.input_meta_file = input_meta_file
        self._meta_pos_list = ['Pos0']
        self._pos_list = self.meta_pos_list
        self.name = input_meta_file["Summary"]["Prefix"]
        self.output_meta_file = []
        self.ImgSmPath = img_sample_path
        self.img_in_pos_path = img_in_pos_path
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
        self.bg = 'No Background'
        self.bg_method = 'Global'
        self.bg_correct = True
        self.ff_method = 'open'
        self.ImgFluorMin = np.full((4, self.height, self.width), np.inf)  # set initial min array to to be Inf
        self.ImgFluorSum = np.zeros(
            (4, self.height, self.width))  # set the default background to be Ones (uniform field)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (100,
                                                   100))  # kernel for image opening operation, 100-200 is usually good
        self.loopZ = 'reconstruct'

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
        """position list to process

        Parameters
        ----------
        value: list
        position list to process

        """
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


class PolAcquReader(mManagerReader):
    """PolAcquistion mManager metadata and image reader
    Parameters
    ----------
    mManagerReader : class
        General mManager metadata and image reader for data saved as separate 2D tiff files

    Attributes
    ----------
    acquScheme : str
        Pol images acquiring schemes. '4-Frame' or '5-Frame'
    bg : str
        background folder name in metadata
    blackLevel : int
        black level of the camera
    mirror : str
        'Yes' or 'No'. Changing this flag will flip the slow axis horizontally
    swing : float
        swing of the elliptical polarization states in unit of fraction of wavelength
    wavelength : int
        wavelenhth of the illumination light (nm)


    """
    def __init__(self, img_sample_path, ImgOutPath=None, input_chan=[], output_chan=[]):

        mManagerReader.__init__(self, img_sample_path, ImgOutPath, input_chan, output_chan)
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
