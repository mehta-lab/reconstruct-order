"""
Class to read mManager format images saved separately and their metadata (JSON) .
"""
import json, os
import numpy as np
import pandas as pd
import cv2
import warnings

from ..utils.imgIO import get_sub_dirs, get_sorted_names
 



class mManagerReader(object):
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
    img_sm_path : str
        path of the acquisition folder
    img_in_pos_path : str
        path of the current position folder
    img_output_path : str
        full path of the output folder
    width : int
        width of the input image
    height : int
        height of the input image
    channels : list
        channels in the meta file
    input_chans : list
        channels to read
    n_input_chans : int
        number of channels to read
    output_chans : list
        output channels
    n_output_chans : int
        number of output channels
    n_pos : int
        number of positions in the meta file
    n_time : int
        number of time points in the meta file
    n_z :
        number of time slices in the meta file
    size_x_um : float
        pixel size in x
    size_y_um : float
        pixel size in y
    size_z_um : float
        z step
    time_stamp : list
        time points in the meta file
    pos_idx : int
        current postion index to process
    t_idx : int
        current time index to process
    z_idx : int
        current z index to process
    bg : str
        background folder name
    bg_method : str
        "Global" or "Local". Type of background correction. "Global" will correct each image
         using the same background. "Local" will do correction with locally estimated
         background in addition to global background
    bg_correct : bool
        Perform background correct (True) or not (False)
    binning : int
        binning (or pooling) size for the images

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

    def _mm1_meta_parser(self):
        input_meta_file = self.input_meta_file
        self._meta_pos_list = ['Pos0']
        pos_dict_list = self.input_meta_file['Summary']['InitialPositionList']
        if pos_dict_list:
            self._meta_pos_list = [pos_dict['Label'] for pos_dict in pos_dict_list]
        self._pos_list = self._meta_pos_list
        self.width = input_meta_file['Summary']['Width']
        self.height = input_meta_file['Summary']['Height']
        self.time_stamp = input_meta_file['Summary']['Time']

    def _mm2_meta_parser(self):
        input_meta_file = self.input_meta_file
        self._meta_pos_list = ['']
        if 'StagePositions' in self.input_meta_file['Summary']:
            pos_dict_list = self.input_meta_file['Summary']['StagePositions']
            self._meta_pos_list = [pos_dict['Label'] for pos_dict in pos_dict_list]
        self._pos_list = self._meta_pos_list
        self.width = int(input_meta_file['Summary']['UserData']['Width']['PropVal'])
        self.height = int(input_meta_file['Summary']['UserData']['Height']['PropVal'])
        self.time_stamp = input_meta_file['Summary']['StartTime']

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

    @property
    def t_list(self):
        return self._t_list

    @t_list.setter
    def t_list(self, value):
        """time list to process

        Parameters
        ----------
        value: list
        time list to process

        """
        assert set(value).issubset(self._meta_t_list), \
            'some positions cannot be found in metadata'
        self._t_list = value

    @property
    def z_list(self):
        return self._z_list

    @z_list.setter
    def z_list(self, value):
        """z list to process

        Parameters
        ----------
        value: list
        z list to process

        """
        assert set(value).issubset(self._meta_z_list), \
            'some positions cannot be found in metadata'
        self._z_list = value

    def _detect_img_name_format(self):
        img_name = self.img_names[0]
        if 'img_000000' in img_name:
            self.img_name_format = 'mm_1_4_22'
        elif 'position' in img_name:
            self.img_name_format = 'mm_2_0'
        elif 'img_' in img_name:
            self.img_name_format = 'recon_order'
        else:
            raise ValueError('Unknown image name format')

    def get_chan_name(self):
        return self.input_chans[self.chan_idx]

    def get_img_name(self):
        if self.img_name_format == 'mm_1_4_22':
            img_name = 'img_000000{:03d}_{}_{:03d}.tif'.\
                format(self.t_idx, self.get_chan_name(), self.z_idx)
        elif self.img_name_format == 'mm_2_0':
            chan_meta_idx = self.channels.index(self.get_chan_name())
            img_name = 'img_channel{:03d}_position{:03d}_time{:09d}_z{:03d}.tif'.\
                format(chan_meta_idx, self.t_idx, self.pos_idx, self.z_idx)
        elif self.img_name_format == 'recon_order':
            img_name = 'img_{}_t{:03d}_p{:03d}_z{:03d}.tif'.\
                format(self.get_chan_name(), self.t_idx, self.pos_idx, self.z_idx)
        else:
            raise ValueError('Undefined image name format')
        return img_name
    
    def read_img(self):
        """read a single image at (c,t,p,z)"""
        img_name = self.get_img_name()
        img_file = os.path.join(self.img_in_pos_path, img_name)
        img = cv2.imread(img_file, -1) # flag -1 to preserve the bit dept of the raw image
        if img is None:
            warnings.warn('image "{}" cannot be found. Return None instead.'.format(img_name))
        else:
            img = img.astype(np.float32, copy=False)  # convert to float32 without making a copy to save memory
        return img


    def read_multi_chan_img_stack(self, z_ids=None):
        """read multi-channel image stack at a given (t,p)"""
        if not os.path.exists(self.img_sm_path):
            raise FileNotFoundError(
                "image file doesn't exist at:", self.img_sm_path
            )
        if not z_ids:
            z_ids = list(range(0, self.nZ))
        img_chann = []  # list of 2D or 3D images from different channels
        for chan_idx in range(self.n_input_chans):
            img_stack = []
            self.chan_idx = chan_idx
            for z_idx in z_ids:
                self.z_idx = z_idx
                img = self.read_img()
                img_stack += [img]
            img_stack = np.stack(img_stack)  # follow zyx order
            img_chann += [img_stack]
        return img_chann
    
    def write_img(self, img):
        """only supports recon_order image name format currently"""
        if not os.path.exists(self.img_output_path): # create folder for processed images
            os.makedirs(self.img_output_path)
        img_name = 'img_'+self.output_chans[self.chan_idx]+'_t%03d_p%03d_z%03d.tif'%(self.t_idx, self.pos_idx, self.z_idx)
        if len(img.shape)<3:
            cv2.imwrite(os.path.join(self.img_output_path, img_name), img)
        else:
            cv2.imwrite(os.path.join(self.img_output_path, img_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
    def writeMetaData(self):
        if not os.path.exists(self.img_output_path): # create folder for processed images
            os.makedirs(self.img_output_path)
        self.input_meta_file['Summary']['ChNames'] = self.input_chans
        self.input_meta_file['Summary']['Channels'] = self.n_input_chans
        metaFileName = os.path.join(self.img_output_path, 'metadata.txt')
        with open(metaFileName, 'w') as f:  
            json.dump(self.input_meta_file, f)
        df_pos_path = os.path.join(self.img_output_path, 'pos_table.csv')
        df_pos = pd.DataFrame(list(enumerate(self.pos_list)),
                          columns=['pos idx', 'pos dir'])
        df_pos.to_csv(df_pos_path, sep=',')

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
    def __init__(self, img_sample_path, img_output_path=None, input_chans=[], output_chans=[], binning=1):

        mManagerReader.__init__(self, img_sample_path, img_output_path, input_chans, output_chans, binning)
        metaFile = self.input_meta_file
        self.acquScheme = metaFile['Summary']['~ Acquired Using']
        self.bg = metaFile['Summary']['~ Background']
        self.blackLevel = metaFile['Summary']['~ BlackLevel']
        self.mirror = metaFile['Summary']['~ Mirror']
        self.swing = metaFile['Summary']['~ Swing (fraction)']
        self.wavelength = metaFile['Summary']['~ Wavelength (nm)']

