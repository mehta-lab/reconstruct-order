"""
Read and write Tiff in mManager format. Will be replaced by mManagerIO.py 
"""
import os
import numpy as np
import glob
import re
import cv2
from shutil import copy2


def GetSubDirName(ImgPath):
    """Return sub-directory names in a directory

    Parameters
    ----------
    ImgPath: str
        Path to the input directory

    Returns
    -------
    subDirName: list
        list of sub-directory names
    """
    assert os.path.exists(ImgPath), 'Input folder does not exist!' 
    subDirPath = glob.glob(os.path.join(ImgPath, '*/'))    
    subDirName = [os.path.split(subdir[:-1])[1] for subdir in subDirPath]
#    assert subDirName, 'No sub directories found'
    return subDirName


def FindDirContainPos(ImgPath):
    """Recursively find the parent directory of "Pos#" directory
    Parameters
    ----------
    ImgPath: str
        Path to the input directory

    Returns
    -------
    ImgPath: str
        Path to the parent directory of "Pos#" directory

    """
    subDirName = GetSubDirName(ImgPath)
    assert subDirName, 'No "Pos" directories found. Check if the input folder contains "Pos"'
    subDir = subDirName[0]  # get pos0 if it exists
    ImgSubPath = os.path.join(ImgPath, subDir)
    if 'Pos' not in subDir:
        ImgPath = FindDirContainPos(ImgSubPath)
        return ImgPath
    else:
        return ImgPath


def process_position_list(img_obj_list, config):
    """Make sure all members of positions are part of io_obj.
    If positions = 'all', replace with actual list of positions

    Parameters
    ----------
    img_obj_list: list
        list of mManagerReader instances
    config: obj
        ConfigReader instance

    Returns
    -------
        img_obj_list: list
        list of modified mManagerReader instances

    """
    for idx, io_obj in enumerate(img_obj_list):
        config_pos_list = config.dataset.positions[idx]

        if not config_pos_list[0] == 'all':
            try:
                img_obj_list[idx].pos_list = config_pos_list
            except Exception as e:
                print('Position list {} for sample in {} is invalid'.format(config_pos_list, io_obj.ImgSmPath))
                ValueError(e)
    return img_obj_list


def process_z_slice_list(img_obj_list, config):
    """Make sure all members of z_slices are part of io_obj.
    If z_slices = 'all', replace with actual list of z_slices

      Parameters
    ----------
    img_obj_list: list
        list of mManagerReader instances
    config: obj
        ConfigReader instance

    Returns
    -------
        img_obj_list: list
        list of modified mManagerReader instances

    """
    n_slice_local_bg = config.processing.n_slice_local_bg
    for idx, io_obj in enumerate(img_obj_list):
        config_z_list = config.dataset.z_slices[idx]
        metadata_z_list = range(io_obj.nZ)
        if config_z_list[0] == 'all':
            z_list = metadata_z_list
        else:
            assert set(config_z_list).issubset(metadata_z_list), \
            'z_slice list {} for sample in {} is invalid'.format(config_z_list, io_obj.ImgSmPath)
            z_list = config_z_list

        if not n_slice_local_bg == 'all':
            # adjust slice number to be multiple of n_slice_local_bg
            z_list = z_list[0:len(z_list)//n_slice_local_bg * n_slice_local_bg]
        img_obj_list[idx].ZList = z_list
    return img_obj_list


def process_timepoint_list(img_obj_list, config):
    """Make sure all members of timepoints are part of io_obj.
    If timepoints = 'all', replace with actual list of timepoints

      Parameters
    ----------
    img_obj_list: list
        list of mManagerReader instances
    config: obj
        ConfigReader instance

    Returns
    -------
        img_obj_list: list
        list of modified mManagerReader instances
    """
    for idx, io_obj in enumerate(img_obj_list):
        config_t_list = config.dataset.timepoints[idx]
        metadata_t_list = range(io_obj.nTime)
        if config_t_list[0] == 'all':
            t_list = metadata_t_list
        else:
            assert set(config_t_list).issubset(metadata_t_list), \
            'timepoint list {} for sample in {} is invalid'.format(config_t_list, io_obj.ImgSmPath)
            t_list = config_t_list
        
        img_obj_list[idx].TimeList = t_list
    return img_obj_list


def copy_files_in_sub_dirs(input_path, output_path):
    """copy files in each sub-directory in the input path to
    output path

    Parameters
    ----------
    input_path: str
        input path
    output_path:
        output path

    """
    assert os.path.exists(input_path), 'Input folder does not exist!'
    os.makedirs(output_path, exist_ok=True)
    sub_dir_paths = glob.glob(os.path.join(input_path, '*/'))
    for sub_dir_path in sub_dir_paths:
        src_file_paths = glob.glob(os.path.join(sub_dir_path, '*.*'))
        for src_file_path in src_file_paths:
            if os.path.isfile(src_file_path):
                copy2(src_file_path, output_path)


def loadTiff(acquDirPath, acquFiles):
    """Load a single tiff file

    Parameters
    ----------
    acquDirPath : str
        directory of the tiff file
    acquFiles
        name of the tiff file
    Returns
    -------
    img : 2D float32 array
        image

    """

    TiffFile = os.path.join(acquDirPath, acquFiles)
    img = cv2.imread(TiffFile,-1) # flag -1 to preserve the bit dept of the raw image
    img = img.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    # img = img.reshape(img.shape[0], img.shape[1],1)
    return img


def parse_tiff_input(img_io):
    """Parse tiff file name following mManager/Polacquisition output format
    return images parsed based on their imaging modalities with shape (channel, height,
    width)

    Parameters
    ----------
    img_io : obj
        mManagerReader instance

    Returns
    -------
    ImgPol : 3d float32 arrays
        images from polarization state channels
    ImgProc : 3d float32 arrays
        Polacquisition processed images. Retardance and slow axis
    ImgFluor : 3d float32 arrays
        images from fluorescence channels. Currently only parse '405', '488', '568', '640' channels
    ImgBF : 3d float32 arrays
        images from bright-field channels

    """
    acquDirPath = img_io.img_in_pos_path
    acquFiles = os.listdir(acquDirPath)
    ImgPol = np.zeros((4, img_io.height,img_io.width)) # pol channels has minimum 4 channels
    ImgProc = []
    ImgBF = []
    ImgFluor = np.zeros((4, img_io.height,img_io.width)) # assuming 4 flour channels for now
    tIdx = img_io.tIdx
    zIdx = img_io.zIdx
    for fileName in acquFiles: # load raw images with Sigma0, 1, 2, 3 states, and processed images
        matchObj = re.match( r'img_000000%03d_(.*)_%03d.tif'%(tIdx,zIdx), fileName, re.M|re.I) # read images with "state" string in the filename
        if matchObj:
            img = loadTiff(acquDirPath, fileName)
            img -= img_io.blackLevel
            if any(substring in matchObj.group(1) for substring in ['State', 'state', 'Pol']):
                if '0' in matchObj.group(1):
                    ImgPol[0, :, :] = img
                elif '1' in matchObj.group(1):
                    ImgPol[1, :, :] = img
                elif '2' in matchObj.group(1):
                    ImgPol[2, :, :] = img
                elif '3' in matchObj.group(1):
                    ImgPol[3, :, :] = img
                elif '4' in matchObj.group(1):
                    img = np.reshape(img, (1, img_io.height, img_io.width))
                    ImgPol = np.concatenate((ImgPol, img))
            elif any(substring in matchObj.group(1) for substring in ['Computed Image']):
                ImgProc += [img]
            elif any(substring in matchObj.group(1) for substring in
                     ['Confocal40','Confocal_40', 'Widefield', 'widefield', 'Fluor']):
                if any(substring in matchObj.group(1) for substring in ['DAPI', '405', '405nm']):
                    ImgFluor[0,:,:] = img
                elif any(substring in matchObj.group(1) for substring in ['GFP', '488', '488nm']):
                    ImgFluor[1,:,:] = img
                elif any(substring in matchObj.group(1) for substring in ['TxR', 'TXR', 'TX', '568', '561', '560']):
                    ImgFluor[2,:,:] = img
                elif any(substring in matchObj.group(1) for substring in ['Cy5', 'IFP', '640', '637']):
                    ImgFluor[3,:,:] = img
            elif any(substring in matchObj.group(1) for substring in ['BF']):
                ImgBF += [img]
    ImgPol = sort_pol_channels(ImgPol)
    if ImgProc:
        ImgProc = np.stack(ImgProc)
    if ImgBF:
        ImgBF = np.stack(ImgBF)
    return ImgPol, ImgProc, ImgFluor, ImgBF


def sort_pol_channels(img_pol):
    """sort Polacquisition output images according to their polarization states

    Parameters
    ----------
    img_pol : 3d float32 arrays
        images of polarization state channels output by Polacquisition plug-in

    Returns
    img_pol : 3d float32 arrays
        sorted polarization images in order of I_ext, I_0, I_45, I_90, I_135
    -------

    """
    I_ext = img_pol[0, :, :]  # Sigma0 in Fig.2
    I_90 = img_pol[1, :, :]  # Sigma2 in Fig.2
    I_135 = img_pol[2, :, :]  # Sigma4 in Fig.2
    I_45 = img_pol[3, :, :]  # Sigma3 in Fig.2
    if img_pol.shape[0] == 4:  # if the images were taken using 4-frame scheme
        img_pol = np.stack((I_ext, I_45, I_90, I_135))  # order the channel following stokes calculus convention
    elif img_pol.shape[0] == 5:  # if the images were taken using 5-frame scheme
        I_0 = img_pol[4, :, :]
        img_pol = np.stack((I_ext, I_0, I_45, I_90, I_135))  # order the channel following stokes calculus convention
    return img_pol


def exportImg(img_io, img_dict):
    """export images in tiff format

    Parameters
    ----------
    img_io : obj
        mManagerReader instance
    img_dict:  dict
        dictionary of images with (key, value) = (channel, image array)
    -------

    """
    tIdx = img_io.tIdx
    zIdx = img_io.zIdx
    posIdx = img_io.posIdx
    output_path = img_io.img_out_pos_path
    for tiffName in img_dict:
        if tiffName in img_io.chNamesOut:
            fileName = 'img_'+tiffName+'_t%03d_p%03d_z%03d.tif'%(tIdx, posIdx, zIdx)
            if len(img_dict[tiffName].shape)<3:
                cv2.imwrite(os.path.join(output_path, fileName), img_dict[tiffName])
            else:
                cv2.imwrite(os.path.join(output_path, fileName), cv2.cvtColor(img_dict[tiffName], cv2.COLOR_RGB2BGR))

