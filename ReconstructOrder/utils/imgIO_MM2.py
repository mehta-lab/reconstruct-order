"""
Read and write Tiff in mManager format. Will be replaced by mManagerIO.py 
"""
import os
import numpy as np
import glob
import cv2
from shutil import copy2
import natsort



def get_sorted_names(dir_name):
    """
    Get image names in directory and sort them by their indices

    :param str dir_name: Image directory name
    :return list of strs im_names: Image names sorted according to indices
    """
    im_names = [f for f in os.listdir(dir_name) if f.endswith('Stack.ome.tif')]
    # Sort image names according to indices
    return natsort.natsorted(im_names)

def get_sub_dirs(ImgPath):
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
    return natsort.natsorted(subDirName)


def FindDirContain_pos(ImgPath):
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
    subDirName = get_sub_dirs(ImgPath)
    assert subDirName, 'No "Pos" directories found. Check if the input folder contains "Pos"'
    subDir = subDirName[0]  # get pos0 if it exists
    ImgSubPath = os.path.join(ImgPath, subDir)
    if 'Pos' not in subDir:
        ImgPath = FindDirContain_pos(ImgSubPath)
        return ImgPath
    else:
        return ImgPath




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
    return img



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


def export_img(img_io, img_dict, separate_pos=False):
    """export images in tiff format

    Parameters
    ----------
    img_io : obj
        mManagerReader instance
    img_dict:  dict
        dictionary of images with (key, value) = (channel, image array)
    separate_pos: bool
        save images from different positions in separate folders if True
    -------

    """
    t_idx = img_io.t_idx
    z_idx = img_io.z_idx
    pos_idx = img_io.pos_idx
    if separate_pos:
        pos_name = img_io.pos_list[pos_idx]
        output_path = os.path.join(img_io.img_output_path, pos_name)
        os.makedirs(output_path, exist_ok=True)  # create folder for processed images
    else:
        output_path = img_io.img_output_path

    for tiffName in img_dict:
        if tiffName in img_io.output_chans:
            fileName = 'img_'+tiffName+'_t%03d_p%03d_z%03d.tif'%(t_idx, pos_idx, z_idx)
            if len(img_dict[tiffName].shape)<3:
                cv2.imwrite(os.path.join(output_path, fileName), img_dict[tiffName])
            else:
                cv2.imwrite(os.path.join(output_path, fileName), cv2.cvtColor(img_dict[tiffName], cv2.COLOR_RGB2BGR))

