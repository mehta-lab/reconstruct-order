# bchhun, {4/26/19}
from ..workflow.multiDimProcess import process_background, loopPos, compute_flat_field, read_metadata, parse_bg_options
from ..utils.ConfigReader import ConfigReader
from ..utils.imgIO import process_position_list, process_z_slice_list, process_timepoint_list
import os


def _processImg(img_obj, bg_obj, config):
    print('Processing ' + img_obj.name + ' ....')

    # Write metadata in processed folder
    img_obj.chNamesIn = img_obj.chNamesOut
    img_obj.writeMetaData()

    # Write config file in processed folder
    config.write_config(os.path.join(img_obj.ImgOutPath, 'config.yml'))  # save the config file in the processed folder

    img_obj, img_reconstructor = process_background(img_obj, bg_obj, config)

    flatField = config.processing.flatfield_correction
    if flatField:  # find background fluorescence for flatField correction
        img_obj = compute_flat_field(img_obj, config)

    img_obj.loopZ = 'reconstruct'
    img_obj = loopPos(img_obj, config, img_reconstructor)


def reconstructBatch(configfile):
    config = ConfigReader()
    config.read_config(configfile)

    # read meta data
    img_obj_list, bg_obj_list = read_metadata(config)
    img_obj_list = process_position_list(img_obj_list, config)
    img_obj_list = process_z_slice_list(img_obj_list, config)
    img_obj_list = process_timepoint_list(img_obj_list, config)
    # process background options
    img_obj_list = parse_bg_options(img_obj_list, config)

    for img_obj, bg_obj in zip(img_obj_list, bg_obj_list):
        _processImg(img_obj, bg_obj, config)
