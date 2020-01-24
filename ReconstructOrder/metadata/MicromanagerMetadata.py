# bchhun, {2020-01-17}
import os


def mm1_meta_parser(metafile):
    """
    for 1.4.22 metadata
    - copied from SM's original parsing schema

    :param metafile:
    :return:
    """
    try:
        pos_dict_list = metafile['Summary']['InitialPositionList']
        if pos_dict_list:
            meta_pos_list = [pos_dict['Label'] for pos_dict in pos_dict_list]
        else:
            meta_pos_list = ['Pos0']
    except KeyError:
        print("WARNING mm1.4.22: no InitialPositionList found in metadata.txt, setting to Pos0")
        meta_pos_list = ['Pos0']

    try:
        width = metafile['Summary']['Width']
        height = metafile['Summary']['Height']
        time_stamp = metafile['Summary']['Time']
    except KeyError:
        print("WARNING mm1.4.22: no width, height, or Time found in metadata.txt, setting to default")
        width = 2048
        height = 2048
        time_stamp = 1

    return meta_pos_list, width, height, time_stamp


def mm2_beta_meta_parser(metafile):
    """
    for 2.0-beta
    - copied from SM's original parsing schema

    :param metafile:
    :return:
    """
    try:
        if 'StagePositions' in metafile['Summary']:
            pos_dict_list = metafile['Summary']['StagePositions']
            meta_pos_list = [pos_dict['Label'] for pos_dict in pos_dict_list]
        else:
            meta_pos_list = ['']
    except KeyError:
        print("WARNING mm2.0-beta: no InitialPositionList found in metadata.txt, setting to ''")
        meta_pos_list = ['']

    try:
        width = int(metafile['Summary']['UserData']['Width']['PropVal'])
        height = int(metafile['Summary']['UserData']['Height']['PropVal'])
        time_stamp = metafile['Summary']['StartTime']
    except KeyError:
        print("WARNING mm2.0-beta: no width, height, or Time found in metadata.txt, setting to default")
        width = 2048
        height = 2048
        time_stamp = 1

    return meta_pos_list, width, height, time_stamp


def mm2_gamma_meta_parser(metafile):
    """
    for 2.0-gamma
        Schema is:
        metadata = {'Summary": ___,


    :param metafile:
    :return:
    """

    try:
        if 'StagePositions' in metafile['Summary']:
            pos_dict_list = metafile['Summary']['StagePositions']
            meta_pos_list = [pos_dict['Label'] for pos_dict in pos_dict_list]
        else:
            raise KeyError("WARNING mm2.0-gamma: no InitialPositionList found in metadata.txt, setting to ''")
    except KeyError as k:
        print(k)
        meta_pos_list = ['']

    # every image in gamma has a corresponding "Coords" and "Metadata" dict in metadata.txt
    #   the "Metadata" dictionary contains width and height info"
    #   here we will take only the first position's metadata to find the image Width and Height
    try:
        first_metadata = [k for k, v in metafile.items() if "Metadata" in k][0]
        width = int(metafile[first_metadata]['Width'])
        height = int(metafile[first_metadata]['Height'])
        time_stamp = metafile["Summary"]['StartTime']
    except KeyError:
        print("WARNING mm2.0-gamma: no width, height, or Time found in metadata.txt, setting to default")
        width = 2048
        height = 2048
        time_stamp = 1

    return meta_pos_list, width, height, time_stamp


def create_metadata_object(data_path, config):
    """
    Reads PolAcquisition metadata, if possible. Otherwise, reads MicroManager metadata.
    TODO: move to imgIO?

    Parameters
    __________
    data_path : str
        Path to data directory
    config : obj
        ConfigReader object

    Returns
    _______
    obj
        Metadata object
    """
    # import here to avoid circular imports
    # but actually, this fails because "from" requires "." to be initialized before import
    from . import mManagerReader, PolAcquReader

    try:
        img_obj = PolAcquReader(data_path,
                                output_chans=config.processing.output_channels,
                                binning=config.processing.binning
                                )
    except:
        img_obj = mManagerReader(data_path,
                                 output_chans=config.processing.output_channels,
                                 binning=config.processing.binning)
    return img_obj


def read_metadata(config):
    """
    Reads the metadata for the sample and background data sets. Passes some
    of the parameters (e.g. swing, wavelength, back level, etc.) from the
    background metadata object into the sample metadata object
    TODO: move to imgIO?

    Parameters
    __________
    config : obj
        ConfigReader object

    Returns
    _______
    obj
        Metadata object
    """

    img_obj_list = []
    bg_obj_list = []

    # If one background is used for all samples, read only once
    if len(set(config.dataset.background)) <= 1:
        background_path = os.path.join(config.dataset.data_dir,config.dataset.background[0])
        bg_obj = create_metadata_object(background_path, config)
        bg_obj_list.append(bg_obj)
    else:
        for background in config.dataset.background:
            background_path = os.path.join(config.dataset.data_dir, background)
            bg_obj = create_metadata_object(background_path, config)
            bg_obj_list.append(bg_obj)

    for sample in config.dataset.samples:
        sample_path = os.path.join(config.dataset.data_dir, sample)
        img_obj = create_metadata_object(sample_path, config)
        img_obj_list.append(img_obj)

    if len(bg_obj_list) == 1:
        bg_obj_list = bg_obj_list*len(img_obj_list)

    for i in range(len(config.dataset.samples)):
        img_obj_list[i].bg = bg_obj_list[i].bg
        img_obj_list[i].swing = bg_obj_list[i].swing
        img_obj_list[i].wavelength = bg_obj_list[i].wavelength
        img_obj_list[i].blackLevel = bg_obj_list[i].blackLevel

    return img_obj_list, bg_obj_list


"""
Try to build a generalized parser
***not finished***

"""


# def flatten(d, parent_key='', sep='_'):
#     items = []
#     for k, v in d.items():
#         new_key = parent_key + sep + k if parent_key else k
#         if isinstance(v, dict):
#             items.extend(flatten(v, new_key, sep=sep).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)
#
#
# def mm_meta_parser(meta_dict):
#     """
#     We want the following metadata values extracted:
#     InitialPositionList
#     Label
#     :param meta_dict:
#     :return:
#     """
#
#     flat_dict = flatten(meta_dict)
#     pass

#
# def mm_meta_parser(metafile, type='gamma'):
#     # Parse position list
#     try:
#         if type == '1.4.22' and 'InitialPositionList' in metafile['Summary']:
#             meta_pos_list = [pos_dict['Label'] for pos_dict in metafile['Summary']['InitialPositionList']]
#         elif type == '1.4.22':
#             # default value
#             meta_pos_list = ['Pos0']
#
#         elif type == 'beta' and 'StagePositions' in metafile['Summary']:
#             meta_pos_list = [pos_dict['Label'] for pos_dict in metafile['Summary']['StagePositions']]
#         elif type == 'beta':
#             # default value
#             meta_pos_list = ['']
#
#         elif type == 'gamma' and 'StagePositions' in metafile['Summary']:
#             meta_pos_list = [pos_dict['Label'] for pos_dict in metafile['Summary']['StagePositions']]
#         elif type == 'gamma':
#             # default value
#             meta_pos_list = ['']
#
#         else:
#             raise ValueError(f"metadata of type {type} is not supported")
#
#     except KeyError:
#         print("WARNING: no InitialPositionList found in metadata.txt, setting to Pos0")
#         if type == '1.4.22':
#             meta_pos_list = ['Pos0']
#         else:
#             meta_pos_list = ['']
#     except ValueError as ve:
#         print(str(ve))
#
#     # Parse Width, Height, Time
#     try:
#         if type == '1.4.22':
#             width = metafile['Summary']['Width']
#             height = metafile['Summary']['Height']
#             time_stamp = metafile['Summary']['Time']
#         elif type == 'beta':
#             width = int(metafile['Summary']['UserData']['Width']['PropVal'])
#             height = int(metafile['Summary']['UserData']['Height']['PropVal'])
#             time_stamp = metafile['Summary']['StartTime']
#         elif type == 'gamma':
#             width = metafile['Summary']['Width']
#             height = metafile['Summary']['Height']
#             time_stamp = metafile['Summary']['StartTime']
#         else:
#             raise ValueError(f"metadata of type {type} is not supported")
#
#     except KeyError:
#         print("WARNING: no width, height, or Time found in metadata.txt, setting to default")
#         width = 2048
#         height = 2048
#         time_stamp = 1
#
#     return meta_pos_list, width, height, time_stamp