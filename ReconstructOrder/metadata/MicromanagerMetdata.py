# bchhun, {2020-01-17}


def mm1_meta_parser(metafile):
    try:
        pos_dict_list = metafile['Summary']['InitialPositionList']
        if pos_dict_list:
            meta_pos_list = [pos_dict['Label'] for pos_dict in pos_dict_list]
        else:
            meta_pos_list = ['Pos0']
    except KeyError:
        print("WARNING: no InitialPositionList found in metadata.txt, setting to Pos0")
        meta_pos_list = ['Pos0']

    try:
        width = metafile['Summary']['Width']
        height = metafile['Summary']['Height']
        time_stamp = metafile['Summary']['Time']
    except KeyError:
        print("WARNING: no width, height, or Time found in metadata.txt, setting to default")
        width = 2048
        height = 2048
        time_stamp = 1

    return meta_pos_list, width, height, time_stamp


def mm2_beta_meta_parser(metafile):
    try:
        if 'StagePositions' in metafile['Summary']:
            pos_dict_list = metafile['Summary']['StagePositions']
            _meta_pos_list = [pos_dict['Label'] for pos_dict in pos_dict_list]
        else:
            _meta_pos_list = ['']
    except KeyError:
        print("WARNING: no InitialPositionList found in metadata.txt, setting to Pos0")
        _meta_pos_list = ['']

    try:
        width = int(metafile['Summary']['UserData']['Width']['PropVal'])
        height = int(metafile['Summary']['UserData']['Height']['PropVal'])
        time_stamp = metafile['Summary']['StartTime']
    except KeyError:
        print("WARNING: no width, height, or Time found in metadata.txt, setting to default")
        width = 2048
        height = 2048
        time_stamp = 1

    return _meta_pos_list, width, height, time_stamp


def mm2_gamma_meta_parser(metafile):

    try:
        if 'StagePositions' in metafile['Summary']:
            pos_dict_list = metafile['Summary']['StagePositions']
            _meta_pos_list = [pos_dict['Label'] for pos_dict in pos_dict_list]
        else:
            _meta_pos_list = ['']
    except KeyError:
        print("WARNING: no InitialPositionList found in metadata.txt, setting to Pos0")
        _meta_pos_list = ['']

    try:
        width = metafile['Summary']['Width']
        height = metafile['Summary']['Height']
        time_stamp = metafile['Summary']['StartTime']
    except KeyError:
        print("WARNING: no width, height, or Time found in metadata.txt, setting to default")
        width = 2048
        height = 2048
        time_stamp = 1

    return _meta_pos_list, width, height, time_stamp


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


def mm_meta_parser(metafile, type='gamma'):
    # Parse position list
    try:
        if type == '1.4.22' and 'InitialPositionList' in metafile['Summary']:
            meta_pos_list = [pos_dict['Label'] for pos_dict in metafile['Summary']['InitialPositionList']]
        elif type == '1.4.22':
            # default value
            meta_pos_list = ['Pos0']

        elif type == 'beta' and 'StagePositions' in metafile['Summary']:
            meta_pos_list = [pos_dict['Label'] for pos_dict in metafile['Summary']['StagePositions']]
        elif type == 'beta':
            # default value
            meta_pos_list = ['']

        elif type == 'gamma' and 'StagePositions' in metafile['Summary']:
            meta_pos_list = [pos_dict['Label'] for pos_dict in metafile['Summary']['StagePositions']]
        elif type == 'gamma':
            # default value
            meta_pos_list = ['']

        else:
            raise ValueError(f"metadata of type {type} is not supported")

    except KeyError:
        print("WARNING: no InitialPositionList found in metadata.txt, setting to Pos0")
        if type == '1.4.22':
            meta_pos_list = ['Pos0']
        else:
            meta_pos_list = ['']
    except ValueError as ve:
        print(str(ve))

    # Parse Width, Height, Time
    try:
        if type == '1.4.22':
            width = metafile['Summary']['Width']
            height = metafile['Summary']['Height']
            time_stamp = metafile['Summary']['Time']
        elif type == 'beta':
            width = int(metafile['Summary']['UserData']['Width']['PropVal'])
            height = int(metafile['Summary']['UserData']['Height']['PropVal'])
            time_stamp = metafile['Summary']['StartTime']
        elif type == 'gamma':
            width = metafile['Summary']['Width']
            height = metafile['Summary']['Height']
            time_stamp = metafile['Summary']['StartTime']
        else:
            raise ValueError(f"metadata of type {type} is not supported")

    except KeyError:
        print("WARNING: no width, height, or Time found in metadata.txt, setting to default")
        width = 2048
        height = 2048
        time_stamp = 1

    return meta_pos_list, width, height, time_stamp
