# bchhun, {2019-09-16}


import pytest
import os

from google_drive_downloader import GoogleDriveDownloader as gdd


@pytest.fixture(scope="session")
def setup_mm1422_background_src():
    """

    :return:
    """
    temp_folder = os.getcwd() + '/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder "+temp_folder)
    if not os.path.isdir(temp_folder+'/src'):
        os.mkdir(temp_folder+'/src')
        print("\nsetting up src folder "+temp_folder+'/src')
    if not os.path.isdir(temp_folder+'/predict'):
        os.mkdir(temp_folder+'/predict')
        print("\nsetting up predict folder "+temp_folder+'/predict')

    # DO NOT ADJUST THESE VALUES
    bulk_file = '1EGDljmabXMheGSbLi-BK0_9p6ykMfW7L'

    srczip = temp_folder + '/src' + '/mm1422_background.zip'
    gdd.download_file_from_google_drive(file_id=bulk_file,
                                        dest_path=srczip,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    yield ''

    # breakdown files
    import shutil
    print("breaking down temp files in "+temp_folder)
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)


@pytest.fixture(scope="session")
def setup_mm1422_kazansky_grid_src():
    """

    :return:
    """
    temp_folder = os.getcwd() + '/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder " + temp_folder)
    if not os.path.isdir(temp_folder + '/src'):
        os.mkdir(temp_folder + '/src')
        print("\nsetting up src folder " + temp_folder + '/src')
    if not os.path.isdir(temp_folder + '/predict'):
        os.mkdir(temp_folder + '/predict')
        print("\nsetting up predict folder " + temp_folder + '/predict')

    # DO NOT ADJUST THESE VALUES
    bulk_file = '1gsnyiBDNtnhmmvZyUDClAKmvoY1J1NmH'

    srczip = temp_folder + '/src' + '/mm1422_kazansky_grid.zip'
    gdd.download_file_from_google_drive(file_id=bulk_file,
                                        dest_path=srczip,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    # read and modify downloaded .yml if necessary here
    config_file = temp_folder+"/src/mm1422_kazansky_grid.yml"

    yield config_file

    # breakdown files
    import shutil
    print("breaking down temp files in " + temp_folder)
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)


@pytest.fixture(scope="session")
def setup_mm1422_kazansky_HCS_snake_src():
    """

    :return:
    """
    temp_folder = os.getcwd() + '/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder " + temp_folder)
    if not os.path.isdir(temp_folder + '/src'):
        os.mkdir(temp_folder + '/src')
        print("\nsetting up src folder " + temp_folder + '/src')
    if not os.path.isdir(temp_folder + '/predict'):
        os.mkdir(temp_folder + '/predict')
        print("\nsetting up predict folder " + temp_folder + '/predict')

    # DO NOT ADJUST THESE VALUES
    bulk_file = '1BRuUacZLBdreVVHUeHeGT5nOrz482Vxh'

    srczip = temp_folder + '/src' + '/mm1422_kazansky_HCS_snake.zip'
    gdd.download_file_from_google_drive(file_id=bulk_file,
                                        dest_path=srczip,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    # read and modify downloaded .yml if necessary here
    config_file = temp_folder+"/src/mm1422_kazansky_HCS_snake.yml"

    yield config_file

    # breakdown files
    import shutil
    print("breaking down temp files in " + temp_folder)
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)


@pytest.fixture(scope="session")
def setup_mm1422_kazansky_one_position_src():
    """

    :return:
    """
    temp_folder = os.getcwd() + '/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder " + temp_folder)
    if not os.path.isdir(temp_folder + '/src'):
        os.mkdir(temp_folder + '/src')
        print("\nsetting up src folder " + temp_folder + '/src')
    if not os.path.isdir(temp_folder + '/predict'):
        os.mkdir(temp_folder + '/predict')
        print("\nsetting up predict folder " + temp_folder + '/predict')

    # DO NOT ADJUST THESE VALUES
    bulk_file = '1wUbnf3tvRhxJopGn2PK-Buv436Eax8xp'

    srczip = temp_folder + '/src' + '/mm1422_kazansky_one_position.zip'
    gdd.download_file_from_google_drive(file_id=bulk_file,
                                        dest_path=srczip,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    # read and modify downloaded .yml if necessary here
    config_file = temp_folder+"/src/mm1422_kazansky_one_position.yml"

    yield config_file

    # breakdown files
    import shutil
    print("breaking down temp files in " + temp_folder)
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)


@pytest.fixture(scope="session")
def setup_mm2_gamma_background_src():
    """

    :return:
    """
    temp_folder = os.getcwd() + '/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder "+temp_folder)
    if not os.path.isdir(temp_folder+'/src'):
        os.mkdir(temp_folder+'/src')
        print("\nsetting up src folder "+temp_folder+'/src')
    if not os.path.isdir(temp_folder+'/predict'):
        os.mkdir(temp_folder+'/predict')
        print("\nsetting up predict folder "+temp_folder+'/predict')

    # DO NOT ADJUST THESE VALUES
    bulk_file = '1yrQRag78zswUNgs8HZvgBy8Mq5Jn6LMu'

    srczip = temp_folder + '/src' + '/mm2_gamma_background.zip'
    gdd.download_file_from_google_drive(file_id=bulk_file,
                                        dest_path=srczip,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    yield ''

    # breakdown files
    import shutil
    print("breaking down temp files in "+temp_folder)
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)


@pytest.fixture(scope="session")
def setup_mm2_gamma_kazansky_grid_src():
    """

    :return:
    """
    temp_folder = os.getcwd() + '/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder " + temp_folder)
    if not os.path.isdir(temp_folder + '/src'):
        os.mkdir(temp_folder + '/src')
        print("\nsetting up src folder " + temp_folder + '/src')
    if not os.path.isdir(temp_folder + '/predict'):
        os.mkdir(temp_folder + '/predict')
        print("\nsetting up predict folder " + temp_folder + '/predict')

    # DO NOT ADJUST THESE VALUES
    bulk_file = '1ncIxsBsYazgOmFLiQFGwKitITwnuGtkj'

    srczip = temp_folder + '/src' + '/mm2_gamma_kazansky_grid.zip'
    gdd.download_file_from_google_drive(file_id=bulk_file,
                                        dest_path=srczip,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    # read and modify downloaded .yml if necessary here
    config_file = temp_folder+"/src/mm2-gamma_kazansky_grid.yml"

    yield config_file

    # breakdown files
    import shutil
    print("breaking down temp files in " + temp_folder)
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)


@pytest.fixture(scope="session")
def setup_mm2_gamma_kazansky_HCS_one_position_src():
    """

    :return:
    """
    temp_folder = os.getcwd() + '/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder " + temp_folder)
    if not os.path.isdir(temp_folder + '/src'):
        os.mkdir(temp_folder + '/src')
        print("\nsetting up src folder " + temp_folder + '/src')
    if not os.path.isdir(temp_folder + '/predict'):
        os.mkdir(temp_folder + '/predict')
        print("\nsetting up predict folder " + temp_folder + '/predict')

    # DO NOT ADJUST THESE VALUES
    bulk_file = '1M2rHDhklYx5XUZxm0qiuOqMHeSgga9nz'

    srczip = temp_folder + '/src' + '/mm2-gamma_kazansky_HCS_one_position.zip'
    gdd.download_file_from_google_drive(file_id=bulk_file,
                                        dest_path=srczip,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    # read and modify downloaded .yml if necessary here
    config_file = temp_folder+"/src/mm2-gamma_kazansky_HCS_one_position.yml"

    yield config_file

    # breakdown files
    import shutil
    print("breaking down temp files in " + temp_folder)
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)


@pytest.fixture(scope="session")
def setup_mm2_gamma_kazansky_HCS_snake_src():
    """

    :return:
    """
    temp_folder = os.getcwd() + '/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder " + temp_folder)
    if not os.path.isdir(temp_folder + '/src'):
        os.mkdir(temp_folder + '/src')
        print("\nsetting up src folder " + temp_folder + '/src')
    if not os.path.isdir(temp_folder + '/predict'):
        os.mkdir(temp_folder + '/predict')
        print("\nsetting up predict folder " + temp_folder + '/predict')

    # DO NOT ADJUST THESE VALUES
    bulk_file = '1bwf29rqxZOs0VDYGmxT-P5dv20bKxtta'

    srczip = temp_folder + '/src' + '/mm2-gamma_kazansky_HCS_snake.zip'
    gdd.download_file_from_google_drive(file_id=bulk_file,
                                        dest_path=srczip,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    # read and modify downloaded .yml if necessary here
    config_file = temp_folder+"/src/mm2-gamma_kazansky_HCS_snake.yml"

    yield config_file

    # breakdown files
    import shutil
    print("breaking down temp files in " + temp_folder)
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)


@pytest.fixture(scope="session")
def setup_mm2_gamma_kazansky_HCS_typewriter_src():
    """

    :return:
    """
    temp_folder = os.getcwd() + '/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder " + temp_folder)
    if not os.path.isdir(temp_folder + '/src'):
        os.mkdir(temp_folder + '/src')
        print("\nsetting up src folder " + temp_folder + '/src')
    if not os.path.isdir(temp_folder + '/predict'):
        os.mkdir(temp_folder + '/predict')
        print("\nsetting up predict folder " + temp_folder + '/predict')

    # DO NOT ADJUST THESE VALUES
    bulk_file = '13tg6V0U0qdii1MRjq597cc6g50tHox05'

    srczip = temp_folder + '/src' + '/mm2-gamma_kazansky_HCS_typewriter.zip'
    gdd.download_file_from_google_drive(file_id=bulk_file,
                                        dest_path=srczip,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    # read and modify downloaded .yml if necessary here
    config_file = temp_folder+"/src/mm2-gamma_kazansky_HCS_typewriter.yml"

    yield config_file

    # breakdown files
    import shutil
    print("breaking down temp files in " + temp_folder)
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)