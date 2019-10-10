# bchhun, {2019-09-16}

import pytest
import os
from google_drive_downloader import GoogleDriveDownloader as gdd


@pytest.fixture
def setup_gdrive_data():

    temp_folder = os.getcwd()+'/temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # DO NOT ADJUST THESE VALUES
    bg_state0_url = '1XY95HdH7fMkeZvf8mmVCygnHNAPMIf5g'
    bg_state1_url = '1pDCAae9ZYiGiwbbiJs0HNKfjoJQ8eBkQ'
    bg_state2_url = '1sE-18lsYixNgcXmWsN5hbb0z0ETnzzF-'
    bg_state3_url = '1sW7XFQDSyKIdaONXZavEp7qT8P-3CytR'
    bg_state4_url = '14yEDdZjgrxnqCJupVWtIBAg_m9GX9X6J'

    sm_state0_url = '1QCJJnDOSRtxnHPyhXnZH0_ZWf_AnyiGY'
    sm_state1_url = '1zCXGbfhadNrRYzxGCEB0iyplYl1hdTaU'
    sm_state2_url = '1Cx9a8nuzzaIeBBYekQIJPsmpmmR_MdOG'
    sm_state3_url = '1hh_P8mosInIjs34-Q6IXqMLiK4Wvaszg'
    sm_state4_url = '1arBhvE-ZABrjrjnk0xJnrWUTL7sya-uF'

    bg_urls = [bg_state0_url, bg_state1_url, bg_state2_url, bg_state3_url, bg_state4_url]
    sm_urls = [sm_state0_url, sm_state1_url, sm_state2_url, sm_state3_url, sm_state4_url]

    bg_out = []
    for idx, url in enumerate(bg_urls):
        output = temp_folder+"/bg_%d.tif" % idx
        gdd.download_file_from_google_drive(file_id=url,
                                            dest_path=output,
                                            unzip=False)
        bg_out.append(output)

    sm_out = []
    for idx, url in enumerate(sm_urls):
        output = temp_folder+"/sm_%d.tif" % idx
        gdd.download_file_from_google_drive(file_id=url,
                                            dest_path=output,
                                            unzip=False)
        sm_out.append(output)

    yield bg_out, sm_out

    # breakdown files
    for bg in bg_out:
        if os.path.isfile(bg):
            os.remove(bg)
            print("\nbreaking down temp file")
    for sm in sm_out:
        if os.path.isfile(sm):
            os.remove(sm)
            print("\nbreaking down temp file")
    if os.path.isdir(temp_folder):
        os.rmdir(temp_folder)
        print("breaking down temp folder")

