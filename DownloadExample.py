import os
from google_drive_downloader import GoogleDriveDownloader as gdd




### Please specify data to download and process! ###

process_data = ['mouse_brain', 'mouse_kidney']
# (list) List specifying dataset to download and process
##   ['mouse_brain', 'mouse_kidney']

data_path_parameter = {'mouse_brain' : {'gdd_id': '1pB25UcE2nL5ZOuOaoAxTFHf1D3rbnH3f', 
                                        'zip_path': '/mouse_brain_downloaded.zip'},
                       
                       'mouse_kidney': {'gdd_id': '1N7TxmohOJRi5kTkvf02RaEoCoAuaQ-X7', 
                                        'zip_path': '/mouse_kidney_downloaded.zip'}}




if __name__ == '__main__':
    """
    Reconstruct data shared on the google drive.
    
    Parameters
    ----------
    process_data : list
                   List specifying dataset to download and process
    
    Returns
    --------
    Outputs data to disk.
    """
    configfiles = []
    
    working_folder = os.getcwd() + '/data_downloaded'
    recon_folder = working_folder + '/recon_result'
    
    if not os.path.isdir(working_folder):
        os.mkdir(working_folder)
        print("\nsetting up data folder "+working_folder)
        
    if not os.path.isdir(recon_folder):
        os.mkdir(recon_folder)
        print("\nsetting up recon folder "+recon_folder)
        
        
    for item in process_data:
        
        item_gdd_id = data_path_parameter[item]['gdd_id']
        zipdir = working_folder + data_path_parameter[item]['zip_path']
        
        gdd.download_file_from_google_drive(file_id=item_gdd_id,
                                            dest_path=zipdir,
                                            unzip=True,
                                            showsize=True,
                                            overwrite=True)

