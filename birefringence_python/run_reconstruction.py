"""
Reconstruct retardance and orientation maps from images taken with different polarized illumination output
by Open PolScope. This script using the 4- or 5-frame reconstruction algorithm described in Michael Shribak and 
Rudolf Oldenbourg, 2003.
 
* outputChann: (list) output channel names
    Current available output channel names:
        'Transmission'
        'Retardance'
        'Orientation' 
        'Retardance+Orientation'
        'Transmission+Retardance+Orientation'
        '405'
        '488'
        '568'
        '640'
        
* circularity: (bool) flip the sign of polarization. Set "True" for Dragonfly and "False" for ASI
* bgCorrect: (str) 
    'Auto' (default) to correct the background using background from the metadata if available, otherwise use input background folder;
    'None' for no background correction; 
    'Input' to always use input background folder   
* flatField: (bool) perform flat-field correction if True
* norm: (bool) scale images individually for optimal dynamic range. Set False for tiled images
    
"""

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import os
sys.path.append(".") # Adds current directory to python search path.
sys.path.append("..") # Adds parent directory to python search path.
# sys.path.append(os.path.dirname(sys.argv[0]))
from PolScope.multiDimProcess import findBackground, loopPos
from utils.imgIO import GetSubDirName
# import seaborn as sns
import os

# sns.set_context("talk")


# In[2]:
def processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann, BgDir_local=None, flatField=False,
               bgCorrect=True, circularity=False, method='Stokes', norm=True):
    print('Processing ' + SmDir + ' ....')
    imgSm = findBackground(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann,
                           BgDir_local=BgDir_local, flatField=flatField,bgCorrect=bgCorrect,
                           recon_method=method, ff_method='open') # find background tile
    imgSm.loopZ ='sample'
    imgSm = loopPos(imgSm, outputChann, flatField=flatField, bgCorrect=bgCorrect, circularity=circularity, norm=norm)
        
            
# In[3]:


RawDataPath = '/flexo/ComputationalMicroscopy/SpinningDisk/RawData/brainarchitecture'
# RawDataPath = '/flexo/ComputationalMicroscopy/SpinningDisk/RawData/virtualstaining'
# # RawDataPath = '/data/sguo/Data'
# ProcessedPath = '/data/sguo/Processed'
# ProcessedPath = '/flexo/ComputationalMicroscopy/Processed/virtualstaining/u2os_leonetti'
ProcessedPath = '/flexo/ComputationalMicroscopy/Processed/brainarchitecture'
# RawDataPath = r'\\flexo\MicroscopyData\ComputationalMicroscopy\Hackathon\Data'
# ProcessedPath = r'\\flexo\MicroscopyData\ComputationalMicroscopy\Hackathon\Processed\test'
# RawDataPath = r'\\flexo\MicroscopyData\AdvancedOpticalMicroscopy\SpinningDisk\RawData\virtualstaining\2018_12_05-07_A549_MembraneLabel_CellMask_WGA'
# ProcessedPath = r'\\flexo\MicroscopyData\AdvancedOpticalMicroscopy\Processed\virtualistaining\RSVinfection\2018_12_05-07_A549_MembraneLabel_CellMask_WGA'

# RawDataPath = r'D:/Box Sync/Data'
# ProcessedPath = r'D:/Box Sync/Processed/'

ImgDir = '2018_12_27_background_optimization_Nowakowski_slide'
SmDir = 'SMS_2018_1227_0138_3'
BgDir = 'BG_2018_1227_1311_1'
BgDir_local = 'BGS_2018_1227_1802_2'

# ImgDir = '2018_12_07_A549_MembraneLabel_WGA_NoPerm_v3'
# SmDir = 'FOV1_1'
# BgDir = 'BG_2018_1207_1058_1'
# ImgDir = '2018_12_07_A549_MembraneLabel_WGA_NoPerm_v3'
# SmDir = 'FOV1_1'
# BgDir = 'BG_2018_1207_1058_1'

# ImgDir = '2018_11_01_kidney_slice'
# SmDir = 'SMS_2018_1101_1713_1_1'
# BgDir = 'BG_2018_1101_1705_1'

# ImgDir = '2018_11_26_Argolight_channel_registration_63X_confocal'
# SmDir = 'SMS_2018_1126_1625_1'
# BgDir = 'BG_2018_1126_1621_1'

# ImgDir = '2018_11_20_488ALDH1L1_594IBA1_647GFAP_63X'
# SmDir = 'SMS_2018_1120_1637_1_1'
# BgDir = 'BG_2018_1120_1650_1'

# ImgDir = '2018_11_28_594CTIP2_647SATB2_10X'
# SmDir = 'SMS_2018_1128_1557_1_3'
# BgDir = 'BG_2018_1128_1557_1'

# ImgDir = '2018_11_20_488L1CAM_594VIM'
# SmDir = 'SMS_2018_1120_1557_1_1'
# BgDir = 'BG_2018_1120_1537_1'

# ImgDir = '2018_09_25_ctx_sagittal_l1cam_vimentin'
# SmDir = 'SMS_2018_1120_1557_1_1'
# BgDir = 'BG_2018_1120_1537_1'

# ImgDir = '2018_10_12_human_brain_NeuN_C11_TB_63X'
# SmDir = 'SMS_2018_1012_1749_1'
# BgDir = 'BG_2018_1012_1902_1'

# ImgDir = '2018_10_18_RainbowCells'
# SmDir = 'SMS_2018_1018_1828_1'
# BgDir = 'BG_2018_1018_1900_1'

# ImgDir = '2018_09_28_U2OS_rainbow'
# SmDir = 'SMS_2018_0928_1706_1_3'
# BgDir = 'BG_2018_0928_1641_1'


# ImgDir = '2018_11_01_kidney_slice'
# SmDir = 'SM_2018_1101_1835_1'
# BgDir = 'BG_2018_1101_1834_1'


# ImgDir = '2018_10_02_MouseBrainSlice_3D'
# SmDir = 'SMS_2018_1002_1714_1'
# BgDir = 'BG_2018_1002_1740_1'

# SmDir = 'SM_2018_0928_1644_1'
# BgDir = 'BG_2018_0928_1641_1'

# ImgDir = '20180914_GW20_CUBIC_DAPI'
# SmDir = 'SMS_2018_0914_1748_1'
# BgDir = 'BG_2018_0914_1732_1'

#ImgDir = '20180816_Sample_Test_CUBIC'
#SmDir = 'SM_2018_0801_1313_1'
#SmDir = 'SM_2018_0816_1838_1'
#BgDir = 'BG_2018_0816_1613_1'

#ImgDir = '2018_08_01_differentiation_Marius'
#SmDir = 'SM_2018_0801_1313_1'
#SmDir = 'BG_2018_0801_1333_1'
#BgDir = 'BG_2018_0801_1333_1'
#BgDir = 'BG_2018_0801_1214_1'

# ImgDir = 'NikonSmallWorld'
# SmDir = 'SMS_2018_0425_1654_1'
# BgDir = 'BG_2018_0425_1649_1'

# ImgDir = '20180710_brainSlice_Tomasz'
# SmDir = 'SM_2018_0710_1830_1'
# BgDir = 'BG_2018_0710_1820_1'

#ImgDir = '2018_08_02_Galina_test_condition'
#SmDir = 'BG_2018_0802_1605_1'
#SmDir = 'SM_2018_0802_1521_1'
#BgDir = 'BG_2018_0802_1508_1'


outputChann = ['Transmission', 'Retardance', 'Orientation', 'Scattering', 'Retardance+Orientation',
               'Transmission+Retardance+Orientation', 'Scattering+Orientation', 'Retardance+Fluorescence',
               '405', '488', '568', '640']

# outputChann = ['Transmission', 'Retardance', 'Orientation', 'Scattering', 'Retardance+Orientation',
# 'Scattering+Orientation', 'Transmission+Retardance+Orientation']
                            
# channels to output, see readme for channel names
circularity= 'rcp' # circularity of the analyzer, lcp or rcp
# bgCorrect='Auto'
# bgCorrect='Local_filter'
bgCorrect='Local_defocus'
# Auto: correct the background using background from the metadata  
flatField = False
batchProc = False
norm = True
recon_method = 'Stokes'
# recon_method = 'Jones'

if batchProc:
    ImgPath = os.path.join(RawDataPath, ImgDir)
    SmDirList = GetSubDirName(ImgPath)
    for SmDir in SmDirList:
        # if 'SM' in SmDir or 'BG' in SmDir :
        if 'SM' in SmDir and 'SMS' not in SmDir:
        # if 'SM' in SmDir:
            processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann,
                       BgDir_local=BgDir_local, flatField=flatField, bgCorrect=bgCorrect,
                       circularity=circularity, method=recon_method, norm=norm)
else:
    processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann,
               BgDir_local=BgDir_local, flatField=flatField, bgCorrect=bgCorrect,
               circularity=circularity, method=recon_method, norm=norm)



