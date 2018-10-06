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
        
* flipPol: (bool) flip the sign of polarization. Set "True" for Dragonfly and "False" for ASI 
* bgCorrect: (str) 
    'Auto' (default) to correct the background using background from the metadata if available, otherwise use input background folder;
    'None' for no background correction; 
    'Input' to always use input background folder   
* flatField: (bool) perform flat-field correction if True
    
"""

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
from PolScope.multiDimProcess import findBackground, loopPos
from utils.imgIO import GetSubDirName
import seaborn as sns
import os

sns.set_context("talk")


# In[2]:
def processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann, flatField=False, bgCorrect=True, flipPol=False):    
    imgSm = findBackground(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann,flatField=flatField,bgCorrect=bgCorrect) # find background tile
    imgSm.loopZ ='sample'
    imgSm = loopPos(imgSm, outputChann, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol)
        
            
# In[3]:


#ImgSmPath = 'C:/Google Drive/2018_04_12_U2OSCells63x1.2/SMS_2018_0412_1735_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/2018_04_12_U2OSCells63x1.2/BG_2018_0412_1726_1' # Background image folder path
#ImgSmPath = 'C:/Users/Sam Guo/Box Sync/Data/2018_04_12_U2OSCells63x1.2/SM_2018_0412_1731_1' # Sample image folder path
#ImgBgPath = 'C:/Users/Sam Guo/Box Sync/Data/2018_04_12_U2OSCells63x1.2/BG_2018_0412_1726_1' # Background image folder path        
#ImgSmPath = 'C:/Google Drive/2018_06_22_U2OS/SM_2018_0622_1545_2/' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/2018_06_22_U2OS/BG_2018_0622_1529_1/' # Background image folder path        
#ImgSmPath = 'C:/Google Drive/2018_04_16_unstained_brain_slice/SMS_2018_0416_1825_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/2018_04_16_unstained_brain_slice/BG_2018_0416_1745_1' # Background image folder path
#ImgSmPath = 'C:/Users/Sam Guo/Box Sync/Data/NikonSmallWorld/someone/2018_04_25_Testing/SMS_2018_0425_1654_1' # Sample image folder path
#ImgBgPath = 'C:/Users/Sam Guo/Box Sync/Data/NikonSmallWorld/someone/2018_04_25_Testing/BG_2018_0425_1649_1' # Background image folder path
#ImgSmPath = 'C:/Users/Sam Guo/Box Sync/Data/2018_05_09_KindneySection/SM_2018_0509_1804_1' # Sample image folder path
#ImgBgPath = 'C:/Users/Sam Guo/Box Sync/Data/2018_05_09_KindneySection/BG_2018_0509_1801_1' # Background image folder path
#ImgSmPath = 'C:/Users/Sam Guo/Box Sync/Data/2018_06_20_Argolight/SM_2018_0620_1734_1' # Sample image folder path
#ImgBgPath = 'C:/Users/Sam Guo/Box Sync/Data/2018_06_20_Argolight/BG_2018_0620_1731_1' # Background image folder path
#ImgSmPath = 'C:/Users/Sam Guo/Box Sync/Data/20180710_brainSlice_Tomasz/SM_2018_0710_1830_1' # Sample image folder path
#ImgBgPath = 'C:/Users/Sam Guo/Box Sync/Data/20180710_brainSlice_Tomasz/BG_2018_0710_1820_1' # Background image folder path        
#ImgSmPath = 'C:/Users/Sam Guo/Box Sync/Data/2018_07_03_KidneyTissueSection/SMS_2018_0703_1835_1' # Sample image folder path
#ImgBgPath = 'C:/Users/Sam Guo/Box Sync/Data/2018_07_03_KidneyTissueSection/BG_2018_0703_1829_1' # Background image folder path                

RawDataPath = 'C:/Data'
#
ProcessedPath = 'C:/Processed'
#


ImgDir = '2018_09_28_U2OS_rainbow'
SmDir = 'SMS_2018_0928_1706_1_3'
BgDir = 'BG_2018_0928_1654_1'

#ImgDir = '20180914_GW20_CUBIC_DAPI'
#SmDir = 'SMS_2018_0914_1748_1'
#BgDir = 'BG_2018_0914_1732_1'

#ImgDir = '20180816_Sample_Test_CUBIC'
#SmDir = 'SM_2018_0801_1313_1'
#SmDir = 'SM_2018_0816_1838_1'
#BgDir = 'BG_2018_0816_1613_1'

#ImgDir = '2018_08_01_differentiation_Marius'
#SmDir = 'SM_2018_0801_1313_1'
#SmDir = 'BG_2018_0801_1333_1'
#BgDir = 'BG_2018_0801_1333_1'
#BgDir = 'BG_2018_0801_1214_1'

#ImgDir = 'NikonSmallWorld'
#SmDir = 'SMS_2018_0425_1654_1'
#BgDir = 'BG_2018_0425_1649_1'

#ImgDir = '20180710_brainSlice_Tomasz'
#SmDir = 'SM_2018_0710_1828_1'
#BgDir = 'BG_2018_0710_1820_1'

#ImgDir = '2018_08_02_Galina_test_condition'
#SmDir = 'BG_2018_0802_1605_1'
#SmDir = 'SM_2018_0802_1521_1'
#BgDir = 'BG_2018_0802_1508_1'


          

#ImgSmPath ='//flexo/MicroscopyData/AdvancedOpticalMicroscopy/SpinningDisk/RawData/PolScope/2018_05_09_KindneySection/SM_2018_0509_1804_1'
#ImgBgPath ='//flexo/MicroscopyData/AdvancedOpticalMicroscopy/SpinningDisk/RawData/PolScope/2018_05_09_KindneySection/BG_2018_0509_1801_1'

#OutputPath = '//flexo/MicroscopyData/AdvancedOpticalMicroscopy/SpinningDisk/Processed/PolScope/2018_07_03_KidneyTissueSection/SMS_2018_0703_1835_1'
outputChann = ['Transmission', 'Retardance', 'Orientation', 'Retardance+Orientation',
'Transmission+Retardance+Orientation']                            
                            
#outputChann = ['Transmission', 'Retardance', 'Orientation',                             
#                            '405','488','568']# channels to output, see readme for channel names
flipPol=True # flip the sign of polarization
bgCorrect='Auto' 
# Auto: correct the background using background from the metadata  
flatField=True
batchProc=False
if batchProc:
    ImgPath = os.path.join(RawDataPath, ImgDir)
    SmDirList = GetSubDirName(ImgPath)
    for SmDir in SmDirList:
#        if 'SM' in SmDir or 'BG' in SmDir :
        if 'SM' in SmDir:
            processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol)
else:
    processImg(RawDataPath, ProcessedPath, ImgDir, SmDir, BgDir, outputChann, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol)



