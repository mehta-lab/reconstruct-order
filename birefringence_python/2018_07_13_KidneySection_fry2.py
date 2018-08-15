
# coding: utf-8

"""
Reconstruct retardance and orientation maps from images taken with different polarized illumination output
by Open PolScope. This script using the 4- or 5-frame reconstruction algorithm described in Michael Shribak and 
Rudolf Oldenbourg, 2003.
 
Output channels indices:
 0-'Transmission'
 1-'Retardance'
 2-'Orientation', 
 3-'Retardance+Orientation'
 4-'Transmission+Retardance+Orientation'
 5-'405'
 6-'488'
 7-'568'
 8-'640'
"""

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
from PolScope.multiDimProcess import findBackground, loopPos
import seaborn as sns
import os

sns.set_context("poster")


# In[2]:
def processImg(ImgSmPath, ImgBgPath, OutputPath, outputChannIdx, flatField=False, bgCorrect=True, flipPol=False):    
    imgSm = findBackground(ImgSmPath, ImgBgPath, OutputPath, outputChannIdx,flatField=flatField) # find background tile
    imgSm.loopZ ='sample'
    imgSm = loopPos(imgSm, outputChannIdx, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol)
        
            
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

RawDataPath = '/data/sguo/Data'
ProcessedPath = '/data/sguo/Processed'

ImgDir = '2018_07_03_KidneyTissueSection'
SmDir = 'SMS_2018_0703_1835_1'
BgDir = 'BG_2018_0703_1829_1'

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


ImgSmPath = os.path.join(RawDataPath, ImgDir, SmDir) # Sample image folder path, of form 'SM_yyyy_mmdd_hhmm_X'
ImgBgPath = os.path.join(RawDataPath, ImgDir, BgDir) # Background image folder path, of form 'BG_yyyy_mmdd_hhmm_X'          
outputChann = ['Transmission', 'Retardance', 'Orientation',                             
                            '405','488','568']# channels to output, see readme for channel names
flipPol=True # flip the sign of polarization
bgCorrect=True
flatField=True

if bgCorrect==True:
    OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir+'_'+BgDir)
else:
    OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir)
    
processImg(ImgSmPath, ImgBgPath, OutputPath, outputChann, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol)
