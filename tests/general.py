
# coding: utf-8

# """
# Reconstruct retardance and orientation maps from images taken with different polarized illumination output
# by Open compute. This script using the 4- or 5-frame reconstruction algorithm described in Michael Shribak and
# Rudolf Oldenbourg, 2003.
# 
# by Syuan-Ming Guo @ CZ Biohub 2018.3.30 
# """

# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
from compute.multiPos import findBackground, loopPos
import seaborn as sns


sns.set_context("poster")


# In[3]:


def processImg(ImgSmPath, ImgBgPath, Chi, flatField=False):
    Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg = findBackground(ImgSmPath, ImgBgPath, Chi,flatField=flatField) # find background tile
    loopPos(ImgSmPath, Chi, Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg,flatField=flatField)




# In[4]:


#%%
figPath = 'C:/Google Drive/Python/figures/'   # change 'fig_path' to the desired figure output path.
#ImgSmPath = 'C:/Google Drive/20180314_GreenbergLabBrainSlice/SM_2018_0314_1550_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/20180314_GreenbergLabBrainSlice/BG_2018_0314_1548_1' # Background image folder path
#ImgSmPath = 'C:/Google Drive/20180314_GreenbergLabBrainSlice/SM_2018_0314_1558_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/20180314_GreenbergLabBrainSlice/BG_2018_0314_1555_1' # Background image folder path
#ImgSmPath = 'C:/Google Drive/20180328_GreenbergLabBrainSlice/SM_2018_0328_1418_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/20180328_GreenbergLabBrainSlice/BG_2018_0328_1338_1' # Background image folder path
#ImgSmPath = 'C:/Google Drive/2018_04_02_Grinberg_Slice484_4x20x/SM_2018_0402_1325_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/2018_04_02_Grinberg_Slice484_4x20x/BG_2018_0402_1312_1' # Background image folder path
#ImgSmPath = 'C:/Google Drive/2018_04_02_Grinberg_Slice484_4x20x/SM_2018_0402_1256_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/2018_04_02_Grinberg_Slice484_4x20x/BG_2018_0402_1246_1' # Background image folder path
#ImgSmPath = 'C:/Google Drive/2018_04_04_UstainedTissue_4x/SM_2018_0404_1816_1' # Sample image folder path
#ImgSmPath = 'C:/Google Drive/2018_04_04_UstainedTissue_4x/SM_2018_0404_1817_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/2018_04_04_UstainedTissue_4x/BG_2018_0404_1811_1' # Background image folder path
#ImgSmPath = 'C:/Google Drive/2018_04_12_U2OSCells63x1.2/SM_2018_0412_1731_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/2018_04_12_U2OSCells63x1.2/BG_2018_0412_1726_1' # Background image folder path
#ImgSmPath = 'C:/Google Drive/2018_04_12_U2OSCells63x1.2/SMS_2018_0412_1735_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/2018_04_12_U2OSCells63x1.2/BG_2018_0412_1726_1' # Background image folder path
#ImgSmPath = 'C:/Google Drive/2018_04_18_mus_anterior_1/SM_2018_0418_1753_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/2018_04_18_mus_anterior_1/BG_2018_0418_1751_1' # Background image folder path
#ImgSmPath = 'C:/Google Drive/2018_04_16_unstained_brain_slice/SMS_2018_0416_1825_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/2018_04_16_unstained_brain_slice/BG_2018_0416_1745_1' # Background image folder path
#ImgSmPath = 'C:/Google Drive/NikonSmallWorld/someone/2018_04_25_Testing/SMS_2018_0425_1654_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/NikonSmallWorld/someone/2018_04_25_Testing/BG_2018_0425_1649_1' # Background image folder path
ImgSmPath = 'C:/Google Drive/20180530_gutCells/SM_2018_0530_1623_1' # Sample image folder path
ImgBgPath = 'C:/Google Drive/20180530_gutCells/BG_2018_0530_1621_1' # Background image folder path
#ImgSmPath = 'C:/Google Drive/20180530_gutCells/SM_2018_0530_1812_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/20180530_gutCells/BG_2018_0530_1707_1' # Background image folder path
#ImgSmPath = 'C:/Google Drive/2018_05_09_KindneySection/SM_2018_0509_1804_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/2018_05_09_KindneySection/BG_2018_0509_1801_1' # Background image folder path
Chi = 0.05 # Swing
#pixsize = 0.624414 # (um) calibrated pixel size of confocal Zyla at 10X
#pixsize = 0.624414 # (um) calibrated pixel size of confocal Zyla at 10X 
#spacing  = 10 # spacing for vector field map
#%%
processImg(ImgSmPath, ImgBgPath, Chi, flatField=False)

