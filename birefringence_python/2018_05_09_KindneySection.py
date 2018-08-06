
# coding: utf-8

# """
# Reconstruct retardance and orientation maps from images taken with different polarized illumination output
# by Open PolScope. This script using the 4- or 5-frame reconstruction algorithm described in Michael Shribak and 
# Rudolf Oldenbourg, 2003.
# 
# by Syuan-Ming Guo @ CZ Biohub 2018.3.30 
# """

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
from PolScope.multiPos import findBackground
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import re

from utils.imgIO import GetSubDirName, ParseTiffInput, ParseFileList, exportImg
from PolScope.reconstruct import computeAB, correctBackground, computeDeltaPhi
from utils.plotting import plot_birefringence, plot_sub_images
from utils.imgProcessing import ImgLimit

sns.set_context("poster")


# In[2]:

def processImg(ImgSmPath, ImgBgPath, OutputPath, Chi,Lambda, flatField=False, bgCorrect=True, flipPol=False):
    Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg = findBackground(ImgSmPath, ImgBgPath, Chi,flatField=flatField) # find background tile
    loopPos(ImgSmPath, OutputPath,Chi,Lambda, Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg,flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol)

def loopPos(ImgSmPath, OutputPath, Chi,Lambda, Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg, flatField=False, bgCorrect=True, flipPol=False): 
#loop through each position in the acquistion folder, perform flat-field correction       
    subDirName = GetSubDirName(ImgSmPath)          
    imgLimits = [[np.Inf,0]]*5
    ## TO DO: track global image limits
    if not os.path.exists(OutputPath): # create folder for processed images
        os.makedirs(OutputPath)
    ind=0
    for subDir in subDirName:
        plt.close("all") # close all the figures from the last run
        acquDirPath = os.path.join(ImgSmPath, subDir) # only load the first acquisition for now  
        if re.match( r'(\d?)-?Pos_?(\d+)_?(\d?)', subDir, re.M|re.I):                            
            PolChan, PolZ, FluorChan, FluorZ = ParseFileList(acquDirPath)
            loopZ(PolZ, ind, acquDirPath, OutputPath, Chi,Lambda, Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg,imgLimits, flatField=flatField, bgCorrect=bgCorrect, flipPol=flipPol)
            ind+=1
def loopZ(PolZ, ind, acquDirPath, OutputPath, Chi,Lambda, Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg,imgLimits, flatField=False, bgCorrect=True, flipPol=False):
    for z in PolZ:
        plt.close("all") # close all the figures from the last run
        DAPI = np.array([])
        TdTomato = np.array([])
        retardMMSm = np.array([])
        azimuthMMSm = np.array([])     
        ImgRawSm, ImgProcSm, ImgFluor, ImgBF = ParseTiffInput(acquDirPath, z)            
        ASm, BSm, IAbsSm = computeAB(ImgRawSm, Chi)
        if bgCorrect == True:        
            A, B = correctBackground(ASm,BSm,Abg,Bbg, ImgRawSm, extra=False) # background subtraction 
        else:
            A, B = ASm, BSm
        retard, azimuth = computeDeltaPhi(A,B,Lambda,flipPol=flipPol)        
        #retard = removeBubbles(retard)     # remove bright speckles in mounted brain slice images       
#        retardBg, azimuthBg = computeDeltaPhi(Abg, Bbg,flipPol=flipPol)
        if not ImgBF.size: # use brightfield calculated from pol-images if there is no brighfield data
            ImgBF = IAbsSm
        else:
            ImgBF = ImgBF[:,:,0]
            
        if ImgFluor.size:
            DAPI = ImgFluor[:,:,0]
            TdTomato = ImgFluor[:,:,1]
            DAPI = DAPI/DAPIBg # flat-field correction 
            TdTomato = TdTomato/TdTomatoBg   #flat-field correction 
        if ImgProcSm.size:
            retardMMSm =  ImgProcSm[:,:,0]
            azimuthMMSm = ImgProcSm[:,:,1]
        if flatField:
            ImgBF = ImgBF/IAbsBg #flat-field correction 
                    
            ## compare python v.s. Polacquisition output#####
#            titles = ['Retardance (MM)','Orientation (MM)','Retardance (Py)','Orientation (Py)']
#            images = [retardMMSm, azimuthMMSm,retard, azimuth]
#            plot_sub_images(images,titles)
#            plt.savefig(os.path.join(acquDirPath,'compare_MM_Py.png'),dpi=200)
            ##################################################################

        imgs = [ImgBF,retard, azimuth, DAPI, TdTomato]
        imgLimits = ImgLimit(imgs,imgLimits)
        
        imgs = plot_birefringence(imgs, OutputPath, ind, z,
                                  spacing=20, vectorScl=1, zoomin=False, dpi=150)
        tiffNames = ['Transmission', 'Retardance', 'Orientation', 'Retardance+Orientation', 'Transmission+Retardance+Orientation', 'Fluor+Retardance']
        exportImg(imgs, tiffNames, ind, z, OutputPath)
            


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
RawDataPath = 'C:/Users/Sam Guo/Box Sync/Data'
ProcessedPath = 'C:/Users/Sam Guo/Box Sync/Processed'

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

ImgSmPath = os.path.join(RawDataPath, ImgDir, SmDir) # Sample image folder path
ImgBgPath = os.path.join(RawDataPath, ImgDir, BgDir) # Background image folder path          
#ImgSmPath ='//flexo/MicroscopyData/AdvancedOpticalMicroscopy/SpinningDisk/RawData/PolScope/2018_05_09_KindneySection/SM_2018_0509_1804_1'
#ImgBgPath ='//flexo/MicroscopyData/AdvancedOpticalMicroscopy/SpinningDisk/RawData/PolScope/2018_05_09_KindneySection/BG_2018_0509_1801_1'

#OutputPath = '//flexo/MicroscopyData/AdvancedOpticalMicroscopy/SpinningDisk/Processed/PolScope/2018_07_03_KidneyTissueSection/SMS_2018_0703_1835_1'
Chi = 0.05 # Swing
#Chi = 0.25 # Swing
Lambda = 532 # Wavelength (nm)
flipPol=True # flip the sign of polarization
bgCorrect=True

if bgCorrect==True:
    OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir+'_'+BgDir)
else:
    OutputPath = os.path.join(ProcessedPath, ImgDir, SmDir)
    
processImg(ImgSmPath, ImgBgPath, OutputPath, Chi,Lambda, flatField=True , bgCorrect=bgCorrect, flipPol=flipPol)

