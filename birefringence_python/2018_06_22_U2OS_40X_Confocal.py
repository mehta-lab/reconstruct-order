
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

sns.set_context("poster")


# In[2]:

def processImg(ImgSmPath, ImgBgPath, Chi, flatField=False):
    Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg = findBackground(ImgSmPath, ImgBgPath, Chi,flatField=flatField) # find background tile
    loopPos(ImgSmPath, Chi, Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg,flatField=flatField)

def loopPos(ImgSmPath, Chi, Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg, flatField=False): 
#loop through each position in the acquistion folder, perform flat-field correction       
    subDirName = GetSubDirName(ImgSmPath)          
    figPath = os.path.join(ImgSmPath, 'processed')
    if not os.path.exists(figPath): # create folder for processed images
        os.makedirs(figPath)
    ind=0
    for subDir in subDirName:
        plt.close("all") # close all the figures from the last run
        acquDirPath = os.path.join(ImgSmPath, subDir) # only load the first acquisition for now  
        if re.match( r'(\d?)-?Pos_?(\d+)_?(\d?)', subDir, re.M|re.I):                            
            PolChan, PolZ, FluorChan, FluorZ = ParseFileList(acquDirPath)
            loopZ(PolZ, ind, acquDirPath, figPath, Chi, Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg,flatField=flatField)
            ind+=1
def loopZ(PolZ, ind, acquDirPath, figPath, Chi, Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg, flatField=False):
    for z in PolZ:
        plt.close("all") # close all the figures from the last run
        DAPI = np.array([])
        TdTomato = np.array([])
        retardMMSm = np.array([])
        azimuthMMSm = np.array([])     
        ImgRawSm, ImgProcSm, ImgFluor = ParseTiffInput(acquDirPath, z)            
        ASm, BSm, IAbsSm = computeAB(ImgRawSm, Chi)        
        A, B = correctBackground(ASm,BSm,Abg,Bbg, ImgRawSm, extra=False)            
        retard, azimuth = computeDeltaPhi(A,B)        
        #retard = removeBubbles(retard)     # remove bright speckles in mounted brain slice images       
        retardBg, azimuthBg = computeDeltaPhi(Abg, Bbg)
        if ImgFluor.size:
            DAPI = ImgFluor[:,:,0]
            TdTomato = ImgFluor[:,:,1]
        if ImgProcSm.size:
            retardMMSm =  ImgProcSm[:,:,0]
            azimuthMMSm = ImgProcSm[:,:,1]
        if flatField:
            IAbsSm = IAbsSm/IAbsBg #flat-field correction 
            DAPI = DAPI/DAPIBg # #flat-field correction 
            TdTomato = TdTomato/TdTomatoBg  # #flat-field correction         
            ## compare python v.s. Polacquisition output#####
#            titles = ['Retardance (MM)','Orientation (MM)','Retardance (Py)','Orientation (Py)']
#            images = [retardMMSm, azimuthMMSm,retard, azimuth]
#            plot_sub_images(images,titles)
#            plt.savefig(os.path.join(acquDirPath,'compare_MM_Py.png'),dpi=200)
            ##################################################################

        images = plot_birefringence(IAbsSm,retard, azimuth, figPath, ind, z, DAPI=DAPI,
                           TdTomato=TdTomato, spacing=20, vectorScl=1, zoomin=False, dpi=150)
        tiffNames = ['Transmission', 'Retardance', 'Orientation', 'Retardance+Orientation', 'Transmission+Retardance+Orientation']
        exportImg(images, tiffNames, ind, z, figPath)
            


# In[3]:


#ImgSmPath = 'C:/Google Drive/2018_04_12_U2OSCells63x1.2/SMS_2018_0412_1735_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/2018_04_12_U2OSCells63x1.2/BG_2018_0412_1726_1' # Background image folder path
#ImgSmPath = 'C:/Users/Sam Guo/Box Sync/Data/2018_04_12_U2OSCells63x1.2/SM_2018_0412_1731_1' # Sample image folder path
#ImgBgPath = 'C:/Users/Sam Guo/Box Sync/Data/2018_04_12_U2OSCells63x1.2/BG_2018_0412_1726_1' # Background image folder path        
#ImgSmPath = 'C:/Google Drive/2018_06_22_U2OS/SM_2018_0622_1545_2/' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/2018_06_22_U2OS/BG_2018_0622_1529_1/' # Background image folder path        
#ImgSmPath = 'C:/Google Drive/2018_04_16_unstained_brain_slice/SMS_2018_0416_1825_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/2018_04_16_unstained_brain_slice/BG_2018_0416_1745_1' # Background image folder path
#ImgSmPath = 'C:/Google Drive/NikonSmallWorld/someone/2018_04_25_Testing/SMS_2018_0425_1654_1' # Sample image folder path
#ImgBgPath = 'C:/Google Drive/NikonSmallWorld/someone/2018_04_25_Testing/BG_2018_0425_1649_1' # Background image folder path
ImgSmPath = 'C:/Users/Sam Guo/Box Sync/Data/2018_05_09_KindneySection/SM_2018_0509_1804_1' # Sample image folder path
ImgBgPath = 'C:/Users/Sam Guo/Box Sync/Data/2018_05_09_KindneySection/BG_2018_0509_1801_1' # Background image folder path
Chi = 0.25 # Swing
#Chi = 0.1 # Swing
processImg(ImgSmPath, ImgBgPath, Chi, flatField=False)

