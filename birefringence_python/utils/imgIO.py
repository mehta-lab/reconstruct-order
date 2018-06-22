import os
import numpy as np
import glob
#import seaborn as sns
#import matplotlib.pyplot as plt
import re
import cv2
import sys
sys.path.append("..") # Adds higher directory to python modules path.
#from mpl_toolkits.axes_grid1 import make_axes_locatable

def GetSubDirName(ImgPath):
    subDirPath = glob.glob(os.path.join(ImgPath, '*/'))    
    subDirName = [os.path.split(subdir[:-1])[1] for subdir in subDirPath]            
    return subDirName

def loadTiff(acquDirPath, acquFiles):   
    TiffFile = os.path.join(acquDirPath, acquFiles)
    img = cv2.imread(TiffFile,-1) # flag -1 to perserve the bit dept of the raw image
    img = img.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    img = img.reshape(img.shape[0], img.shape[1],1)
    return img
        
def ParseTiffInput(acquDirPath): # Load the TIFF stack format output by the acquisition software created by Shalin (software name?)
    acquFiles = os.listdir(acquDirPath)    
    ImgRaw = np.array([])
    ImgProc = np.array([])
    ImgFluor = np.array([])
    for i in range(len(acquFiles)): # load raw images with Sigma0, 1, 2, 3 states, and processed images
#        matchObjRaw = re.match( r'img_000000000_State(\d+) - Acquired Image_000.tif', acquFiles[i], re.M|re.I) # read images with "state" string in the filename
        matchObjRaw = re.match( r'img_000000000_(State|PolAcquisition)(\d+)( - Acquired Image|)_000.tif', acquFiles[i], re.M|re.I) # read images with "state" string in the filename
        matchObjProc = re.match( r'img_000000000_(.*) - Computed Image_000.tif', acquFiles[i], re.M|re.I) # read computed images 
        matchObjFluor = re.match( r'img_000000000_Zyla_Confocal40_(.*).tif', acquFiles[i], re.M|re.I) # read computed images 
        if matchObjRaw:            
            img = loadTiff(acquDirPath, acquFiles[i])
            if ImgRaw.size:            
                ImgRaw = np.concatenate((ImgRaw, img), axis=2)
            else:
                ImgRaw = img
        elif matchObjProc:
            img = loadTiff(acquDirPath, acquFiles[i])
            if ImgProc.size:            
                ImgProc = np.concatenate((ImgProc, img), axis=2)
            else:
                ImgProc = img
        elif matchObjFluor:
            img = loadTiff(acquDirPath, acquFiles[i])
            if ImgFluor.size:            
                ImgFluor = np.concatenate((ImgFluor, img), axis=2)
            else:
                ImgFluor = img 
    
    retardMM = ImgProc[:,:,0]
    azimuthMM = ImgProc[:,:,1]  
    
    return ImgRaw, retardMM, azimuthMM, ImgFluor 

#def exportImg():
#    if zoomin:
#        imList = [IAbs,retard, azimuth]
#        imListCrop = imcrop(imList, IAbs)
#        IAbs,retard, azimuth = imListCrop
#    IHsv, IHv, IAbs= PolColor(IAbs*200, retard*1000, azimuth)
##    DAPI = cv2.convertScaleAbs(DAPI*20)
##    TdTomato = cv2.convertScaleAbs(TdTomato*2)
##    IFluorAbs = np.stack([DAPI+IAbs/2, IAbs/2, TdTomato+IAbs/2],axis=2)
#    
#    
#    R=retard*IAbs
#    R = R/np.nanmean(R) #normalization
#    R=vectorScl*R
#    #%%
#    figSize = (12,12)
#    fig = plt.figure(figsize = figSize)                                        
#    a=fig.add_subplot(2,2,1)
#    plt.tick_params(labelbottom='off',labelleft='off') # labels along the bottom edge are off          
#    plt.imshow(imadjust(IAbs), cmap='gray')
#    plt.title('Transmission')
#    plt.xticks([]),plt.yticks([])                                      
#    plt.show()
#    
#    a=fig.add_subplot(2,2,2)
#    plt.tick_params(labelbottom='off',labelleft='off') # labels along the bottom edge are off            
#    im = plt.imshow(IHv, cmap='hsv')
#    plt.title('Retardance+Orientation')
#    plt.xticks([]),plt.yticks([])
#    divider = make_axes_locatable(a)
#    cax = divider.append_axes('right', size='5%', pad=0.05)
#    cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=np.linspace(0,255, 5))    
#    cbar.ax.set_yticklabels([r'$0^o$', r'$45^o$', r'$90^o$', r'$135^o$', r'$180^o$'])  # vertically oriented colorbar                                     
#    plt.show()
#
#    a=fig.add_subplot(2,2,3)
#    plotVectorField(retard, azimuth, R=R, spacing=spacing)
##    plotVectorField(retard, azimuth, R=vectorScl, spacing=spacing)
#    plt.tick_params(labelbottom='off',labelleft='off') # labels along the bottom edge are off               
#    plt.title('Retardance+Orientation')   
#    plt.xticks([]),plt.yticks([])                                  
#    plt.show()
#    
#    a=fig.add_subplot(2,2,4)
#    plt.tick_params(labelbottom='off',labelleft='off') # labels along the bottom edge are off            
#    plt.imshow(IHsv)
#    plt.title('Transmission+Retardance\n+Orientation')  
#    plt.xticks([]),plt.yticks([])                                   
#    plt.show()
#    if zoomin:
#        figName = 'Transmission+Retardance+Orientation_Zoomin.png'
#    else:
#        figName = 'Transmission+Retardance+Orientation.png'
#        
#    plt.savefig(os.path.join(figPath, figName),dpi=dpi)
#    #%%
##    
#
##    images = [IAbs, retard*10**4, azimuth*10**4, IHv, IHsv, DAPI, TdTomato]
##    tiffNames = ['Transmission', 'Retardance', 'Orientation', 'Retardance+Orientation', 'Transmission+Retardance+Orientation', 'DAPI', 'TdTomato']
#
##    images = [retardAzi]
##    tiffNames = ['Retardance+Orientation_grey']
##    images = [IFluorAbs]
##    tiffNames = ['DAPI+Transmission+TdTomato']        
##    for im, tiffName in zip(images, tiffNames):
##        fileName = tiffName+'_%02d.tif'%ind
##        cv2.imwrite(os.path.join(figPath, fileName), im)
    