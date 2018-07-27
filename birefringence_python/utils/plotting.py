#%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.imgProcessing import nanRobustBlur,imadjust
from utils.imgCrop import imcrop
#import sys
#sys.path.append("..") # Adds higher directory to python modules path.

#%%
def plotVectorField(I, azimuth, R=40, spacing=40): # plot vector field representaiton of the orientation map,
    # Currently only plot single pixel value when spacing >0. 
    # To do: Use average pixel value to reduce noise
#    retardSmooth = nanRobustBlur(retard, (spacing, spacing))
#    retardSmooth/np.nanmean(retardSmooth)
#    R = R/np.nanmean(R)
    U, V = R*spacing*np.cos(2*azimuth), R*spacing*np.sin(2*azimuth)    
    USmooth = nanRobustBlur(U,(spacing,spacing)) # plot smoothed vector field
    VSmooth = nanRobustBlur(V,(spacing,spacing)) # plot smoothed vector field
    azimuthSmooth = 0.5*np.arctan2(VSmooth,USmooth)
    RSmooth = np.sqrt(USmooth**2+VSmooth**2)
    USmooth, VSmooth = RSmooth*np.cos(azimuthSmooth), RSmooth*np.sin(azimuthSmooth) 
   
#    azimuthSmooth  = azimuth
    nY, nX = I.shape
    Y, X = np.mgrid[0:nY, 0:nX] # notice the inversed order of X and Y    
    
    I = imadjust(I,tol=0.1)
#    I = histequal(I)
#    figSize = (10,10)
#    fig = plt.figure(figsize = figSize) 
    plt.imshow(I, cmap='gray')
    plt.title('Orientation map')                              
    plt.quiver(X[::spacing, ::spacing], Y[::spacing,::spacing], 
               USmooth[::spacing,::spacing], VSmooth[::spacing,::spacing],
               edgecolor='g',facecolor='g',units='xy', alpha=1, width=2,
               headwidth = 0, headlength = 0, headaxislength = 0,
               scale_units = 'xy',scale = 1 )  
    
#    plt.show()    

def plot_birefringence(IAbs,retard, azimuth, figPath, ind, z, DAPI=[], TdTomato=[], spacing=20, vectorScl=1, zoomin=False, dpi=300): 
    
    if zoomin:
        imList = [IAbs,retard, azimuth]
        imListCrop = imcrop(imList, IAbs)
        IAbs,retard, azimuth = imListCrop
    IHsv, IHv= PolColor(IAbs*200, retard*10**4, azimuth)
    IFluorRetard = CompositeImg([TdTomato*8, DAPI*8, retard*2*10**4])
#    DAPI = cv2.convertScaleAbs(DAPI*20)
#    TdTomato = cv2.convertScaleAbs(TdTomato*2)
#    IFluorAbs = np.stack([DAPI+IAbs/2, IAbs/2, TdTomato+IAbs/2],axis=2)    
    
    R=retard*IAbs
    R = R/np.nanmean(R) #normalization
    R=vectorScl*R
    #%%
    figSize = (12,12)
    fig = plt.figure(figsize = figSize)                                        
    plt.subplot(2,2,1)
    plt.tick_params(labelbottom=False,labelleft=False) # labels along the bottom edge are off          
    plt.imshow(imadjust(IAbs), cmap='gray')
    plt.title('Transmission')
    plt.xticks([]),plt.yticks([])                                      
#    plt.show()
    
    a = plt.subplot(2,2,2)
    plt.tick_params(labelbottom=False,labelleft=False) # labels along the bottom edge are off            
    im = plt.imshow(imadjust(IHv, bit=8), cmap='hsv')
    plt.title('Retardance+Orientation')
    plt.xticks([]),plt.yticks([])
    divider = make_axes_locatable(a)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=np.linspace(0,255, 5))    
    cbar.ax.set_yticklabels([r'$0^o$', r'$45^o$', r'$90^o$', r'$135^o$', r'$180^o$'])  # vertically oriented colorbar                                     
#    plt.show()

    plt.subplot(2,2,3)
    plotVectorField(imadjust(retard), azimuth, R=R, spacing=spacing)
#    plotVectorField(retard, azimuth, R=vectorScl, spacing=spacing)
    plt.tick_params(labelbottom=False,labelleft=False) # labels along the bottom edge are off               
    plt.title('Retardance+Orientation')   
    plt.xticks([]),plt.yticks([])                                  
#    plt.show()
    
    plt.subplot(2,2,4)
    plt.tick_params(labelbottom=False,labelleft=False) # labels along the bottom edge are off            
    plt.imshow(imadjust(IHsv, bit=8))
    plt.title('Transmission+Retardance\n+Orientation')  
    plt.xticks([]),plt.yticks([])                                   
    plt.show()
    if zoomin:
        figName = 'Transmission+Retardance+Orientation_Zoomin.png'
    else:
        figName = 'Transmission+Retardance+Orientation_%03d_%03d.png'%(ind,z)
        
    plt.savefig(os.path.join(figPath, figName),dpi=dpi)
    #%%   
#    images = [IAbs, retard*10**4, azimuth*10**4, IHv, IHsv, DAPI, TdTomato]
#    tiffNames = ['Transmission', 'Retardance', 'Orientation', 'Retardance+Orientation', 'Transmission+Retardance+Orientation', 'DAPI', 'TdTomato']
    images = [IAbs*100, retard*10**4, azimuth*10**4, IHv, IHsv, IFluorRetard]
   
#    images = [retardAzi]
#    tiffNames = ['Retardance+Orientation_grey']
#    images = [IFluorAbs]
#    tiffNames = ['DAPI+Transmission+TdTomato']        
    return images
  

def PolColor(IAbs, retard, azimuth):
#    retard = imBitConvert(retard,bit = 8)
#    retard = imBitConvert(retard,bit = 8)
#    retard = imadjust(retard)
#    IAbs = imadjust(IAbs)
   
#    IAbs = imBitConvert(IAbs,bit = 8)
#    retard = cv2.convertScaleAbs(retard, alpha=(2**8-1)/np.max(retard))
#    IAbs = cv2.convertScaleAbs(IAbs, alpha=(2**8-1)/np.max(IAbs))
    retard = cv2.convertScaleAbs(retard, alpha=1)
    IAbs = cv2.convertScaleAbs(IAbs, alpha=1)
#    retard = histequal(retard)
    azimuth = azimuth/np.pi*180
    azimuth = azimuth.astype(np.uint8, copy=False)
    retardAzi = np.stack([azimuth, retard, np.ones(retard.shape).astype(np.uint8)*255],axis=2)
    IHsv = np.stack([azimuth, retard,IAbs],axis=2)
    IHv = np.stack([azimuth, np.ones(retard.shape).astype(np.uint8)*255,retard],axis=2)
    IHsv = cv2.cvtColor(IHsv, cv2.COLOR_HSV2RGB)    
    IHv = cv2.cvtColor(IHv, cv2.COLOR_HSV2RGB)    #
#    retardAzi = np.stack([azimuth, retard],axis=2)    
    return IHsv,IHv

def CompositeImg(images):
    assert len(images)==3,'CompositeImg currently only supports 3-channel image'
    ImgColor = []
    for img in images:
        img8bit = cv2.convertScaleAbs(img, alpha=1)    
        ImgColor +=[img8bit]
    ImgColor = np.stack(ImgColor,axis=2)
    return ImgColor
    
#%%
def plot_sub_images(images,titles): 
    figSize = (12,12)
    fig = plt.figure(figsize = figSize)            
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(imadjust(images[i]),'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()



