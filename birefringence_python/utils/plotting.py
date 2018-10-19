#%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.imgProcessing import nanRobustBlur,imadjust, imBitConvert, imClip
from utils.imgCrop import imcrop
#import sys
#sys.path.append("..") # Adds higher directory to python modules path.

#%%
def plotVectorField(I, azimuth, R=40, spacing=40, clim=[None, None]): # plot vector field representaiton of the orientation map,
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
 
#    I = histequal(I)
#    figSize = (10,10)
#    fig = plt.figure(figsize = figSize) 
    imAx = plt.imshow(I, cmap='gray', vmin=clim[0], vmax=clim[1])
    plt.title('Orientation map')                              
    plt.quiver(X[::spacing, ::spacing], Y[::spacing,::spacing], 
               USmooth[::spacing,::spacing], VSmooth[::spacing,::spacing],
               edgecolor='g',facecolor='g',units='xy', alpha=1, width=2,
               headwidth = 0, headlength = 0, headaxislength = 0,
               scale_units = 'xy',scale = 1 )  
    
#    plt.show()
    return imAx    

def plot_birefringence(imgInput, imgs, outputChann, spacing=20, vectorScl=1, zoomin=False, dpi=300, norm=True, plot=True):
    I_trans,retard, azimuth, polarization, ImgFluor = imgs
    scattering = 1-polarization
    tIdx = imgInput.tIdx 
    zIdx = imgInput.zIdx
    posIdx = imgInput.posIdx
    if zoomin: # crop the images
        imList = [I_trans, retard, azimuth]
        imListCrop = imcrop(imList, I_trans)
        I_trans,retard, azimuth = imListCrop

    azimuth_degree = azimuth/np.pi*180
    I_azi_ret_trans, I_azi_ret, I_azi_scat = PolColor(I_trans, retard, azimuth_degree, scattering, norm=norm)

    if plot:
        plot_recon_images(I_trans, retard, azimuth, scattering, I_azi_ret, I_azi_scat, zoomin=False, spacing=20, vectorScl=1, dpi=300)
        if zoomin:
            figName = 'Transmission+Retardance+Orientation_Zoomin.png'
        else:
            figName = 'Transmission+Retardance+Orientation_t%03d_p%03d_z%03d.png' % (tIdx, posIdx, zIdx)

        plt.savefig(os.path.join(imgInput.ImgOutPath, figName), dpi=dpi, bbox_inches='tight')

    IFluorRetard = CompositeImg([10*retard, ImgFluor[2,:,:]*0.05, ImgFluor[0,:,:]*0.2], norm=norm)
#    images = [I_trans, retard, azimuth_degree, I_azi_ret, I_azi_ret_trans, IFluorRetard]
    I_trans = imBitConvert(I_trans * 10 ** 3, bit=16, norm=norm)  # AU, set norm to False for tiling images
    retard = imBitConvert(retard * 10 ** 3, bit=16)  # scale to pm
    scattering = imBitConvert(scattering * 10 ** 4, bit=16)
    azimuth_degree = imBitConvert(azimuth_degree * 100, bit=16)  # scale to [0, 18000], 100*degree
    imagesTrans = [I_trans, retard, azimuth_degree, scattering, I_azi_ret, I_azi_scat, I_azi_ret_trans] #trasmission channels
    imagesFluor = [imBitConvert(ImgFluor[i,:,:]*5, bit=16, norm=norm) for i in range(ImgFluor.shape[0])]+[IFluorRetard]
    
    images = imagesTrans+imagesFluor   
    chNames = ['Transmission', 'Retardance', 'Orientation', 'Scattering',
                            'Retardance+Orientation', 'Scattering+Orientation',
               'Transmission+Retardance+Orientation',
                            '405','488','568','640', 'Retardance+Fluorescence']
    
    imgDict = dict(zip(chNames, images))
    imgInput.chNames = outputChann
    imgInput.nChann = len(outputChann)
    return imgInput, imgDict 
  

def PolColor(I_trans, retard, azimuth, scattering, norm=True):
    if norm:
        retard = imadjust(retard, tol=1, bit=8)
        I_trans = imadjust(I_trans, tol=1, bit=8)
        scattering = imadjust(scattering, tol=1, bit=8)
        # retard = cv2.convertScaleAbs(retard, alpha=(2**8-1)/np.max(retard))
        # I_trans = cv2.convertScaleAbs(I_trans, alpha=(2**8-1)/np.max(I_trans))
    else:
        retard = cv2.convertScaleAbs(retard, alpha=2)
        I_trans = cv2.convertScaleAbs(I_trans, alpha=200)
        scattering = cv2.convertScaleAbs(scattering, alpha=2000)
#    retard = histequal(retard)
    
    azimuth = cv2.convertScaleAbs(azimuth, alpha=1)
#    retardAzi = np.stack([azimuth, retard, np.ones(retard.shape).astype(np.uint8)*255],axis=2)
    I_azi_ret_trans = np.stack([azimuth, retard, I_trans], axis=2)
    I_azi_ret = np.stack([azimuth, np.ones(retard.shape).astype(np.uint8)*255, retard], axis=2)
    I_azi_scat = np.stack([azimuth, np.ones(retard.shape).astype(np.uint8) * 255, scattering], axis=2)
    I_azi_ret_trans = cv2.cvtColor(I_azi_ret_trans, cv2.COLOR_HSV2RGB)
    I_azi_ret = cv2.cvtColor(I_azi_ret, cv2.COLOR_HSV2RGB)
    I_azi_scat = cv2.cvtColor(I_azi_scat, cv2.COLOR_HSV2RGB)  #
#    retardAzi = np.stack([azimuth, retard],axis=2)    
    return I_azi_ret_trans, I_azi_ret, I_azi_scat

def CompositeImg(images, norm=True):
    assert len(images)==3,'CompositeImg currently only supports 3-channel image'
    ImgColor = []
    for img in images:
        if norm:
            img8bit = imadjust(img, tol=1, bit=8)
            # img8bit = cv2.convertScaleAbs(img, alpha=(2**8-1)/np.max(img))
        else:
            img8bit = cv2.convertScaleAbs(img, alpha=1)
        ImgColor +=[img8bit]
    ImgColor = np.stack(ImgColor,axis=2)
    return ImgColor
    
#%%
def plot_recon_images(I_trans, retard, azimuth, scattering, I_azi_ret, I_azi_scat, zoomin=False, spacing=20, vectorScl=1, dpi=300):

    R = retard
    R = R / np.nanmean(R)  # normalization
    R = vectorScl * R
    # %%
    figSize = (18, 12)
    fig = plt.figure(figsize=figSize)
    ax1 = plt.subplot(2, 3, 1)
    plt.tick_params(labelbottom=False, labelleft=False)  # labels along the bottom edge are off
    ax_trans = plt.imshow(imClip(I_trans, tol=1), cmap='gray')
    plt.title('Transmission')
    plt.xticks([]), plt.yticks([])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(ax_trans, cax=cax, orientation='vertical')
    #    plt.show()

    ax2 = plt.subplot(2, 3, 2)
    plt.tick_params(labelbottom=False, labelleft=False)  # labels along the bottom edge are off
    ax_retard = plt.imshow(imClip(retard, tol=5), cmap='gray')
    plt.title('Retardance(nm)')
    plt.xticks([]), plt.yticks([])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(ax_retard, cax=cax, orientation='vertical')

    ax3 = plt.subplot(2, 3, 3)
    plt.tick_params(labelbottom=False, labelleft=False)  # labels along the bottom edge are off
    ax_pol = plt.imshow(imClip(scattering, tol=1), cmap='gray')
    plt.title('Scattering')
    plt.xticks([]), plt.yticks([])
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(ax_pol, cax=cax, orientation='vertical')

    ax4 = plt.subplot(2, 3, 4)
    plt.tick_params(labelbottom=False, labelleft=False)  # labels along the bottom edge are off
    ax_hv = plt.imshow(imadjust(I_azi_ret, tol=5, bit=8), cmap='hsv')
    plt.title('Retardance+Orientation')
    plt.xticks([]), plt.yticks([])
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(ax_hv, cax=cax, orientation='vertical', ticks=np.linspace(0, 255, 5))
    cbar.ax.set_yticklabels([r'$0^o$', r'$45^o$', r'$90^o$', r'$135^o$',
                             r'$180^o$'])  # vertically oriented colorbar
    #    plt.show()

    ax5 = plt.subplot(2, 3, 5)
    imAx = plotVectorField(imClip(retard / 1000, tol=1), azimuth, R=R, spacing=spacing)
    plt.tick_params(labelbottom=False, labelleft=False)  # labels along the bottom edge are off
    plt.title('Retardance(nm)+Orientation')
    plt.xticks([]), plt.yticks([])

    ax6 = plt.subplot(2, 3, 6)
    plt.tick_params(labelbottom=False, labelleft=False)  # labels along the bottom edge are off
    ax_hsv = plt.imshow(imadjust(I_azi_scat, tol=5, bit=8), cmap='hsv')
    # plt.title('Transmission+Retardance\n+Orientation')
    plt.title('Scattering+Orientation')
    plt.xticks([]), plt.yticks([])
    plt.show()
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(ax_hsv, cax=cax, orientation='vertical', ticks=np.linspace(0, 255, 5))
    cbar.ax.set_yticklabels([r'$0^o$', r'$45^o$', r'$90^o$', r'$135^o$', r'$180^o$'])  # vertically oriented colorbar

def plot_sub_images(images,titles,imgInput):
    figSize = (12,12)
    figName = 'test'
    fig = plt.figure(figsize = figSize)            
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(imadjust(images[i]),'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    plt.savefig(os.path.join(imgInput.ImgOutPath, figName), dpi=300, bbox_inches='tight')



