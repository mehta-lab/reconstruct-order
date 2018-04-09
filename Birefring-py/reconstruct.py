"""
Reconstruct retardance and orientation maps from polaried brightfield images acquired
by PolScope using the 4-frame reconstruction algorithm from Michael Shribak and 
Rudolf Oldenbourg, 2003.
Note: 
by Syuan-Ming Guo @ CZ Biohub 2018.3.30 
"""
import os
import numpy as np
import glob
import seaborn as sns
#import seaborn as sns
import matplotlib.pyplot as plt
import re
import cv2
import bisect
import colorsys
from mpl_toolkits.axes_grid1 import make_axes_locatable
sns.set_context("poster")
plt.close("all") # close all the figures from the last run
#%%
def loadTiffStk(ImgPath): # Load the TIFF stack format output by the acquisition software created by Shalin (software name?)
    subDirPath = glob.glob(os.path.join(ImgPath, '*/'))    
    subDirName = [os.path.split(subdir[:-1])[1] for subdir in subDirPath]            
    nDir = len(subDirName)
    acquDirPath = os.path.join(ImgPath, subDirName[0]) # only load the first acquisition for now    
    acquFiles = os.listdir(acquDirPath)    
    ImgRaw = np.array([])
    ImgProc = np.array([])
    for i in range(len(acquFiles)): # load raw images with Sigma0, 1, 2, 3 states, and processed images
        matchObjRaw = re.match( r'img_000000000_State(\d+) - Acquired Image_000.tif', acquFiles[i], re.M|re.I) # read images with "state" string in the filename 
        matchObjProc = re.match( r'img_000000000_(.*) - Computed Image_000.tif', acquFiles[i], re.M|re.I) # read computed images 
        if matchObjRaw:
            TiffFile = os.path.join(acquDirPath, acquFiles[i])
            img = cv2.imread(TiffFile,-1) # flag -1 to perserve the bit dept of the raw image
            img = img.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
            img = img.reshape(img.shape[0], img.shape[1],1)
            if ImgRaw.size:            
                ImgRaw = np.concatenate((ImgRaw, img), axis=2)
            else:
                ImgRaw = img
        elif matchObjProc:
            TiffFile = os.path.join(acquDirPath, acquFiles[i])
            img = cv2.imread(TiffFile,-1) # flag -1 to perserve the bit dept of the raw image
            img = img.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
            img = img.reshape(img.shape[0], img.shape[1],1)
            if ImgProc.size:            
                ImgProc = np.concatenate((ImgProc, img), axis=2)
            else:
                ImgProc = img
    Iextiction = ImgRaw[:,:,0] # Sigma0 in Fig.2
    IalphaPlusChi = ImgRaw[:,:,1] # Sigma2 in Fig.2
    IbetaPlusChi = ImgRaw[:,:,2] # Sigma4 in Fig.2
    IbetaMinusChi = ImgRaw[:,:,3] # Sigma3 in Fig.2
    retardMM = ImgProc[:,:,0]
    azimuthMM = ImgProc[:,:,1]    
    
    return Iextiction, IalphaPlusChi, IbetaPlusChi, IbetaMinusChi , retardMM, azimuthMM          
#%%               
def computeAB(Iextiction, IalphaPlusChi, IbetaPlusChi, IbetaMinusChi,Chi): # output numerators and denominators of A, B along with to retain the phase info   
    nB = (IbetaPlusChi-IbetaMinusChi)*np.tan(Chi/2)
    nA = (IbetaMinusChi+IbetaPlusChi -2*IalphaPlusChi)*np.tan(Chi/2)   # Eq. 10 in reference
    dAB = IbetaMinusChi+IbetaPlusChi-2*Iextiction
    A = nA/dAB
    B = nB/dAB   
    IAbs = IbetaMinusChi+IbetaPlusChi-2*np.cos(Chi)*Iextiction
    return  A, B, IAbs

def computeDeltaPhi(A,B):
    retardPos = np.arctan(np.sqrt(A**2+B**2))
    retardNeg = np.pi + np.arctan(np.sqrt(A**2+B**2)) # different from Eq. 10 due to the definition of arctan in numpy
    retard = retardNeg*(retardPos<0)+retardPos*(retardPos>=0)        
#    azimuth = 0.5*((np.arctan2(A,B)+2*np.pi)%(2*np.pi)) # make azimuth fall in [0,pi]    
    azimuth = (0.5*np.arctan2(A,B)+0.5*np.pi) # make azimuth fall in [0,pi]    
    return retard, azimuth
#%%
def plotVectorField(I, azimuth, R=40, spacing=40): # plot vector field representaiton of the orientation map,
    # Currently only plot single pixel value when spacing >0. 
    # To do: Use average pixel value to reduce noise
    azimuthSmooth = nanRobustBlur(azimuth,(spacing,spacing)) # plot smoothed vector field
#    azimuthSmooth  = azimuth
    nY, nX = I.shape
    Y, X = np.mgrid[0:nY, 0:nX] # notice the inversed order of X and Y    
    U, V = R * np.cos(azimuthSmooth), R * np.sin(azimuthSmooth)
    I = imadjust(I,tol=0.1)
    I = histequal(I)
#    figSize = (10,10)
#    fig = plt.figure(figsize = figSize) 
    plt.imshow(I, cmap='gray')
    plt.title('Orientation map')                              
    plt.quiver(X[::spacing, ::spacing], Y[::spacing,::spacing], 
               U[::spacing,::spacing], V[::spacing,::spacing],
               edgecolor='r',facecolor='r',units='xy', alpha=1,
               headwidth = 0, headlength = 0, headaxislength = 0,
               scale_units = 'xy',scale = 1 )  
#    plt.xticks(())
#    plt.yticks(())
    
    plt.show()
    
def PolColor(IAbs, retard, azimuth):
    retard = imadjust(retard,bit = 16)
    retard = imadjust(retard,bit = 8)
    retard = histequal(retard)
    IAbs = imadjust(IAbs,bit = 8)
    azimuth = azimuth/np.pi*180
    azimuth = azimuth.astype(np.uint8, copy=False)
#    retard = retard.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
#    IAbs = IAbs.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    IHsv = np.stack([azimuth, retard, IAbs],axis=2)
    IHv = np.stack([azimuth, np.ones(retard.shape).astype(np.uint8)*255,retard],axis=2)

    IHsv = cv2.cvtColor(IHsv, cv2.COLOR_HSV2RGB)    
    IHv = cv2.cvtColor(IHv, cv2.COLOR_HSV2RGB)
#    IHv = (IHv.astype(np.float32)/255)**(1/2.2)
#    IHv = cv2.normalize(IHv, None, 0.0, 1.0, cv2.NORM_MINMAX)
#    IHv = (IHv*255).astype(np.uint8)
#    IHv = IHv[:,:, [1,0,2]]
#    
    return IHsv,IHv 
    
#%%
def nanRobustBlur(I, dim):
    V=I.copy()
    V[I!=I]=0
    VV=cv2.blur(V,dim)
    
    W=0*I.copy()+1
    W[I!=I]=0
    WW=cv2.blur(W,dim)    
    Z=VV/WW
    return Z  
  
def histequal(ImgSm0): # histogram eaqualiztion for contrast enhancement
    ImgSm0 = ImgSm0/ImgSm0.max()*255 # rescale to 8 bit as OpenCV only takes 8 bit (REALLY????)
    ImgSm0 = ImgSm0.astype(np.uint8, copy=False) # convert to 8 bit
#    ImgAd = cv2.equalizeHist(ImgSm0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20)) # Contrast Limited Adaptive Histogram Equalization
    ImgAd = clahe.apply(ImgSm0)
    return ImgAd

def imadjust(src, tol=1, bit=16,vin=[0,2**16-1]):
    # Python implementation of "imadjust" from MATLAB for stretching intensity histogram. Slow
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img
    
    vout=(0,2**bit-1)
    assert len(src.shape) == 2 ,'Input image should be 2-dims'
    
    src = src/np.nanmax(src[:])*2**bit-1 # rescale to 8 bit as OpenCV only takes 8 bit (REALLY????)
    
        
#    if bit==8:
#        src = src.astype(np.uint8, copy=False) # convert to 8 bit
#    else:
#        src = src.astype(np.uint16, copy=False) # convert to 8 bit

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src,bins=list(range(2**bit)),range=(0,2**bit-1))[0]

        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, 2**bit-1): cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    dst = vd
    if bit==8:
        dst = dst.astype(np.uint8, copy=False) # convert to 8 bit
    else:
        dst = dst.astype(np.uint16, copy=False) # convert to 8 bit

    return dst

def removeBubbles(I, kernelSize = (11,11)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  kernelSize)
    Bg = cv2.morphologyEx(I, cv2.MORPH_OPEN, kernel)
    I8bit = I/np.nanmax(I[:])*255 # rescale to 8 bit as OpenCV only takes 8 bit (REALLY????)
    I8bit = I8bit.astype(np.uint8, copy=False) # convert to 8 bit
    ITh = cv2.adaptiveThreshold(I8bit,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,201,-1)
    kernelSize = (3,3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  kernelSize)
    IThBig = cv2.morphologyEx(ITh, cv2.MORPH_CLOSE, kernel)
    kernelSize = (21,21)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  kernelSize)
    IThBig = cv2.morphologyEx(IThBig, cv2.MORPH_OPEN, kernel)
    ITh=ITh-IThBig
    IBi = ITh.astype(np.bool_, copy=True) # convert to 8 bit
    INoBub = np.copy(I)
    INoBub[IBi] = Bg[IBi]
    figSize = (8,8)
    fig = plt.figure(figsize = figSize)                                        
    a=fig.add_subplot(2,2,1)
    plt.tick_params(labelbottom='off',labelleft='off') # labels along the bottom edge are off          
    plt.imshow(imadjust(I), cmap='gray')
    plt.title('Retardance (MM)')                                      
    plt.show()
    
    a=fig.add_subplot(2,2,2)
    plt.tick_params(labelbottom='off',labelleft='off') # labels along the bottom edge are off            
    plt.imshow(IThBig, cmap='gray')
    plt.title('Orientation (MM)')                                     
    plt.show()

    a=fig.add_subplot(2,2,3)
    plt.tick_params(labelbottom='off',labelleft='off') # labels along the bottom edge are off            
    plt.imshow(ITh, cmap='gray')
    plt.title('Retardance (Py)')                                     
    plt.show()
    
    a=fig.add_subplot(2,2,4)
    plt.tick_params(labelbottom='off',labelleft='off') # labels along the bottom edge are off            
    plt.imshow(imadjust(INoBub), cmap='gray')
    plt.title('Orientation (Py)')                                     
    plt.show()
    
    return INoBub    
    
#%%
def plot_sub_images(images,titles):       
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(imadjust(images[i]),'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
        plt.show()
#        plt.axis('tight') 

def plot_birefringence(IAbs,retard, retardSmooth, azimuth, IHsv, IHv, spacing=20): 
    
    figSize = (12,12)
    fig = plt.figure(figsize = figSize)                                        
    a=fig.add_subplot(2,2,1)
    plt.tick_params(labelbottom='off',labelleft='off') # labels along the bottom edge are off          
    plt.imshow(imadjust(IAbs), cmap='gray')
    plt.title('Absorption')
    plt.xticks([]),plt.yticks([])                                      
    plt.show()
    
    a=fig.add_subplot(2,2,2)
    plt.tick_params(labelbottom='off',labelleft='off') # labels along the bottom edge are off            
    im = plt.imshow(IHv, cmap='hsv')
    plt.title('Retardance+Orientation')
    plt.xticks([]),plt.yticks([])
    divider = make_axes_locatable(a)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=np.linspace(0,255, 5))    
    cbar.ax.set_yticklabels([r'$0^o$', r'$45^o$', r'$90^o$', r'$135^o$', r'$180^o$'])  # vertically oriented colorbar                                     
    plt.show()

    a=fig.add_subplot(2,2,3)
    plotVectorField(retard, azimuth, R=0.7*spacing*retardSmooth/np.nanmean(retardSmooth), spacing=spacing)
    plt.tick_params(labelbottom='off',labelleft='off') # labels along the bottom edge are off               
    plt.title('Retardance+Orientation')   
    plt.xticks([]),plt.yticks([])                                  
    plt.show()
    
    a=fig.add_subplot(2,2,4)
    plt.tick_params(labelbottom='off',labelleft='off') # labels along the bottom edge are off            
    plt.imshow(IHsv)
    plt.title('Absorption+Retardance\n+Orientation')  
    plt.xticks([]),plt.yticks([])                                   
    plt.show()
     
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
ImgSmPath = 'C:/Google Drive/2018_04_02_Grinberg_Slice484_4x20x/SM_2018_0402_1256_1' # Sample image folder path
ImgBgPath = 'C:/Google Drive/2018_04_02_Grinberg_Slice484_4x20x/BG_2018_0402_1246_1' # Background image folder path
Chi = 0.1 # Swing
spacing  = 40 # spacing for vector field map
IextictionSm, IalphaPlusChiSm, IbetaPlusChiSm, IbetaMinusChiSm, retardMMSm, azimuthMMSm = loadTiffStk(ImgSmPath)
IextictionBg, IalphaPlusChiBg, IbetaPlusChiBg, IbetaMinusChiBg , retardMMBg, azimuthMMBg = loadTiffStk(ImgBgPath)
Asm, Bsm, IAbsSm = computeAB(IextictionSm, IalphaPlusChiSm, IbetaPlusChiSm, IbetaMinusChiSm,Chi)
Abg, Bbg, IAbsBg = computeAB(IextictionBg, IalphaPlusChiBg, IbetaPlusChiBg, IbetaMinusChiBg,Chi) 
A = Asm-Abg # correct contributions from background
B = Bsm-Bbg
retard, azimuth = computeDeltaPhi(A,B)
retard = removeBubbles(retard)
retardSmooth = nanRobustBlur(retard, (spacing, spacing))
retardBg, azimuthBg = computeDeltaPhi(Abg, Bbg)
IextictionSm = np.squeeze(IextictionSm)
IHsv, IHv = PolColor(IAbsSm, retard, azimuth)
titles = ['Retardance (MM)','Orientation (MM)','Retardance (Py)','Orientation (Py)']
images = [retardMMSm, azimuthMMSm,retard, azimuth]
plot_sub_images(images,titles)
plt.savefig(os.path.join(ImgSmPath,'fourFrameSm.png'),dpi=300)
#plot_sub_images(IextictionSm, retardBg, azimuthBg, retardMMBg, azimuthMMBg)
#plt.savefig(os.path.join(ImgSmPath,'fourFrameBg.png'),dpi=300)
plotVectorField(retard, azimuth, R=0.7*spacing*retardSmooth/np.nanmean(retardSmooth), spacing=spacing)
plt.savefig(os.path.join(ImgSmPath,'fourFrameSmVF.png'),dpi=300)

plot_birefringence(IAbsSm,retard, retardSmooth, azimuth, IHsv, IHv, spacing)
plt.savefig(os.path.join(ImgSmPath,'fourFrame.png'),dpi=150)
#plotVectorField(azimuthBg, azimuthBg, R=spacing*retardBg/np.nanmean(retardBg), spacing=40)
#plt.savefig(os.path.join(ImgSmPath,'fourFrameBgVF.png'),dpi=300)