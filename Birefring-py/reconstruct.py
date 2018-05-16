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
import matplotlib.pyplot as plt
import re
import cv2
import bisect
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets  import RectangleSelector
import warnings
sns.set_context("poster")
plt.close("all") # close all the figures from the last run
#%%
def processImg(ImgSmPath, ImgBgPath, Chi):
    Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg = findBackground(ImgSmPath, ImgBgPath, Chi) # find background tile
    loopPos(ImgSmPath, Chi, Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg)
    
def findBackground(ImgSmPath, ImgBgPath, Chi):
    subDirName = GetSubDirName(ImgBgPath)               
    acquDirPath = os.path.join(ImgBgPath, subDirName[0]) # only load the first acquisition for now        
    IextBg, I90Bg, I135Bg, I45Bg , retardMMBg, azimuthMMBg, ImgFluor = loadTiffStk(acquDirPath)
    Abg, Bbg, IAbsBg = computeAB(IextBg, I90Bg, I135Bg, I45Bg,Chi)    
    subDirName = GetSubDirName(ImgSmPath)            
#    nDir = len(subDirName)   
    DAPIBg = np.inf
    TdTomatoBg = np.inf    
    for subDir in subDirName:
        plt.close("all") # close all the figures from the last run
        if re.match( r'(\d+)-Pos_(\d+)_(\d+)', subDir, re.M|re.I):
            acquDirPath = os.path.join(ImgSmPath, subDir) # only load the first acquisition for now            
            IextSm, I90Sm, I135Sm, I45Sm , retardMMSm, azimuthMMSm, ImgFluor = loadTiffStk(acquDirPath)
            if ImgFluor.size:            
                DAPI = ImgFluor[:,:,0] # Needs to be generalized in the future
                TdTomato = ImgFluor[:,:,1]  # Needs to be generalized in the future
                DAPIBg = ImgMin(DAPI, DAPIBg)
                TdTomatoBg = ImgMin(TdTomato, TdTomatoBg)
    return Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg        
        
def ImgMin(Img, ImgBg):    
    ImgArr = np.array([Img, ImgBg])
    ImgMeanArr = np.array([np.mean(Img), np.mean(ImgBg)])
    ImgBg = ImgArr[np.argmin(ImgMeanArr)]
    return ImgBg        
    
def loopPos(ImgSmPath, Chi, Abg, Bbg, IAbsBg, DAPIBg, TdTomatoBg): # loop through each position in the acquistion folder       
    subDirName = GetSubDirName(ImgSmPath)            
#    nDir = len(subDirName)
    ind = 0
    for subDir in subDirName:
        plt.close("all") # close all the figures from the last run
        acquDirPath = os.path.join(ImgSmPath, subDir) # only load the first acquisition for now  
        if re.match( r'(\d+)-Pos_(\d+)_(\d+)', subDir, re.M|re.I):          
            IextSm, I90Sm, I135Sm, I45Sm , retardMMSm, azimuthMMSm, ImgFluor = loadTiffStk(acquDirPath)    
            ASm, BSm, IAbsSm = computeAB(IextSm, I90Sm, I135Sm, I45Sm, Chi)        
            A, B = correctBackground(ASm,BSm,Abg,Bbg, IextSm, extra=False)
            IAbsSm = IAbsSm/IAbsBg
            retard, azimuth = computeDeltaPhi(A,B)        
            #retard = removeBubbles(retard)        
            retardBg, azimuthBg = computeDeltaPhi(Abg, Bbg)
            if ImgFluor.size:            
                DAPI = ImgFluor[:,:,0]/DAPIBg # Needs to be generalized in the future
                TdTomato = ImgFluor[:,:,1]/TdTomatoBg  # Needs to be generalized in the future                  
            titles = ['Retardance (MM)','Orientation (MM)','Retardance (Py)','Orientation (Py)']
            images = [retardMMSm, azimuthMMSm,retard, azimuth]
    #        plot_sub_images(images,titles)
    #        plt.savefig(os.path.join(acquDirPath,'compare_MM_Py.png'),dpi=200)    
            plot_birefringence(IAbsSm,retard, azimuth, os.path.join(ImgSmPath,'Raw'), ind, DAPI=DAPI,
                               TdTomato=TdTomato, spacing=20, vectorScl=2, zoomin=False, dpi=300)
            ind+=1
        
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
        
def loadTiffStk(acquDirPath): # Load the TIFF stack format output by the acquisition software created by Shalin (software name?)
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
    Iext = ImgRaw[:,:,0] # Sigma0 in Fig.2
    I90 = ImgRaw[:,:,1] # Sigma2 in Fig.2
    I135 = ImgRaw[:,:,2] # Sigma4 in Fig.2
    I45 = ImgRaw[:,:,3] # Sigma3 in Fig.2
    retardMM = ImgProc[:,:,0]
    azimuthMM = ImgProc[:,:,1]  
    
    return Iext, I90, I135, I45 , retardMM, azimuthMM, ImgFluor 
#%%               
def computeAB(Iext, I90, I135, I45,Chi): # output numerators and denominators of A, B along with to retain the phase info   
    nB = (I135-I45)*np.tan(Chi/2)
    nA = (I45+I135 -2*I90)*np.tan(Chi/2)   # Eq. 10 in reference
    dAB = I45+I135-2*Iext
    A = nA/dAB
    B = nB/dAB   
    IAbs = I45+I135-2*np.cos(Chi)*Iext
    return  A, B, IAbs

def correctBackground(ASm,BSm,Abg,Bbg,IextSm,extra=False):
    # for low birefringence sample that requires 0 background, set extra=True to manually offset the background 
    ASmBg = 0
    BSmBg = 0
    if extra: # extra background correction to set backgorund = 0
        imList = [ASm, BSm]
        imListCrop = imcrop(imList, IextSm) # manually select ROI with only background retardance    
        ASmCrop,BSmCrop = imListCrop 
        ASmBg = np.nanmean(ASmCrop)
        BSmBg = np.nanmean(BSmCrop)               
    A = ASm-Abg-ASmBg # correct contributions from background
    B = BSm-Bbg-BSmBg    
    return A, B        

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
    
    plt.show()    

def plot_birefringence(IAbs,retard, azimuth, figPath, ind, DAPI=[], TdTomato=[], spacing=20, vectorScl=1, zoomin=False, dpi=300): 
    
    if zoomin:
        imList = [IAbs,retard, azimuth]
        imListCrop = imcrop(imList, IAbs)
        IAbs,retard, azimuth = imListCrop
    IHsv, IHv, IAbs= PolColor(IAbs*200, retard*1000, azimuth)
    DAPI = cv2.convertScaleAbs(DAPI*20)
    TdTomato = cv2.convertScaleAbs(TdTomato*2)
    R=retard*IAbs
    R = R/np.nanmean(R) #normalization
    R=vectorScl*R
    #%%
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
    #%%
#    

    images = [IAbs, retard*10**4, azimuth*10**4, IHv, IHsv, DAPI, TdTomato]
    tiffNames = ['Transmission', 'Retardance', 'Orientation', 'Retardance+Orientation', 'Transmission+Retardance+Orientation', 'DAPI', 'TdTomato']

#    images = [retardAzi]
#    tiffNames = ['Retardance+Orientation_grey']    
    for im, tiffName in zip(images, tiffNames):
        fileName = tiffName+'_%02d.tif'%ind
        cv2.imwrite(os.path.join(figPath, fileName), im)

def PolColor(IAbs, retard, azimuth):
#    retard = imBitConvert(retard,bit = 8)
#    retard = imBitConvert(retard,bit = 8)
#    retard = histequal(retard)
#    IAbs = imBitConvert(IAbs,bit = 8)
    retard = cv2.convertScaleAbs(retard)
    IAbs = cv2.convertScaleAbs(IAbs)
#    retard = retard.astype(np.uint8, copy=False)
#    IAbs = IAbs.astype(np.uint8, copy=False)
    azimuth = azimuth/np.pi*180
    azimuth = azimuth.astype(np.uint8, copy=False)
    retardAzi = np.stack([azimuth, retard, np.ones(retard.shape).astype(np.uint8)*255],axis=2)
    IHsv = np.stack([azimuth, retard,IAbs],axis=2)
    IHv = np.stack([azimuth, np.ones(retard.shape).astype(np.uint8)*255,retard],axis=2)
    IHsv = cv2.cvtColor(IHsv, cv2.COLOR_HSV2RGB)    
    IHv = cv2.cvtColor(IHv, cv2.COLOR_HSV2RGB)    #
#    retardAzi = np.stack([azimuth, retard],axis=2)    
    return IHsv,IHv, IAbs
    
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

def imBitConvert(im,bit=16, norm=False):
    im = im.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    if norm:
        im = (im-np.nanmin(im[:]))/(np.nanmax(im[:])-np.nanmin(im[:]))*(2**bit-1) # rescale to 16 bit   
    else:
        scale = (2**bit-1)/10
        im = im*scale
    if bit==8:
        im = im.astype(np.uint8, copy=False) # convert to 8 bit
    else:
        im = im.astype(np.uint16, copy=False) # convert to 16 bit
    return im

def imadjustStack(imStk, tol=1, bit=16,vin=[0,2**16-1]):
    for i in range(imStk.shape[2]):
        imStk[:,:,i] = imadjust(imStk[:,:,i])
    return imStk    
#%%
def imadjust(src, tol=1, bit=16,vin=[0,2**16-1]):
    # Python implementation of "imadjust" from MATLAB for stretching intensity histogram. Slow
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # bit : bits of the I/O
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img
    bitTemp = 16 # temporary bit depth for calculation. Convert to 32bit for calculation to minimize the info loss
    vout=(0,2**bitTemp-1)
    assert len(src.shape) == 2 ,'Input image should be 2-dims'
    
    src = imBitConvert(src) # rescale to 16 bit       
    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src,bins=list(range(2**bitTemp)),range=(0,2**bitTemp-1))[0]

        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, 2**bitTemp-1): cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    if vin[1] == vin[0]:
        warnings.warn("Tolerance is too high. No contrast adjustment is perfomred")
        dst = src
        
    else:
        scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
        vs = src-vin[0]
        vs[src<vin[0]]=0
        vd = vs*scale+0.5 + vout[0]
        vd[vd>vout[1]] = vout[1]
        dst = vd
    dst = imBitConvert(dst,bit=bit)
    return dst
#%%
def removeBubbles(I, kernelSize = (11,11)): # remove bright spots (mostly bubbles) in retardance images. Need to add a size filter
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
    figSize = (12,12)
    fig = plt.figure(figsize = figSize)            
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(imadjust(images[i]),'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
        plt.show()
#        plt.axis('tight') 


#%%    
def imcrop(imList,imV): # interactively select an ROI in imV, crop the same ROI for each image in imList
    figSize = (8,8)
    fig = plt.figure(figsize = figSize) 
    ax = plt.subplot()
#    r = cv2.selectROI(imadjust(im),fromCenter)  
    ax.imshow(imadjust(imV),cmap='gray')
    
    mouse_click = True
    pts = [] 
    
    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                   drawtype='box', useblit=False, button=[1], 
                   minspanx=5, minspany=5, spancoords='pixels', 
                   interactive=True)
#    pts = np.asarray(plt.ginput(2, timeout=-1))
    plt.connect('key_press_event', toggle_selector)
    plt.show()
    plt.waitforbuttonpress()
    mouse_click =  plt.waitforbuttonpress()
    r= toggle_selector.RS.extents
    
    print(r)
    imListCrop = []
    # Crop image
    for im in imList:
        if len(im.shape)>2:
            imC =  im[int(r[2]):int(r[3]), int(r[0]):int(r[1]),:]
        else:
            imC =  im[int(r[2]):int(r[3]), int(r[0]):int(r[1])]
            
        imListCrop.append(imC)
    
    return imListCrop

def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)

def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print(' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata))
    print(' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata))
    print(' used button   : ', eclick.button)


#    rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
#    ax.add_patch(rect)

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
ImgSmPath = 'C:/Google Drive/NikonSmallWorld/someone/2018_04_25_Testing/SMS_2018_0425_1654_1' # Sample image folder path
ImgBgPath = 'C:/Google Drive/NikonSmallWorld/someone/2018_04_25_Testing/BG_2018_0425_1649_1' # Background image folder path

Chi = 0.1 # Swing
#pixsize = 0.624414 # (um) calibrated pixel size of confocal Zyla at 10X
#pixsize = 0.624414 # (um) calibrated pixel size of confocal Zyla at 10X 
#spacing  = 10 # spacing for vector field map
#%%
processImg(ImgSmPath, ImgBgPath, Chi)

