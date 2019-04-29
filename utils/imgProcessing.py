import numpy as np
import matplotlib.pyplot as plt
import cv2
import bisect
import warnings
import sys
from workflow.multiDimProcess import loopPos
sys.path.append("..") # Adds higher directory to python modules path.
#sns.set_context("poster")
plt.close("all") # close all the figures from the last run

#%%
def ImgMin(Img, ImgBg):
    """Given 2 arrays, return the array with smaller mean value

    Parameters
    ----------
    Img
    ImgBg

    Returns
    -------

    """
    ImgArr = np.array([Img, ImgBg])
    ImgMeanArr = np.array([np.mean(Img), np.mean(ImgBg)])
    ImgBg = ImgArr[np.argmin(ImgMeanArr)]
    return ImgBg

def ImgLimit(imgs,imgLimits): # tracking the global image limit 
    imgLimitsNew = []
    for img, imgLimit in zip(imgs,imgLimits):
        if img.size:
            limit = [np.nanmin(img[:]), np.nanmax(img[:])]
            imgLimitNew = [np.minimum(limit[0], imgLimit[0]), np.maximum(limit[1], imgLimit[1])]
        else:
            imgLimitNew = imgLimit
        imgLimitsNew += [imgLimitNew]
    return imgLimitsNew               

def nanRobustBlur(I, dim):
    """Blur image with mean filter that is robust to NaN in the image

    Parameters
    ----------
    I : array
        image to blur
    dim: tuple
        size of the filter (n, n)

    Returns
    Z : array
        filtered image
    -------

    """
    V=I.copy()
    V[I!=I]=0
    VV=cv2.blur(V,dim)    
    W=0*I.copy()+1
    W[I!=I]=0
    WW=cv2.blur(W,dim)    
    Z=VV/WW
    return Z  
  
def histequal(ImgSm0):
    """histogram eaqualiztion for contrast enhancement

    Parameters
    ----------
    ImgSm0

    Returns
    -------

    """
    ImgSm0 = ImgSm0/ImgSm0.max()*255 # rescale to 8 bit as OpenCV only takes 8 bit (REALLY????)
    ImgSm0 = ImgSm0.astype(np.uint8, copy=False) # convert to 8 bit
#    ImgAd = cv2.equalizeHist(ImgSm0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20)) # Contrast Limited Adaptive Histogram Equalization
    ImgAd = clahe.apply(ImgSm0)
    return ImgAd

def imBitConvert(im,bit=16, norm=False, limit=None):
    """covert bit depth of the image

    Parameters
    ----------
    im : array
        input image
    bit : int
        output bit depth. 8 or 16
    norm : bool
        scale the image intensity range specified by limit to the full bit depth if True.
        Use min and max of the image if limit is not provided
    limit: list
        lower and upper limits of the image intensity

    Returns
        im : array
        converted image
    -------

    """
    im = im.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    if norm: # local or global normalization (for tiling)
        if not limit: # if lmit is not provided, perform local normalization, otherwise global (for tiling)
            limit = [np.nanmin(im[:]), np.nanmax(im[:])] # scale each image individually based on its min and max
        im = (im-limit[0])/(limit[1]-limit[0])*(2**bit-1) 

    im = np.clip(im, 0, 2**bit-1) # clip the values to avoid wrap-around by np.astype
    if bit==8:        
        im = im.astype(np.uint8, copy=False) # convert to 8 bit
    else:
        im = im.astype(np.uint16, copy=False) # convert to 16 bit
    return im

def imadjustStack(imStk, tol=1, bit=16,vin=[0,2**16-1]):
    for i in range(imStk.shape[2]):
        imStk[:,:,i] = imadjust(imStk[:,:,i])
    return imStk    

def imadjust(src, tol=1, bit=16,vin=[0,2**16-1]):
    """Python implementation of "imadjust" from MATLAB for stretching intensity histogram. Slow

    Parameters
    ----------
    src : array
        input image
    tol : int
        tolerance in [0, 100]
    bit : int
        output bit depth. 8 or 16
    vin : list
        src image bounds

    Returns
    -------

    """
    # TODO: rewrite using np.clip

    bitTemp = 16 # temporary bit depth for calculation.
    vout=(0,2**bitTemp-1)       
    if src.dtype == 'uint8':
        bit = 8

    src = imBitConvert(src, norm=True) # rescale to 16 bit       
    srcOri = np.copy(src) # make a copy of source image
    if len(src.shape) > 2:    
        src = np.mean(src, axis=2)
        src = imBitConvert(src, norm=True) # rescale to 16 bit                    
    
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
        dst = srcOri
    else:
        if len(srcOri.shape)>2:
            dst = np.array([])
            for i in range(0, srcOri.shape[2]):
                src = srcOri[:,:,i] 
                src = src.reshape(src.shape[0], src.shape[1],1)                              
                vd = linScale(src,vin, vout)                
                if dst.size:            
                    dst = np.concatenate((dst, vd), axis=2)
                else:
                    dst = vd      
        else:
            vd = linScale(src,vin, vout)
            dst = vd
    dst = imBitConvert(dst,bit=bit, norm=True)
    return dst

def imClip(img, tol=1):
    """
    Clip the images for better visualization
    """
    limit = np.percentile(img, [tol, 100-tol])
    img_clpped = np.clip(img, limit[0], limit[1])
    return img_clpped
    
def linScale(src,vin, vout):
    """Scale the source image according to input and output ranges

    Parameters
    ----------
    src
    vin
    vout

    Returns
    -------

    """
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale + 0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    return vd
#%%
def removeBubbles(I, kernelSize = (11,11)):
    """remove bright spots (mostly bubbles) in retardance images. Need to add a size filter

    Parameters
    ----------
    I
    kernelSize

    Returns
    -------

    """
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


def compute_flat_field(img_io, config):
    """
    Compute illumination function of fluorescence channels
    for flat-field correction

    Parameters
    ----------
    img_io: object
        mManagerReader object that holds the image parameters
    config: object
        ConfigReader object that holds the user input config parameters

    Returns
    -------
    img_io: object
        mManagerReader object that holds the image parameters with
        illumination function saved in img_io.img_fluor_bg

    """
    print('Calculating illumination function for flatfield correction...')
    ff_method = config.processing.ff_method
    img_io.ff_method = ff_method
    img_io.loopZ = 'flat_field'
    img_io = loopPos(img_io, config)
    if ff_method == 'open':
        img_fluor_bg = img_io.ImgFluorSum
    elif ff_method == 'empty':
        img_fluor_bg = img_io.ImgFluorMin
    for channel in range(img_fluor_bg.shape[0]):
        img_fluor_bg[channel] = img_fluor_bg[channel] - min(np.nanmin(img_fluor_bg[channel]), 0) + 1 #add 1 to avoid 0
        img_fluor_bg[channel] /= np.mean(img_fluor_bg[channel])  # normalize the background to have mean = 1
    img_io.img_fluor_bg = img_fluor_bg
    return img_io

def correct_flat_field(img_io, ImgFluor):
    """
    flat-field correction for fluorescence channels
    Parameters
    ----------
    img_io: object
        mManagerReader object that holds the image parameters
    ImgFluor: float32 array
        stack of fluorescence images with with shape (channel, y, x)

    Returns
    -------
    ImgFluor: array
        flat-field corrected fluorescence images
    """

    for i in range(ImgFluor.shape[0]):
        if np.any(ImgFluor[i, :, :]):  # if the flour channel exists
            ImgFluor[i, :, :] = ImgFluor[i, :, :] / img_io.img_fluor_bg[i, :, :]
    return ImgFluor