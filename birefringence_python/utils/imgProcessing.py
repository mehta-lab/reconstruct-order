import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import bisect
import warnings
import sys
sys.path.append("..") # Adds higher directory to python modules path.
sns.set_context("poster")
plt.close("all") # close all the figures from the last run
#%%
def ImgMin(Img, ImgBg):    
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

def imBitConvert(im,bit=16, norm=False, limit=None):    
    im = im.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    if norm: # local or global normalization (for tiling)
        if not limit: # if lmit is not provided, perform local normalization, otherwise global (for tiling)
            limit = [np.nanmin(im[:]), np.nanmax(im[:])] # scale each image individually based on its min and max
            
        im = (im-limit[0])/(limit[1]-limit[0])*(2**bit-1) 
    else: # only clipping, no scaling        
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
    return dst, vin

def imClip(img, tol=1, bit=16,vin=[0,2**16-1]):
    """
    Clip the images for better visualization
    """
    limit = np.percentile(img, [tol, 100-tol])
    imgClpped = np.clip(img, limit[0], limit[1])
    return imgClpped       
    
def linScale(src,vin, vout):
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale + 0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    return vd
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