import sys
sys.path.append("..") # Adds higher directory to python modules path.
#import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets  import RectangleSelector

#sns.set_context("poster")
plt.close("all") # close all the figures from the last run
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

