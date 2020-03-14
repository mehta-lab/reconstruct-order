import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..utils.imgProcessing import nanRobustBlur, imadjust, im_bit_convert, imClip
from ..utils.imgProcessing import imcrop



def plotVectorField(img,
                    orientation,
                    anisotropy=1,
                    spacing=20,
                    window=20,
                    linelength=20,
                    linewidth=3,
                    linecolor='g',
                    colorOrient=True,
                    cmapOrient='hsv',
                    threshold=None,
                    alpha=1,
                    clim=[None, None],
                    cmapImage='gray',
                    showPlot=True):
    """Overlays orientation field on the image. Returns matplotlib image axes.

    Options:
        threshold:
        colorOrient: if it is True, then color the lines by their orientation.
        linelength : can be a scalar or an image the same size as the orientation.
    Parameters
    ----------
    img: nparray
        image to overlay orientation lines on
    orientation: nparray
        orientation in radian
    anisotropy: nparray
    spacing: int

    window: int
    linelength: int
        can be a scalar or an image the same size as the orientation
    linewidth: int
        width of the orientation line
    linecolor: str
    colorOrient: bool
        if it is True, then color the lines by their orientation.
    cmapOrient:
    threshold: nparray
        a binary numpy array, wherever the map is 0, ignore the plotting of the line
    alpha: int
        line transparency. [0,1]. lower is more transparent
    clim: list
        [min, max], min and max intensities for displaying img
    cmapImage:
        colormap for displaying the image
    Returns
    -------
    im_ax: obj
        matplotlib image axes
    """

    # plot vector field representaiton of the orientation map
    
    # Compute U, V such that they are as long as line-length when anisotropy = 1.
    U, V =  anisotropy*linelength * np.cos(2 * orientation), anisotropy*linelength * np.sin(2 * orientation)
    USmooth = nanRobustBlur(U, (window, window)) # plot smoothed vector field
    VSmooth = nanRobustBlur(V, (window, window)) # plot smoothed vector field
    azimuthSmooth = (0.5*np.arctan2(VSmooth,USmooth)) % np.pi
    RSmooth = np.sqrt(USmooth**2+VSmooth**2)
    USmooth, VSmooth = RSmooth*np.cos(azimuthSmooth), RSmooth*np.sin(azimuthSmooth)

    nY, nX = img.shape
    Y, X = np.mgrid[0:nY,0:nX] # notice the reversed order of X and Y
    
    # Sample variables at the specified spacing.
    Plotting_X = X[::spacing, ::spacing]
    Plotting_Y = Y[::spacing, ::spacing]
    Plotting_U = linelength * USmooth[::spacing, ::spacing]
    Plotting_V = linelength * VSmooth[::spacing, ::spacing]
    Plotting_R = RSmooth[::spacing, ::spacing]

    # Threshold the vector lines if specified.
    if threshold is None:
        threshold = np.ones_like(X) # no threshold
    Plotting_thres = threshold[::spacing, ::spacing]
    Plotting_orient=azimuthSmooth[::spacing, ::spacing]

    thresholdIdx = Plotting_thres==1
    Plotting_X=Plotting_X[thresholdIdx]
    Plotting_Y=Plotting_Y[thresholdIdx]
    Plotting_U=Plotting_U[thresholdIdx]
    Plotting_V=Plotting_V[thresholdIdx]
    Plotting_orient=Plotting_orient[thresholdIdx]
    Plotting_R=Plotting_R[thresholdIdx]
    Plotting_color=Plotting_orient * (180/np.pi)


    if showPlot:
        if colorOrient:
            im_ax = plt.imshow(img, cmap=cmapImage, vmin=clim[0], vmax=clim[1])
            plt.title('Orientation map')
            plt.quiver(Plotting_X, Plotting_Y,
                       Plotting_U, Plotting_V, Plotting_color,
                       cmap=cmapOrient,
                       edgecolor=linecolor, facecolor=linecolor, units='xy', alpha=alpha, width=linewidth,
                       headwidth=0, headlength=0, headaxislength=0,
                       scale_units = 'xy', scale = 1, angles = 'uv', pivot = 'mid')
        else:
            im_ax = plt.imshow(img, cmap=cmapImage, vmin=clim[0], vmax=clim[1])
            plt.title('Orientation map')
            plt.quiver(Plotting_X, Plotting_Y,
                       Plotting_U, Plotting_V,
                       edgecolor=linecolor,facecolor=linecolor,units='xy', alpha=alpha, width=linewidth,
                       headwidth = 0, headlength = 0, headaxislength = 0,
                       scale_units = 'xy',scale = 1, angles = 'uv', pivot = 'mid')
    else:
        im_ax=None

    return im_ax, Plotting_R, Plotting_orient, Plotting_X, Plotting_Y, Plotting_U, Plotting_V

def angular_hist(orientation,
                 n_bins=20,
                 weighted=True,
                 retardance=None,
                 ):

    """Plot angular histogram of orientation.
    The histogram is weighted by retardance

    Parameters
    ----------
    orientation: nparray
        orientation image in radian. Smoothed orientation output by
        plotVectorField is recommended to remove peaks from edge birefringence
    n_bins: int
        number of bins for plotting the histogram
    weighted: bool
        plot retardance-weighted orientation histogram (True) or
        the plain orientation histogram
    retardance: nparray
        retardance or anisotropy image. Smoothed retardance output by
        plotVectorField is recommended to remove peaks from edge birefringence
    Returns
    -------
    im_ax: obj
        matplotlib plot axes
    bars: obj
        histogram object
    """
    # Compute pie slices
    theta = np.linspace(0.0, np.pi, n_bins + 1, endpoint=True)
    width = np.pi / n_bins
    # get bin index for each pixel
    ori_bin_idx = np.digitize(orientation, theta) - 1
    if weighted:
        assert retardance is not None, \
            'Retardance is required to compute weighted histograms'
        counts = [np.sum(retardance[ori_bin_idx == ind]) for ind in range(n_bins)]
    else:
        counts = [np.sum(ori_bin_idx == ind) for ind in range(n_bins)]
    ax = plt.subplot(111, projection='polar')
    bars = ax.bar(theta[:-1], counts, width=width, bottom=0.0, align='edge')
    ax.axes.yaxis.set_ticklabels([])
    return ax, bars

def render_birefringence_imgs(img_io, imgs, config, spacing=20, vectorScl=5, zoomin=False, dpi=300, norm=True, plot=True):
    """ Parses transmission, retardance, orientation, and polarization images, scale and render them for export.  """

    outputChann = img_io.output_chans
    chann_scale = [0.25, 1, 0.05, 1, 1]  # scale fluor channels for composite images when norm=False
    
    I_trans,retard, azimuth, polarization, ImgFluor = imgs
    if zoomin: # crop the images
        imList = [I_trans, retard, azimuth]
        I_trans,retard, azimuth = imcrop(imList, I_trans)
    azimuth_degree = azimuth/np.pi*180

    # Compute Retardance+Orientation, Polarization+Orientation, and Retardance+Retardance+Orientation overlays
    # Slow, results in large files
    # Optionally, plot results in tiled image
    if plot or any(chann in ['Retardance+Orientation', 'Polarization+Orientation', 'Brightfield+Retardance+Orientation']
                   for chann in outputChann):
        I_azi_ret_trans, I_azi_ret, I_azi_scat = PolColor(I_trans, retard, azimuth_degree, polarization, norm=norm)
        t_idx = img_io.t_idx; z_idx = img_io.z_idx; pos_idx = img_io.pos_idx
        if plot:
            plot_recon_images(I_trans, retard, azimuth, polarization, I_azi_ret, I_azi_scat, zoomin=False, spacing=spacing,
                              vectorScl=vectorScl, dpi=dpi)
            if zoomin:
                figName = 'Retardance+Retardance+Orientation_Zoomin.jpg'
            else:
                figName = 'Retardance+Retardance+Orientation_t%03d_p%03d_z%03d.jpg' % (t_idx, pos_idx, z_idx)
    
            plt.savefig(os.path.join(img_io.img_output_path, figName), dpi=dpi, bbox_inches='tight')
     
    # Compute Retardance+Fluorescence and Retardance+Fluorescence_all overlays
    # Very slow
    if any(chann in ['Retardance+Fluorescence', 'Retardance+Fluorescence_all'] for chann in outputChann):
        IFluorRetard = CompositeImg([100*retard, ImgFluor[2,:,:]*chann_scale[2], ImgFluor[0,:,:]*chann_scale[0]], norm=norm)
        I_fluor_all_retard = CompositeImg([100 * retard,
                                           ImgFluor[4, :, :] * chann_scale[4],
                                           ImgFluor[3, :, :] * chann_scale[3],
                                           ImgFluor[2, :, :] * chann_scale[2],
                                           ImgFluor[1, :, :] * chann_scale[1],
                                           ImgFluor[0, :, :] * chann_scale[0]], norm=norm)
    
    imgDict = {}
    for chann in outputChann:
        if chann == 'Brightfield_computed':
            img = im_bit_convert(I_trans * config.plotting.transmission_scaling, bit=16, norm=False)  # AU, set norm to False for tiling images
            
        elif chann == 'Retardance':
            img = im_bit_convert(retard * config.plotting.retardance_scaling, bit=16)  # scale to pm
            
        elif chann == 'Orientation':
            img = im_bit_convert(azimuth_degree * 100, bit=16)  # scale to [0, 18000], 100*degree
            
        elif chann == 'Polarization':
            img = im_bit_convert(polarization * 50000, bit=16)
        
        elif chann == 'Orientation_x':
            azimuth_x = np.cos(2 * azimuth)
            img = im_bit_convert((azimuth_x + 1) * 1000, bit=16) # scale to [0, 1000]
        
        elif chann == 'Orientation_y':
            azimuth_y = np.sin(2 * azimuth)
            img = im_bit_convert((azimuth_y + 1) * 1000, bit=16)  # scale to [0, 1000]
        elif chann == 'Retardance+Orientation':
            img = I_azi_ret
            
        elif chann == 'Polarization+Orientation':
            img = I_azi_scat
            
        elif chann == 'Brightfield+Retardance+Orientation':
            img = I_azi_ret_trans

        elif chann == 'Retardance+Fluorescence':
            img = IFluorRetard
            
        elif chann == 'Retardance+Fluorescence_all':
            img = I_fluor_all_retard
            
        else:
            # TODO: write error handling
            continue
            
        imgDict[chann] = img
        
    img_io.channels = outputChann
    img_io.nChann = len(outputChann)
    return img_io, imgDict


def PolColor(s0, retardance, orientation, polarization, norm=True):
    """ Generate colormaps with following mappings, where H is Hue, S is Saturation, and V is Value.
        I_azi_ret_trans: H=Orientation, S=retardance, V=Brightfield.
        I_azi_ret: H=Orientation, V=Retardance.
        I_azi_pol: H=Orientation, V=Polarization.
    """

    if norm:
        retardance = imadjust(retardance, tol=1, bit=8)
        s0 = imadjust(s0, tol=1, bit=8)
        polarization = imadjust(polarization, tol=1, bit=8)
        # retardance = cv2.convertScaleAbs(retardance, alpha=(2**8-1)/np.max(retardance))
        # s0 = cv2.convertScaleAbs(s0, alpha=(2**8-1)/np.max(s0))
    else:
        #TODO: make scaling factors parameters in the config
        retardance = cv2.convertScaleAbs(retardance, alpha=50)
        s0 = cv2.convertScaleAbs(s0, alpha=100)
        polarization = cv2.convertScaleAbs(polarization, alpha=2000)
#    retardance = histequal(retardance)

    orientation = cv2.convertScaleAbs(orientation, alpha=1)
#    retardAzi = np.stack([orientation, retardance, np.ones(retardance.shape).astype(np.uint8)*255],axis=2)
    I_azi_ret_trans = np.stack([orientation, retardance, s0], axis=2)
    I_azi_ret = np.stack([orientation, np.ones(retardance.shape).astype(np.uint8) * 255, retardance], axis=2)
    I_azi_scat = np.stack([orientation, np.ones(retardance.shape).astype(np.uint8) * 255, polarization], axis=2)
    I_azi_ret_trans = cv2.cvtColor(I_azi_ret_trans, cv2.COLOR_HSV2RGB)
    I_azi_ret = cv2.cvtColor(I_azi_ret, cv2.COLOR_HSV2RGB)
    I_azi_scat = cv2.cvtColor(I_azi_scat, cv2.COLOR_HSV2RGB)  #
#    retardAzi = np.stack([orientation, retardance],axis=2)
    return I_azi_ret_trans, I_azi_ret, I_azi_scat


def CompositeImg(images, norm=True):
    """ Generate overlay of multiple images.  """
    #TODO: update name and docstring to clarify intent.

    img_num = len(images)
    hsv_cmap = cm.get_cmap('hsv', 256)
    color_idx = np.linspace(0, 1, img_num+1)
    color_idx = color_idx[:-1] # leave out 1, which corresponds color similar to 0

    ImgColor = []
    idx = 0
    for img in images:
        if norm:
            img_one_chann = imadjust(img, tol=0.01, bit=8)
        else:
            img_one_chann = im_bit_convert(img, bit=8, norm=False)

        img_one_chann = img_one_chann.astype(np.float32,
                                             copy=False)  # convert to float32 without making a copy to save memory
        img_one_chann_rgb = [img_one_chann * hsv_cmap(color_idx[idx])[0], img_one_chann * hsv_cmap(color_idx[idx])[1],
                             img_one_chann * hsv_cmap(color_idx[idx])[2]]
        img_one_chann_rgb = np.stack(img_one_chann_rgb, axis=2)
        ImgColor +=[img_one_chann_rgb]
        idx += 1
    ImgColor = np.stack(ImgColor)
    ImgColor = np.sum(ImgColor, axis=0)
    ImgColor = im_bit_convert(ImgColor, bit=8, norm=False)

    return ImgColor


def plot_recon_images(s0, retard, azimuth, polarization, I_azi_ret, I_azi_scat, zoomin=False, spacing=20, vectorScl=1, dpi=300):
    """ Plot transmission, retardance, orientation, and polarization images, and prepares them for export.  """

    anisotropy = retard
    anisotropy = anisotropy / np.nanmean(anisotropy)  # normalization
    # %%
    figSize = (18, 12)
    fig = plt.figure(figsize=figSize)
    ax1 = plt.subplot(2, 3, 1)
    plt.tick_params(labelbottom=False, labelleft=False)  # labels along the bottom edge are off
    ax_trans = plt.imshow(imClip(s0, tol=1), cmap='gray')
    plt.title('Brightfield_computed')
    plt.xticks([]), plt.yticks([])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(ax_trans, cax=cax, orientation='vertical')
    #    plt.show()

    ax2 = plt.subplot(2, 3, 2)
    plt.tick_params(labelbottom=False, labelleft=False)  # labels along the bottom edge are off
    ax_retard = plt.imshow(imClip(retard, tol=1), cmap='gray')
    plt.title('Retardance(nm)')
    plt.xticks([]), plt.yticks([])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(ax_retard, cax=cax, orientation='vertical')

    ax3 = plt.subplot(2, 3, 3)
    plt.tick_params(labelbottom=False, labelleft=False)  # labels along the bottom edge are off
    ax_pol = plt.imshow(imClip(polarization, tol=1), cmap='gray')
    plt.title('Polarization')
    plt.xticks([]), plt.yticks([])
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(ax_pol, cax=cax, orientation='vertical')

    ax4 = plt.subplot(2, 3, 4)
    plt.tick_params(labelbottom=False, labelleft=False)  # labels along the bottom edge are off
    ax_hv = plt.imshow(imadjust(I_azi_ret, tol=1, bit=8), cmap='hsv')
    plt.title('Retardance+Orientation')
    plt.xticks([]), plt.yticks([])
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(ax_hv, cax=cax, orientation='vertical', ticks=np.linspace(0, 255, 5))
    cbar.ax.set_yticklabels([r'$0^o$', r'$45^o$', r'$90^o$', r'$135^o$',
                             r'$180^o$'])  # vertically oriented colorbar
    #    plt.show()

    ax5 = plt.subplot(2, 3, 5)
    im_ax = plotVectorField(imClip(retard / 1000, tol=1), azimuth, anisotropy=anisotropy, spacing=spacing)
    plt.tick_params(labelbottom=False, labelleft=False)  # labels along the bottom edge are off
    plt.title('Retardance(nm)+Orientation')
    plt.xticks([]), plt.yticks([])

    ax6 = plt.subplot(2, 3, 6)
    plt.tick_params(labelbottom=False, labelleft=False)  # labels along the bottom edge are off
    ax_hsv = plt.imshow(imadjust(I_azi_scat, tol=1, bit=8), cmap='hsv')
    # plt.title('Brightfield+Retardance\n+Orientation')
    plt.title('Polarization+Orientation')
    plt.xticks([]), plt.yticks([])
    plt.show()
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(ax_hsv, cax=cax, orientation='vertical', ticks=np.linspace(0, 255, 5))
    cbar.ax.set_yticklabels([r'$0^o$', r'$45^o$', r'$90^o$', r'$135^o$', r'$180^o$'])  # vertically oriented colorbar


def plot_stokes(img_io, img_stokes, img_stokes_sm):
    t_idx = img_io.t_idx
    z_idx = img_io.z_idx
    pos_idx = img_io.pos_idx
    titles = ['s0', 's1', 's2', 's3']
    fig_name = 'stokes_t%03d_p%03d_z%03d.jpg' % (t_idx, pos_idx, z_idx)
    plot_sub_images(img_stokes, titles, img_io.img_output_path, fig_name, colorbar=True)
    fig_name = 'stokes_sm_t%03d_p%03d_z%03d.jpg' % (t_idx, pos_idx, z_idx)
    plot_sub_images(img_stokes_sm, titles, img_io.img_output_path, fig_name, colorbar=True)


def plot_pol_imgs(img_io, imgs_pol, titles):
    t_idx = img_io.t_idx
    z_idx = img_io.z_idx
    pos_idx = img_io.pos_idx
    fig_name = 'imgs_pol_t%03d_p%03d_z%03d.jpg' % (t_idx, pos_idx, z_idx)
    plot_sub_images(imgs_pol, titles, img_io.img_output_path, fig_name, colorbar=True)


def plot_sub_images(images,titles, img_output_path, figName, colorbar=False):
    n_imgs = len(images)
    n_rows = 2
    n_cols = np.ceil(n_imgs/n_rows).astype(np.uint32)
    fig, ax = plt.subplots(n_rows, n_cols, squeeze=False)
    ax = ax.flatten()
    for axs in ax:
        axs.axis('off')
    fig.set_size_inches((5 * n_cols, 5 * n_rows))
    axis_count = 0
    for img, title in zip(images, titles):
        if len(img.shape) == 3:
            ax_img = ax[axis_count].imshow(img)
        else:
            ax_img = ax[axis_count].imshow(imClip(img), cmap='gray')
        ax[axis_count].axis('off')
        ax[axis_count].set_title(title)
        if colorbar:
            divider = make_axes_locatable(ax[axis_count])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = plt.colorbar(ax_img, cax=cax, orientation='vertical')
        axis_count += 1
    fig.savefig(os.path.join(img_output_path, figName), dpi=300, bbox_inches='tight')



