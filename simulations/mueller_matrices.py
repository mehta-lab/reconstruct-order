# -*- coding: utf-8 -*-

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def stokes2ellipse(S):
    """
    Compute ellipticity and orientation of a polarization state specified by four Stokes parameters.

    :param S: Stokes vector
    :return: ellipticity: -1,...,0,...,1 for left circular, linear,  and right circular states
    :return: orientation: Orientation of polarization ellipse in radian.
    """
    s1Norm = S[1] / S[0]
    s2Norm = S[2] / S[0]
    s3Norm = S[3] / S[0]
    linearity = np.sqrt(s1Norm ** 2 + s2Norm ** 2)
    if linearity:
        ellipseAngle = np.arctan2(s3Norm, linearity)
        ellipt = np.sin(ellipseAngle)

    else:
        ellipt = np.sign(s3Norm)

    orient = 0.5 * np.arctan2(s2Norm, s1Norm)
    return (ellipt, orient)


def plotEllipse(ellipt, orient, size=5, axes=None):
    def add_arrow(line, size=20, color='k'):
        """
        add an arrow to a line.

        line:       Line2D object
        position:   x-position of the arrow. If None, mean of xdata is taken
        direction:  'left' or 'right'
        size:       size of the arrow in fontsize points
        color:      if None, line color is taken.

        adapted from: https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
        """
        if color is None:
            color = line.get_color()

        xdata = line.get_xdata()
        ydata = line.get_ydata()

        position = xdata.mean()
        # find closest index
        start_ind = np.argmin(np.absolute(xdata - position))
        end_ind = start_ind - 1

        line.axes.annotate('',
                           xytext=(xdata[start_ind], ydata[start_ind]),
                           xy=(xdata[end_ind], ydata[end_ind]),
                           arrowprops=dict(arrowstyle="->", color=color),
                           size=size)

    theta = np.linspace(0, 2 * np.pi, 360)
    rotmat = [[np.cos(orient), -np.sin(orient)], [np.sin(orient), np.cos(orient)]]

    x = np.cos(theta)
    y = ellipt * np.sin(theta)
    (xr, yr) = np.matmul(rotmat, [x.flatten(1), y.flatten(1)])

    if axes == None:
        plt.figure(figsize=[size, size])
        axes = plt.axes()
    else:
        plt.sca(axes)
    axes.axis('square')
    line = plt.plot(xr, yr)[0]
    add_arrow(line)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)


def M_LinearPolarizer(theta=0):
    """
    Mueller matrix of ideal linear polarizer rotated at angle theta
    
    Parameters
    ----------
    theta: symbol or float
        Angle of the linear polarizer, default value is zero
    
    """
    M = 0.5 * sp.Matrix([[1, sp.cos(2*theta), sp.sin(2*theta), 0],
                        [sp.cos(2*theta), sp.cos(2*theta)**2, sp.sin(2*theta)*sp.cos(2*theta), 0],
                        [sp.sin(2*theta), sp.sin(2*theta)*sp.cos(2*theta), sp.sin(2*theta)**2, 0],
                        [0, 0, 0, 0]])
    return M

def M_Diattenuator(theta=0, Tmax=1, Tmin=0):
    """
    Mueller matrix of linear diattenuator
    
    Parameters
    ----------
    theta: symbol or float
        Angle of the linear polarizer, default value is zero
    
    Tmax: symbol or float
        Maximum transmission of the diattenuator, defaults to one
        
    Tmin: symbol or float
        Minimum transmission of the diattenuator, defaults to zero 
    
    """
    t1 = Tmax + Tmin
    t2 = Tmax - Tmin
    t3 = 2*sp.sqrt(Tmax*Tmin)
    
    M = 0.5 * sp.Matrix([[t1, t2*sp.cos(2*theta), t2*sp.sin(2*theta), 0],
                        [t2*sp.cos(2*theta), t1*sp.cos(2*theta)**2 + t3*sp.sin(2*theta)**2, (t1-t3)*sp.sin(2*theta)*sp.cos(2*theta), 0],
                        [t2*sp.sin(2*theta), (t1-t3)*sp.sin(2*theta)*sp.cos(2*theta), t1*sp.sin(2*theta)**2 + t3*sp.cos(2*theta)**2, 0],
                        [0, 0, 0, t3]])
    return M

def M_Retarder(theta=0, delta=np.pi):
    """
    Mueller matrix of linear retarder
    
    Parameters
    ----------
    theta: symbol or float
        Angle of fast axis of the retarder, defaults to zero
        
    delta: symbol or float
        Retardance of the retarder, defaults to pi for half-waveplate
    """
    M = sp.Matrix([[1, 0, 0, 0],
                  [0, sp.cos(2*theta)**2+sp.sin(2*theta)**2*sp.cos(delta), sp.sin(2*theta)*sp.cos(2*theta)*(1-sp.cos(delta)), -sp.sin(2*theta)*sp.sin(delta)],
                  [0, sp.sin(2*theta)*sp.cos(2*theta)*(1-sp.cos(delta)), sp.sin(2*theta)**2+sp.cos(2*theta)**2*sp.cos(delta), sp.cos(2*theta)*sp.sin(delta)],
                  [0, sp.sin(2*theta)*sp.sin(delta), -sp.cos(2*theta)*sp.sin(delta), sp.cos(delta)]])
    return M

def M_rotate(M, theta):
    """
    Rotates Mueller matrix M by angle theta
    
    Parameters
    ----------
    M: sympy or numpy array
        Input Mueller matrix
        
    theta: symbol or float
        Rotation angle
    """
    rmTheta = sp.Matrix([[1, 0, 0, 0],
                        [0, sp.cos(2*theta), sp.sin(2*theta), 0],
                        [0, -sp.sin(2*theta), sp.cos(2*theta), 0],
                        [0, 0, 0, 1]])
    
    rmNegTheta = sp.Matrix([[1, 0, 0, 0],
                        [0, sp.cos(2*theta), -sp.sin(2*theta), 0],
                        [0, sp.sin(2*theta), sp.cos(2*theta), 0],
                        [0, 0, 0, 1]])
    
    M_out = rmNegTheta @ M @ rmTheta
    return M_out