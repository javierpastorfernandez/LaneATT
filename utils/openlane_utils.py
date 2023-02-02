# SHELL COLOR CLASS

import matplotlib.pyplot as plt
import logging
import math
import numpy as np



class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    DARKGREEN = '\033[36m'
    WHITE = '\033[37m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
white_bv = True

def getRotation(angle,axis=0,units="radians"):
    log = logging.getLogger('logger')

    if units.lower()=="degrees":
        angle=angle*math.pi/180
        log.trace(bcolors.OKGREEN+"Input to degrees "+bcolors.ENDC)

    axis=str(axis)
    if ((axis=="0")or(axis.lower()=="x") or (axis.lower()=="roll")):
        # Roll,  x
        R= np.array([[1,            0,              0],
                    [0, math.cos(angle), -math.sin(angle)],
                    [0, math.sin(angle),  math.cos(angle)]])


    elif ((axis=="1")or(axis.lower()=="y") or (axis.lower()=="pitch")):
        # Pitch, y
        R= np.array([[ math.cos(angle), 0,   math.sin(angle)],
                    [              0, 1,               0],
                    [-math.sin(angle),  0,  math.cos(angle)]])

    elif ((axis=="2")or(axis.lower()=="z") or (axis.lower()=="yaw")):
        # Yaw, z
        R= np.array([[math.cos(angle), -math.sin(angle),              0],
                    [ math.sin(angle),  math.cos(angle),              0],
                    [             0,              0,              1]])


    else:
        assert False, "Axis argument incorrect! Check arguments"

    log.trace("Axes: "+str(axis)+" Rotation Matrix:\n"+str(R))

    return R



def getRotation_2d(angle,axis=0,units="radians"):
    R= np.array([[np.cos(angle), -np.sin(angle)],
                [ np.sin(angle),  np.cos(angle)]])
    return R


def getRigidTransformation(R,T):
    log = logging.getLogger('logger')

    R=np.array(R) # 3x3
    T=np.array(T).reshape(3,1)

    M=np.append(R,T,axis=1)

    log.trace("Rigid Transformation:\n"+str(M))

    return M

def getRigidTransformation_h(R,T):
    log = logging.getLogger('logger')
    R=np.array(R) # 3x3
    T=np.array(T).reshape(3,1)
    M=np.append(R,T,axis=1) # 3 x 4
    M=np.append(M,np.array([0,0,0,1]).reshape(1,-1),axis=0)
    log.debug("Rigid Transformation:\n"+str(M))

    return M



def FormatAxes(ax,fontsize=14,tick_labels=True):
    ax.view_init(elev=90, azim=0) # 2D View
    ax.grid(True)
    if not tick_labels:
        if  ax.name == "3d":
            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_ticks([])
        else:
            for dim in (ax.xaxis, ax.yaxis):
                dim.set_ticks([])

    # Set axes labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)

    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    if  ax.name == "3d":
        ax.set_zlabel('z')
        ax.zaxis.label.set_size(fontsize)
        ax.tick_params(axis='z', labelsize=10)


    # Properties:
        # size -> Size of the axis
        # labelsize -> Size of ticks




def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def create_trace_loglevel(logging):
    "Add TRACE log level and Logger.trace() method."

    logging.TRACE = 11
    logging.addLevelName(logging.TRACE, "TRACE")

    def _trace(logger, message, *args, **kwargs):
        if logger.isEnabledFor(logging.TRACE):
            logger._log(logging.TRACE, message, args, **kwargs)

    logging.Logger.trace = _trace
