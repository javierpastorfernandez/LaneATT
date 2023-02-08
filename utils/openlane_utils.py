# SHELL COLOR CLASS

import matplotlib.pyplot as plt
import logging
import math
import numpy as np
from scipy.interpolate import interp1d
import random
from interval import interval as pyinterval
import os,sys,glob



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

from scipy.optimize import linear_sum_assignment
def associate_elements(old_trackedLanes, trackedLanes, thr_dist):
    log = logging.getLogger('logger')

    log.trace(bcolors.OKGREEN+"old_trackedLanes (function):"+bcolors.ENDC+str(old_trackedLanes))
    log.trace(bcolors.OKGREEN+"trackedLanes (function):"+bcolors.ENDC+str(trackedLanes))

    n = old_trackedLanes.shape[0]
    m = trackedLanes.shape[0]

    cost_matrix = np.zeros((n, m))
    infoCount=0
    for i in range(n):
        for j in range(m):
            diff = abs(old_trackedLanes[i,0] - trackedLanes[j,0])
            if diff > thr_dist:
                cost_matrix[i, j] = 100
            else:
                cost_matrix[i, j] = diff

    log.trace(bcolors.OKGREEN+"cost_matrix (function):\n"+bcolors.ENDC+str(cost_matrix))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # i -> old_tracked; j-> trackedLanes

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j]<=thr_dist:
            trackedLanes[j,1]=old_trackedLanes[i,1]

    # sort_idxs=np.argsort(trackedLanes[:,0])
    # sort_idxs=sort_idxs[::-1]
    # trackedLanes=trackedLanes[sort_idxs,:]


    sort_idxs=np.argsort(old_trackedLanes[:,1])
    old_trackedLanes=old_trackedLanes[sort_idxs,:]

    log.trace(bcolors.OKGREEN+"old_trackedLanes (after linear assignment):"+bcolors.ENDC+str(old_trackedLanes))
    log.trace(bcolors.OKGREEN+"trackedLanes ( after linear assignment):"+bcolors.ENDC+str(trackedLanes))


    noInfo_idxs=np.where(np.isnan(trackedLanes[:,1]))[0]
    for idx_i in noInfo_idxs:
        for idx_j in range(old_trackedLanes.shape[0]-1):
            if ((idx_j==0) and  (trackedLanes[idx_i,0]<old_trackedLanes[0,0])):

                mask = ~np.isnan(trackedLanes[:,1])
                values  = trackedLanes[mask,1]
                values=np.append(values,old_trackedLanes[:,1])
                trackedLanes[idx_i,1]=np.min(values)-1

            elif ( (trackedLanes[idx_i,0]>old_trackedLanes[idx_j,0]) and  (trackedLanes[idx_i,0]<=old_trackedLanes[idx_j+1,0]) ):
                trackedLanes[idx_i,1]=0.5*(old_trackedLanes[idx_j,1]+old_trackedLanes[idx_j+1,1])

        if (trackedLanes[idx_i,0]>old_trackedLanes[-1,0]):
            mask = ~np.isnan(trackedLanes[:,1])
            values  = trackedLanes[mask,1]
            values=np.append(values,old_trackedLanes[:,1])
            trackedLanes[idx_i,1]=np.max(values)+1


    # original_idxs = np.argsort(sort_idxs)
    # trackedLanes = trackedLanes[original_idxs, :]

    return trackedLanes




def prepare4resampling(lane,dims=["ref","num","num","cat"],sem_weight=True):
    ref=dims.index("ref")


    # Get unique x values and their indices
    unique_ref, indices = np.unique(lane[:, ref], return_index=True)
    new_lane=[]

    # Iterate over unique x values and calculate mean y values
    for unique_value in unique_ref:
        duplicates=lane[lane[:, ref] == unique_value]
        if sem_weight: # filter results that do not overlap with semantics
            mask=duplicates[:, 3] != -1

            if np.sum(mask)>0:
                duplicates=duplicates[mask] # variable categorica distinta de -1

        row=[]
        for idx_i in range(len(dims)):
            dim=dims[idx_i]
            if dim=="ref":
                row.append(unique_value)
            elif dim=="num":
                row.append(np.mean(duplicates[:, idx_i]))

            elif dim=="cat":
                row.append(np.mean(duplicates[:, idx_i]).round().astype("int"))
        new_lane.append(row)

    return np.array(new_lane)


def associate_polylines(polylines,trackedPolys, thr_dist=[3.5,5.0],dims=["ref","query","num","cat"]):
    log = logging.getLogger('logger')

    ref=dims.index("ref")
    query=dims.index("query")
    lanes=[]


    if len(polylines)>0:
        poly_01_range=pyinterval[polylines[0][0,ref],polylines[0][-1,ref]]

    for idx_i in range(len(polylines)-1):
        poly_02_range=pyinterval[polylines[(idx_i+1)][0,ref],polylines[(idx_i+1)][-1,ref]]
        overlap=poly_01_range&poly_02_range

        log.debug(bcolors.OKGREEN+"poly_01_range:"+bcolors.ENDC+str(poly_01_range))
        log.debug(bcolors.OKGREEN+"poly_02_range:"+bcolors.ENDC+str(poly_02_range))

        if len(overlap)>0: # overlap is a pyinterval
            overlap=overlap[0]
            log.trace(bcolors.OKGREEN+"overlap:"+bcolors.ENDC+str(overlap))

            poly_01_id=trackedPolys[(idx_i),1]
            poly_02_id=trackedPolys[(idx_i+1),1]

            poly_01=polylines[(idx_i)]
            poly_02=polylines[(idx_i+1)]

            poly_01=poly_01[ (poly_01[:, ref] >= overlap[0])& (poly_01[:, ref] <= overlap[1])]
            poly_02=poly_02[ (poly_02[:, ref] >= overlap[0])& (poly_02[:, ref] <= overlap[1])]
            log.debug(bcolors.OKGREEN+"poly_01:\n"+bcolors.ENDC+str(poly_01))
            log.debug(bcolors.OKGREEN+"poly_02:\n"+bcolors.ENDC+str(poly_02))


            diff=np.absolute(poly_01[:, query]-poly_02[:, query])
            log.debug(bcolors.OKGREEN+"diff:\n"+bcolors.ENDC+str(diff))

            diff=np.mean(diff)
            log.debug(bcolors.OKGREEN+"diff:"+bcolors.ENDC+str(diff))

            if((diff>=thr_dist[0]) and  (diff<=thr_dist[1])) :
                # lanes.append(idx_i,(idx_i+1))
                lanes.append([poly_01_id,poly_02_id])

        poly_01_range=poly_02_range

    return np.array(lanes)






def resample_laneline(lane, dims=["ref","num","num","cat"]):
    log = logging.getLogger('logger')

    """
    Interpolate lanes so that they have equal sampling
    """
    """
    MAYBE USING POLYFIT ALSO IN THE FUTURE
    coeffs= np.polyfit(input_lane[:, 1], input_lane[:, 0], order[0])
    x_values = func_mod(y_steps, coeffs)
    """

    log.debug(bcolors.OKGREEN+"lane:\n"+bcolors.ENDC+str(lane))
    lane=prepare4resampling(lane,dims=dims)
    log.debug(bcolors.OKGREEN+"lane:\n"+bcolors.ENDC+str(lane))

    ref=dims.index("ref")
    min_ref=int(np.min(lane[:,ref])-1)
    max_ref=int(np.max(lane[:,ref])+1)

    new_ref=np.append(np.arange(min_ref,max_ref,1),max_ref)
    new_lane=np.empty((new_ref.shape[0],len(dims)))

    for idx_i in range(len(dims)):
        if dims[idx_i]=="num":
            if lane.shape[0]>=3: # quadratic
                new_info = interp1d(lane[:,ref], lane[:,idx_i], kind='quadratic',fill_value="extrapolate")(new_ref)
            elif lane.shape[0]>=2: #  linear
                new_info = interp1d(lane[:,ref], lane[:,idx_i], kind='linear',fill_value="extrapolate")(new_ref)
            elif lane.shape[0]>=1: # constant
                new_info=np.squeeze(np.ones((new_ref.shape[0],1)))*lane[0,ref]

        elif dims[idx_i]=="cat": # categorical
            new_info = interp1d(lane[:,ref], lane[:,idx_i], kind='nearest',fill_value="extrapolate")(new_ref)

        elif dims[idx_i]=="ref":
            new_info=new_ref

        new_lane[:,idx_i]=new_info


    log.debug(bcolors.OKGREEN+"new_lane:\n"+bcolors.ENDC+str(new_lane))

    return new_lane




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


def init_logger(name,verbose):
    create_trace_loglevel(logging)

    if verbose==0:
        level=logging.INFO

    elif verbose==1:
        level=logging.TRACE

    elif verbose==2:
        level=logging.DEBUG


    logger = logging.getLogger(name)  #1
    logger.setLevel(level)  #2
    handler = logging.StreamHandler(sys.stderr)  #3
    # handler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName)) #4

    handler.setLevel(level)  #4
    # formatter = logging.Formatter('%(created)f:%(levelname)s:%(name)s:%(module)s:%(message)s') #5
    # formatter = logging.Formatter(bcolors.HEADER+'%(created)f:%(module)s:%(levelname)s'+bcolors.ENDC+':%(message)s') #5
    formatter = logging.Formatter(bcolors.HEADER+'%(module)s:%(levelname)s'+bcolors.ENDC+'| %(message)s') #5
    formatter = logging.Formatter(bcolors.HEADER+'%(module)s:%(lineno)d:%(levelname)s'+bcolors.ENDC+'| %(message)s') #5




    handler.setFormatter(formatter)  #6
    logger.addHandler(handler)  #7

    return logger
