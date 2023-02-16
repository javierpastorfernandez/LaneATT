# SHELL COLOR CLASS

import matplotlib.pyplot as plt
import logging
import math
import numpy as np
from scipy.interpolate import interp1d
import random
from interval import interval as pyinterval
import os,sys,glob
import cv2



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
def associate_elements(ref_array, query_array, thr_dist,tag_col=0,score_col=2):
    log = logging.getLogger('logger')

    log.trace(bcolors.OKGREEN+"ref_array (function):\n"+bcolors.ENDC+str(ref_array))
    log.trace(bcolors.OKGREEN+"query_array (function):\n"+bcolors.ENDC+str(query_array))

    n = ref_array.shape[0]
    m = query_array.shape[0]

    cost_matrix = np.zeros((n, m))
    infoCount=0
    for i in range(n):
        for j in range(m):
            diff = abs(ref_array[i,score_col] - query_array[j,score_col]) # compare scores
            if diff > thr_dist:
                cost_matrix[i, j] = 100
            else:
                cost_matrix[i, j] = diff

    log.trace(bcolors.OKGREEN+"cost_matrix (function):\n"+bcolors.ENDC+str(cost_matrix))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # i -> old_tracked; j-> query_array

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j]<=thr_dist:
            query_array[j,tag_col]=ref_array[i,tag_col] # set tags

    # Sort ref_array according to their tags
    sort_idxs=np.argsort(ref_array[:,tag_col])
    ref_array=ref_array[sort_idxs,:]

    log.trace(bcolors.OKGREEN+"ref_array (after linear assignment)\n:"+bcolors.ENDC+str(ref_array))
    log.trace(bcolors.OKGREEN+"query_array ( after linear assignment)\n:"+bcolors.ENDC+str(query_array))


    noInfo_idxs=np.where(np.isnan(query_array[:,tag_col]))[0] # search in query_array where there is no tag info
    for idx_i in noInfo_idxs:
        for idx_j in range(ref_array.shape[0]-1):
            # if the value not matched is BETWEEN values present in the ref_array
            if  ( (query_array[idx_i,score_col]>ref_array[idx_j,score_col]) and  (query_array[idx_i,score_col]<=ref_array[idx_j+1,score_col]) ):
                query_array[idx_i,tag_col]=0.5*(ref_array[idx_j,tag_col]+ref_array[idx_j+1,tag_col])

        # if the value not matched is SMALLER values present in the ref_array
        if (query_array[idx_i,score_col]<ref_array[0,score_col]): # tag smaller than first tag of old_trackedLaness
            mask = ~np.isnan(query_array[:,tag_col]) # query_array mask where there is info
            values  = query_array[mask,tag_col] # valores tag de query_array
            values=np.append(values,ref_array[:,tag_col]) # valores tag de ref_array
            query_array[idx_i,tag_col]=np.min(values)-1

        # if the value not matched is SMALLER values present in the ref_array
        elif (query_array[idx_i,score_col]>ref_array[-1,score_col]):
            mask = ~np.isnan(query_array[:,tag_col]) # tag values of query_array
            values  = query_array[mask,tag_col]
            values=np.append(values,ref_array[:,tag_col]) # tag values of OldTrackedLanes
            query_array[idx_i,tag_col]=np.max(values)+1
    return query_array


def ImageBalance(img,sem_img,plot=True):
    log = logging.getLogger('logger')
    if plot:
        cv2.imshow("sem_img", sem_img)# Show image
    unique_colors = np.unique(sem_img.reshape(-1, 3), axis=0)
    log.trace(bcolors.OKGREEN+"unique_colors: "+bcolors.ENDC+str(unique_colors))


    sem_img = cv2.resize(sem_img, (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST)

    """
    img=(np.array(img,copy=True)*255).astype("uint8") # Torch to numpy array
    img=np.swapaxes(img, 0, 1)
    img=np.swapaxes(img, 1,2)
    """

    height,width,_=img.shape
    alpha=0.8

    sky_color=[255, 206, 135]
    vegetation_color= [194, 253, 147]

    dashed_color = [255, 0, 128]
    solid_color = [37,  193,  255]
    pavement_color =  [255,   0, 255]



    mask_option=2
    if mask_option==0:
        # create a black image
        color_mask_3d = np.zeros(img.shape, dtype=np.uint8)

        # define the four corners of the trapezoid
        pt1 = (50, height)
        pt2 = (width-50, height)
        pt3 = (width-300, height-200)
        pt4 = (300, height-200)

        # create a polygon using the four points
        pts = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)

        # fill the polygon with ones (white)
        cv2.fillPoly(color_mask_3d, [pts], color=(255,255,255))

    elif mask_option==1:
        color_mask = np.all(sem_img == sky_color, axis=2)| np.all(sem_img == vegetation_color, axis=2)
        color_mask_3d = np.ones_like(sem_img)*255
        color_mask_3d[color_mask,:] = [0,0,0]

    elif mask_option==2:
        color_mask = np.all(sem_img == dashed_color, axis=2)| np.all(sem_img == solid_color, axis=2)| np.all(sem_img == pavement_color, axis=2)
        color_mask_3d = np.zeros_like(sem_img)
        color_mask_3d[color_mask,:] = [255,255,255]


    # color_indexes = np.transpose(np.nonzero(color_mask))


    """
    img= alpha*img1 +(1-alpha)img2
    img = cv2.addWeighted(img1,0.3,img2_resized,0.7,0)
    """

    img = cv2.addWeighted(img,alpha,color_mask_3d,alpha, 0) # Following line overlays transparent rectangle over the image


    if plot:
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
        cv2.namedWindow("color_mask_3d", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions

        cv2.imshow("img", img)# Show image
        cv2.imshow("color_mask_3d", color_mask_3d)# Show image

        cv2.waitKey(0)
        cv2.destroyAllWindows()





def prepare4resampling(lane,dims=["ref","num","num","cat"],sem_weight=True):
    ref=dims.index("ref")


    # Get unique x values and their indices
    unique_ref, indices = np.unique(lane[:, ref], return_index=True)
    new_lane=[]

    # Iterate over unique x values and calculate mean y values
    for unique_value in unique_ref:
        duplicates=lane[lane[:, ref] == unique_value]
        if sem_weight and len(dims)>=4: # filter results that do not overlap with semantics
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





poly_01_range=pyinterval[1,0]
poly_02_range=pyinterval[1,0]
overlap=poly_01_range&poly_02_range

def associate_polylines_with_tracking(polylines,tracking_LUT, thr_dist=[3.5,5.0],dims=["ref","query","num","cat"]):
    log = logging.getLogger('logger')
    tag_col=0

    ref=dims.index("ref")
    query=dims.index("query")
    lanes=[]

    """ polylines follow the same order as  tracking_LUT"""

    for idx_i in range(len(tracking_LUT)-1):

        if (polylines[idx_i].shape[0]>0) and (polylines[idx_i+1].shape[0]>0):

            poly_01_range=pyinterval[polylines[(idx_i)][0,ref],polylines[(idx_i)][-1,ref]]
            poly_02_range=pyinterval[polylines[(idx_i+1)][0,ref],polylines[(idx_i+1)][-1,ref]]
            overlap=poly_01_range&poly_02_range

            log.debug(bcolors.OKGREEN+"poly_01_range:"+bcolors.ENDC+str(poly_01_range))
            log.debug(bcolors.OKGREEN+"poly_02_range:"+bcolors.ENDC+str(poly_02_range))

            # si hay overlap, hay possible matching to lanes
            if len(overlap)>0: # overlap is a pyinterval
                overlap=overlap[0]
                log.trace(bcolors.OKGREEN+"overlap:"+bcolors.ENDC+str(overlap))

                tag_01=tracking_LUT[(idx_i),tag_col]
                tag_02=tracking_LUT[(idx_i+1),tag_col]

                poly_01=polylines[(idx_i)]
                poly_02=polylines[(idx_i+1)]

                poly_01=poly_01[ (poly_01[:, ref] >= overlap[0])& (poly_01[:, ref] <= overlap[1])]
                poly_02=poly_02[ (poly_02[:, ref] >= overlap[0])& (poly_02[:, ref] <= overlap[1])]

                log.debug(bcolors.OKGREEN+"poly_01:\n"+bcolors.ENDC+str(poly_01))
                log.debug(bcolors.OKGREEN+"poly_02:\n"+bcolors.ENDC+str(poly_02))


                diff=np.absolute(poly_01[:, query]-poly_02[:, query])
                log.trace(bcolors.OKGREEN+"diff:\n"+bcolors.ENDC+str(diff))

                diff=np.mean(diff)
                log.trace(bcolors.OKGREEN+"diff:"+bcolors.ENDC+str(diff))
                # breakpoint()

                if((diff>=thr_dist[0]) and  (diff<=thr_dist[1])) :
                    # lanes.append(idx_i,(idx_i+1))
                    lanes.append([tag_01,tag_02])

        else:
            """HARDCODED"""
            # breakpoint()


    return np.array(lanes)





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

"""
numerator = -math.sqrt(a22**2) - a12 * b - 2 * a * a12 * t1 - 2 * a * a11 * a12 * x + a22
denominator = 2 * a * a12**2

"""

def transform_equation(a, b, c, a11, a12, a21, a22, t1, t2,x):
    numerator = -math.sqrt(a12**2 * b**2 - 2 * a12 * a22 * b - 4 * a * a12**2 * c - 4 * a * a12 * a22 * t1 + 4 * a * a12**2 * t2 - 4 * a * a11 * a12 * a22 * x + 4 * a * a21 * a12**2 * x + a22**2) - a12 * b - 2 * a * a12 * t1 - 2 * a * a11 * a12 * x + a22
    denominator = 2 * a * a12**2
    result=numerator / denominator

    return result



def predict_equation_V3(old_coeffs,lidar_prev2lidar_tf, xs_prev,filter_negative=True):
    log = logging.getLogger('logger')
    #dDist indica que el coche esta avanzando

    poly_predictions=[]
    # for old_coeff in old_coeffs:

    for key in list(old_coeffs.keys()):
        old_coeff=old_coeffs[key]
        log.trace(bcolors.OKGREEN+"old_coeff:"+bcolors.ENDC+str(old_coeff))

        ys_prev=[]
        zs_prev=[]

        """"
        for x_prev in xs_prev:
            ys_prev.append(old_coeff[0,0]*x_prev**2+old_coeff[0,1]*x_prev+old_coeff[0,2])
            zs_prev.append(old_coeff[1,0]*x_prev**2+old_coeff[1,1]*x_prev+old_coeff[1,2])
        """

        ys_prev= np.polyval(old_coeff[0,:], xs_prev)   # evaluate the polynomial
        zs_prev= np.polyval(old_coeff[1,:], xs_prev)   # evaluate the polynomial


        info_prev=np.append(np.array(xs_prev).reshape(-1,1),np.array(ys_prev).reshape(-1,1),axis=1)
        info_prev=np.append(info_prev,np.array(zs_prev).reshape(-1,1),axis=1)
        info_prev=np.append(info_prev,np.ones((info_prev.shape[0],1)),axis=1)

        info_current=np.matmul(lidar_prev2lidar_tf,info_prev.T).T # npts,3

        for dim in range(info_current.shape[1]):
            info_current[:,dim]= info_current[:,dim]/info_current[:,-1]
        info_current=info_current[:,:-1]


        sort_idxs=np.argsort((info_current[:,0]))

        if not ((sort_idxs==np.arange(info_current.shape[0])).all()):
            """HARDCODED"""
            # breakpoint()

        # if filter_negative:
        #     info_current=info_current[info_current[:,0]>=0]

        poly_predictions.append(info_current)


    log.trace(bcolors.OKGREEN+"poly_predictions:"+bcolors.ENDC+str(poly_predictions))

    return poly_predictions



def predict_equation_V2(old_coeffs,dHeading,dPitch,dDist, x_samples):
    log = logging.getLogger('logger')
    #dDist indica que el coche esta avanzando
    x_samples=x_samples+dDist

    # R_heading=getRotation_2d(-dHeading,units="degrees") # Prueba_01
    R_heading=getRotation_2d(dHeading,units="degrees") # Prueba_02
    T_heading=[-dDist,0]

    A_heading=np.append(R_heading,np.array(T_heading).reshape(-1,1),axis=1)
    A_heading=np.append(A_heading,np.array([0,0,1]).reshape(1,-1),axis=0)
    A_heading_inv=np.linalg.inv(A_heading)

    R_pitch=getRotation_2d(dPitch,units="degrees") # Prueba_02




    log.trace(bcolors.OKGREEN+"Rotation matrix:\n"+bcolors.ENDC+str(R))
    log.trace(bcolors.OKGREEN+"Translation matrix:\n"+bcolors.ENDC+str(T))
    log.trace(bcolors.OKGREEN+"Affine transformation matrix:\n"+bcolors.ENDC+str(A))
    poly_predictions=[]

    for old_coeff in old_coeffs:
        log.trace(bcolors.OKGREEN+"old_coeff:"+bcolors.ENDC+str(old_coeff))

        y_prev=[]
        zs=[]

        for x_prev in x_samples:
            y_prev.append(old_coeff[0,0]*x_prev**2+old_coeff[0,1]*x_prev+old_coeff[0,2])
            zs.append(old_coeff[1,0]*x_prev**2+old_coeff[1,1]*x_prev+old_coeff[1,2])


        info_prev=np.append(np.array(x_samples).reshape(-1,1),np.array(y_prev).reshape(-1,1),axis=1)
        info_prev=np.append(info_prev,np.ones((info_prev.shape[0],1)),axis=1)
        # info_prev=np.append(info_prev,np.array(zs).reshape(-1,1),axis=1)


        info_current=np.matmul(A,info_prev.T).T # npts,3
        info_current[:,0]= info_current[:,0]/info_current[:,2]
        info_current[:,1]=info_current[:,1]/info_current[:,2]
        info_current=info_current[:,:2]

        prediction=np.append(info_current,np.array(zs).reshape(-1,1),axis=1)

        sort_idxs=np.argsort((prediction[:,0]))
        assert((sort_idxs==np.arange(prediction.shape[0])).all())

        poly_predictions.append(prediction)


    log.trace(bcolors.OKGREEN+"poly_predictions:"+bcolors.ENDC+str(poly_predictions))

    return poly_predictions

def predict_equation(old_coeffs,dHeading,dDist, x_samples):
    log = logging.getLogger('logger')

    R=getRotation_2d(dHeading,units="degrees")
    T=[dDist,0]

    log.trace(bcolors.OKGREEN+"Rotation matrix:\n"+bcolors.ENDC+str(R))
    log.trace(bcolors.OKGREEN+"Translation matrix:\n"+bcolors.ENDC+str(T))

    a11=R[0,0]
    a21=R[1,0]
    a12=R[0,1]
    a22=R[1,1]
    t1=T[0]
    t2=T[1]

    poly_predictions=[]


    for old_coeff in old_coeffs:
        log.trace(bcolors.OKGREEN+"old_coeff:"+bcolors.ENDC+str(old_coeff))

        ys=[]
        zs=[]
        for x in x_samples:
            ys.append(transform_equation(old_coeff[0,0],old_coeff[0,1],old_coeff[0,2], a11, a12, a21, a22, t1, t2,x))
            zs.append(transform_equation(old_coeff[1,0],old_coeff[1,1],old_coeff[1,2], a11, a12, a21, a22, t1, t2,x))

        prediction=np.append(np.array(x_samples).reshape(-1,1),np.array(ys).reshape(-1,1),axis=1)
        prediction=np.append(prediction,np.array(zs).reshape(-1,1),axis=1)
        poly_predictions.append(prediction)


    log.trace(bcolors.OKGREEN+"poly_predictions:"+bcolors.ENDC+str(poly_predictions))

    return poly_predictions


def resample_laneline_with_coeffs(coeffs , x_samples):
    log = logging.getLogger('logger')

    new_y=np.polyval(coeffs[0,:],x_samples)
    new_z=np.polyval(coeffs[1,:],x_samples)

    new_info=np.append(x_samples.reshape(-1,1),new_y.reshape(-1,1),axis=1)
    new_info=np.append(new_info,new_z.reshape(-1,1),axis=1)

    log.trace(bcolors.OKGREEN+"new_info:\n"+bcolors.ENDC+str(new_info))
    return new_info





def resample_laneline(lane, dims=["ref","num","num","cat"],filter_negative=True):
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

    resample_option=1
    tracked_coeffs=[]

    if resample_option==1:
        for idx_i in range(len(dims)):
            if dims[idx_i]=="num":
                if lane.shape[0]>=3: # quadratic
                    coeffs= np.polyfit(lane[:,ref], lane[:,idx_i], 2)
                elif lane.shape[0]>=2: #  linear
                    coeffs= np.polyfit(lane[:,ref], lane[:,idx_i], 1)
                elif lane.shape[0]>=1: # constant
                    coeffs= np.polyfit(lane[:,ref], lane[:,idx_i], 0)

                new_info=np.polyval(coeffs,new_ref)

                tracked_coeffs.append(coeffs)

            elif dims[idx_i]=="cat": # categorical
                if  lane.shape[0]>=2: #  linear
                    new_info = interp1d(lane[:,ref], lane[:,idx_i], kind='nearest',fill_value="extrapolate")(new_ref)
                else:
                    new_info=(np.squeeze(np.ones((new_ref.shape[0],1)))*lane[0,idx_i]).astype("int")

            elif dims[idx_i]=="ref":
                new_info=new_ref

            new_lane[:,idx_i]=new_info


    elif resample_option==2:


        for idx_i in range(len(dims)):
            if dims[idx_i]=="num":
                if lane.shape[0]>=3: # quadratic
                    new_info = interp1d(lane[:,ref], lane[:,idx_i], kind='quadratic',fill_value="extrapolate")(new_ref)
                elif lane.shape[0]>=2: #  linear
                    new_info = interp1d(lane[:,ref], lane[:,idx_i], kind='linear',fill_value="extrapolate")(new_ref)
                elif lane.shape[0]>=1: # constant
                    new_info=np.squeeze(np.ones((new_ref.shape[0],1)))*lane[0,idx_i]

            elif dims[idx_i]=="cat": # categorical
                if  lane.shape[0]>=2: #  linear
                    new_info = interp1d(lane[:,ref], lane[:,idx_i], kind='nearest',fill_value="extrapolate")(new_ref)
                else:
                    new_info=(np.squeeze(np.ones((new_ref.shape[0],1)))*lane[0,idx_i]).astype("int")

            elif dims[idx_i]=="ref":
                new_info=new_ref

            new_lane[:,idx_i]=new_info



    if filter_negative:
        new_lane=new_lane[new_lane[:,0]>=5]

    log.debug(bcolors.OKGREEN+"new_lane:\n"+bcolors.ENDC+str(new_lane))
    tracked_coeffs=np.array(tracked_coeffs)

    return new_lane,tracked_coeffs




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



def getRotation_2d(angle,units="radians"):
    log = logging.getLogger('logger')

    if units.lower()=="degrees":
        angle=angle*math.pi/180 # From degrees to radians
        log.trace(bcolors.OKGREEN+"Input to degrees "+bcolors.ENDC)

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
