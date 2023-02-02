import numpy as np


def SampleFromPlane(plane_model, view_region,sampling_period=1): # view_region=[x_min, x_max, y_min, y_max]
    x=np.arange(view_region[0],view_region[1],sampling_period) # in LIDAR points
    y=np.arange(view_region[2],view_region[3],sampling_period) # in LIDAR points
    [a, b, c, d] = plane_model

    # points=np.empty((0,4))
    points=[]
    for x_ in x:
        for y_ in y:
            z_=(-d-a*x_-b*y_)/c
            # points=np.append(points,[[x_,y_,z_,1]],axis=1)
            points.append([x_,y_,z_,1])

    return np.array(points)



def Homography2Cart(array):

    if array.ndim<=1:
        array=array.reshape(1,-1) # npts (1), dims

    for dim in range(array.shape[1]):
        array[:,dim]= array[:,dim]/array[:,-1]


    array=array[:,:-1]
    return array



