import numpy as np
import cv2


def rescale_projection(org_size,tf_size,matrix):
    scale_factor_projection=np.array([
    [(tf_size[0]/org_size[0]),0, 0],
    [0,  (tf_size[1]/org_size[1]), 0],
    [0,             0, 1]])

    matrix = np.matmul(scale_factor_projection,matrix)
    return matrix


def DrawPoints(img,points,alpha=False,option="numpy",color=(20, 20, 20), thickness = 5,radius = 5):
    overlay = img.copy()

    points=points.astype("int") # int -> Coordenadas de imagen
    if option=="opencv":
        for x,y in points:
            overlay = cv2.circle(overlay, (x,y), radius, color, thickness)

    elif option=="numpy":
        points=points[(points[:,1]<=(overlay.shape[0]-1))&(points[:,1]>=0)]
        points=points[(points[:,0]<=(overlay.shape[1]-1))&(points[:,0]>=0)]
        overlay[points[:,1],points[:,0]]=color

    if alpha:
        alpha = alpha[0]  # Transparency factor.
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0) # Following line overlays transparent rectangle over the image
        return img

    return overlay



def increasing_separation_arange(start, stop,step,factor):
    i = 0
    result = [start]
    end=start

    while (end+step) <= stop:
        end=end+step
        result.append(end)
        step*=factor

    result.append(stop)
    return np.array(result)


def SampleFromPlane(plane_model, view_region,sampling_period=[1,1]): # view_region=[x_min, x_max, y_min, y_max]
    # x=np.arange(view_region[0],view_region[1],sampling_period[0]) # in LIDAR points
    x=increasing_separation_arange(view_region[0],view_region[1],sampling_period[0],1.5)
    y=np.arange(view_region[2],view_region[3],sampling_period[1]) # in LIDAR points


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



