import pickle
import random
import logging
import os,glob,sys
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation


from utils.openlane_utils import bcolors,FormatAxes, get_cmap, create_trace_loglevel,\
    associate_elements,resample_laneline,associate_polylines,getRotation_2d,predict_equation,predict_equation_V2,\
    predict_equation_V3,associate_polylines_with_tracking,resample_laneline_with_coeffs,ImageBalance
from utils.symbolic_tracking import LaneTracker
from utils.projections_utils import SampleFromPlane,Homography2Cart,DrawPoints,rescale_projection

from tfmatrix import transformations
import math
from interval import interval as pyinterval
import copy


# Point clouds

import open3d as o3d
print(o3d.__version__)

class Runner:
    def __init__(self, cfg, exp, device, resume=False, view=None, deterministic=False):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.logger = logging.getLogger(__name__)

        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self):
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg['epochs']
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()
        for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.train()
            pbar = tqdm(train_loader)
            for i, (images, labels, _) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(images, **self.cfg.get_train_parameters())
                loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scheduler step (iteration based)
                scheduler.step()

                # Log
                postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
                postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)
                postfix_dict['loss'] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)

            # Validate
            if (epoch + 1) % self.cfg['val_every'] == 0:
                self.eval(epoch, on_val=True)
        self.exp.train_end_callback()

    def eval(self, epoch, on_val=False, save_predictions=False):
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch))
        model = model.to(self.device)
        model.eval()
        if on_val:
            dataloader = self.get_val_dataloader()
        else:
            dataloader = self.get_test_dataloader()
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        self.logger.info(bcolors.OKGREEN+'View?: ' +bcolors.ENDC+str(self.view))

        # Initialize tracker
        """ def __init__(self, n_lanes, proc_noise_scale, meas_noise_scale,
                 process_cov_parallel=0, proc_noise_type='white'):"""

        initialization=True
        tracking_init=False
        colors_by_lane={}
        pcb_fields=["x","y","z","type","poly_id","tracking_id","iter","weight"]
        point_cloud=np.empty((0,len(pcb_fields)))

        gps_init=False
        tracking_predictions=[]

        # HARDCODED
        dDist=0
        dHeading=0


        tag_col=0
        pred_idx_col=1
        score_col=2


        poly_id=0
        iter_n=-1

        # SAVING OPTIONS
        save_local_info=False
        save_point_cloud=False

        save_global_info=True
        video=True
        save_csv=True


        # Adjusting poly_id and iter_n to previous dataframe save
        output_dir=os.path.join(dataloader.dataset.dataset.saving_dir,"csv")
        os.makedirs(output_dir, exist_ok=True)
        self.logger.trace (bcolors.OKGREEN + "Looking for csv in directory: " + bcolors.WHITE+ str(output_dir))

        pattern = '{}/*{}'.format(output_dir,".csv")
        self.logger.trace (bcolors.OKGREEN + "with the pattern: " + bcolors.WHITE+ str(pattern))
        files= glob.glob(pattern)
        files.sort()

        if len(files)>0: # if there are csv, update poly_id and iter_n
            df=pd.read_csv(files[-1]) # last csv
            """ pcb_fields=["x","y","z","type","poly_id","tracking_id","iter","weight"]"""

            iter_n=int(df.loc[:,"iter"].max())
            poly_id=int(df.loc[:,"poly_id"].max())

            self.logger.trace (bcolors.OKGREEN + "iter_n: " + bcolors.WHITE+ str(iter_n)+
            bcolors.OKGREEN + " poly_id: " + bcolors.WHITE+ str(poly_id))
            # HARDCODED
            # breakpoint()

        poly_id+=1
        with torch.no_grad():
            for idx, (images, _, _) in enumerate(tqdm(dataloader)):
                iter_n+=1 # +1 iteration

                images = images.to(self.device)
                output = model(images, **test_parameters)
                prediction = model.decode(output, as_lanes=True)
                predictions.extend(prediction)

                # RETRIEVING LANE POINTS FROM PREDICTIONS
                img, label, _ = dataloader.dataset.__getitem__(idx)
                label = dataloader.dataset.label_to_lanes(label)
                img_path=dataloader.dataset.dataset[idx]["path"]
                self.logger.trace(bcolors.OKGREEN+'img_path: ' +bcolors.ENDC+str(img_path))
                filename=int(img_path.split("/")[-1].split(".")[-2])
                self.logger.info(bcolors.OKGREEN+'filename: ' +bcolors.ENDC+str(filename))

                cam_num=filename
                lidar_num=dataloader.dataset.dataset.UseSyncFile("cam0","lidar",cam_num) # From lidar file to cam
                gps_num=dataloader.dataset.dataset.UseSyncFile("cam0","gps",cam_num) # From lidar file to cam

                if lidar_num==None: # si no hay info, no me la voy a inventar
                    continue



                dataset_name=dataloader.dataset.dataset_name
                self.logger.trace(bcolors.OKGREEN+'dataset_name: ' +bcolors.ENDC+str(dataset_name))

                # HARDCODED
                # if cam_num==1653479961500:
                #     breakpoint()
                if dataset_name=="bosch":
                    lidar_path = dataloader.dataset.dataset.sequence_path + "/pc_filtered/semantic/camera_0/" + str(lidar_num) + ".pcd"
                    self.logger.trace(bcolors.OKGREEN+'lidar_path: ' +bcolors.ENDC+str(lidar_path))
                    pcd = o3d.io.read_point_cloud(lidar_path)
                    pcb_pts=np.array(pcd.points)
                    self.logger.debug(bcolors.OKGREEN+'pcd: ' +bcolors.ENDC+str(pcd)) # PointCloud with ... npts
                    self.logger.debug(bcolors.OKGREEN+'pcd (Npts): ' +bcolors.ENDC+str(pcb_pts.shape[0]))


                    # Check if there is PointCloud

                    if (pcb_pts.shape[0]==0) and (not tracking_init):
                        continue
                    if pcb_pts.shape[0]==0:
                        point_cloud_exist=False
                    else:
                        point_cloud_exist=True

                    if point_cloud_exist:
                        pcb_pts=np.append(pcb_pts,np.ones((pcb_pts.shape[0],1)),axis=1)

                        # Filter by height
                        pcb_pts_filtered_idxs=np.where(pcb_pts[:,2]<=-1.5)[0] # Points near floor; tupple size 1
                        pcb_pts_filtered=pcb_pts[pcb_pts_filtered_idxs,:]

                        filtered_pcd = o3d.geometry.PointCloud()
                        filtered_pcd.points = o3d.utility.Vector3dVector(pcb_pts_filtered[:,0:3])


                        # Obtain plane from the road

                        """
                        plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)
                        """
                        plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.05,ransac_n=3,num_iterations=1000)
                            # ax + by + cz +d=0
                            # z=(d-ax-by)/c

                        # plane_points=SampleFromPlane(plane_model,[0,50,-15,15],sampling_period=1) -> MARTA
                        # A partir del 5 se ve
                        plane_points=SampleFromPlane(plane_model,[4,103,-15,15],sampling_period=[0.1,0.01])


                        [a, b, c, d] = plane_model
                        print(f"Lidar plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")


                        """
                        TRYING TO PROJECT THE PLANE EQUATION INTO IMAGE COORDS: IMPOSSIBLE
                        point_lidar=plane_points[0,:].reshape(1,-1)

                        # Cart2Homography (plane_lidar)
                        plane_lidar=list(plane_model[0:3])
                        plane_lidar.append(1)
                        plane_lidar=np.array(plane_lidar).reshape(1,-1)

                        point_camera=np.matmul(point_lidar,dataloader.dataset.dataset.vel2cam_M).reshape(-1)
                        point_camera=Homography2Cart(point_camera).reshape(-1)


                        plane_camera=np.matmul(plane_lidar,dataloader.dataset.dataset.vel2cam_M).reshape(-1)
                        plane_camera=Homography2Cart(plane_camera).reshape(-1)
                        [a_cam, b_cam, c_cam] = plane_camera
                        d_cam=-a_cam*point_camera[0]-b_cam*point_camera[1]-c_cam*point_camera[2]

                        plane_camera=[a_cam, b_cam, c_cam,d_cam]

                        #ax +by +cz+d=0
                        #d=-ax-by-cz

                        print(f"Camera plane equation: {a_cam:.2f}x + {b_cam:.2f}y + {c_cam:.2f}z + {d_cam:.2f} = 0")
                        plane_points_camera=SampleFromPlane(plane_camera,[-50,50,-50,50],sampling_period=0.1)

                        plane_points_projected=np.matmul(plane_points_camera,np.transpose(dataloader.dataset.dataset.P_sem)) # dim,nPts *presult (4*3)= npts,3
                        plane_points_projected=Homography2Cart(plane_points_projected)
                        plane_points_projected=plane_points_projected[:,[0,1]]



                        """

                        draw_plane_sem=False

                        if draw_plane_sem:
                            # PROJECT POINTS TO SEMANTICS
                            plane_points_cpy=np.array(plane_points,copy=True)
                            plane_points_projected_sem=np.matmul(plane_points_cpy,dataloader.dataset.dataset.P_lidar2sem) # dim,nPts *presult (4*3)= npts,3
                            plane_points_projected_sem[:,0]=plane_points_projected_sem[:,0]/plane_points_projected_sem[:,2]
                            plane_points_projected_sem[:,1]=plane_points_projected_sem[:,1]/plane_points_projected_sem[:,2]
                            plane_points_projected_sem=plane_points_projected_sem[:,:2] # 0,1 -> x,y


                        # PROJECT POINTS IMAGE
                        plane_points_cpy=np.array(plane_points,copy=True)
                        plane_points_projected_img=np.matmul(plane_points_cpy,dataloader.dataset.dataset.P_lidar2img_resize) # dim,nPts *presult (4*3)= npts,3
                        plane_points_projected_img[:,0]=plane_points_projected_img[:,0]/plane_points_projected_img[:,2]
                        plane_points_projected_img[:,1]=plane_points_projected_img[:,1]/plane_points_projected_img[:,2]
                        plane_points_projected_img=plane_points_projected_img[:,:2] # 0,1 -> x,y


                        # Visualize the point cloud
                        sem_image_path = dataloader.dataset.dataset.sequence_path + "/semantic_images/camera_0/" + str(cam_num) + ".png"
                        self.logger.trace(bcolors.OKGREEN+'sem_image_path: ' +bcolors.ENDC+str(sem_image_path))

                        sem_img=cv2.imread(sem_image_path)
                        sem_img_paint=cv2.imread(sem_image_path)

                        img_paint=np.array(img,copy=True) # Torch to numpy array
                        img_paint=img_paint*255
                        img_paint=img_paint.astype("uint8")

                        img_paint=np.swapaxes(img_paint, 0, 1)
                        img_paint=np.swapaxes(img_paint, 1,2)


                        if draw_plane_sem:
                            pcb_pts_projected=np.matmul(pcb_pts_filtered,dataloader.dataset.dataset.P_lidar2sem) # dim,nPts *presult (4*3)= npts,3
                            pcb_pts_projected=Homography2Cart(pcb_pts_projected)

                            self.logger.debug(bcolors.OKGREEN+'pcb_pts_projected: ' +bcolors.ENDC+str(pcb_pts_projected))
                            outliers = np.setdiff1d(np.arange(len(filtered_pcd.points)), inliers)

                            pcb_pts_projected_inliers=pcb_pts_projected[inliers]
                            pcb_pts_projected_outliers=pcb_pts_projected[outliers]

                            idxs=np.where(  (pcb_pts_projected[:,0]>0)& (pcb_pts_projected[:,0]<sem_img.shape[0])  &
                            (pcb_pts_projected[:,1]>0)& (pcb_pts_projected[:,1]>sem_img.shape[1]))[0] # Points near floor; tupple size 1
                            # self.logger.trace(bcolors.OKGREEN+'idxs: ' +bcolors.ENDC+str(idxs))


                        # img_balanced=ImageBalance(img_paint,sem_img_paint)

                        # CROP FILTER POINTS ACCORDING TO IMAGE DIMENSIONS
                        indexes =np.squeeze(np.where(   (plane_points_projected_img[:,0]>0) &  (plane_points_projected_img[:,0]<img_paint.shape[1])  ))
                        plane_points_projected_img=plane_points_projected_img[indexes]
                        plane_points=plane_points[indexes]

                        indexes =np.squeeze(np.where(   (plane_points_projected_img[:,1]>0) &  (plane_points_projected_img[:,1]<img_paint.shape[0])  ))
                        plane_points_projected_img=plane_points_projected_img[indexes]
                        plane_points=plane_points[indexes]

                        # Settings for image drawing/painting
                        radius = 5 # Radius of circle
                        thickness = 5   # Line thickness of 2 px


                        if draw_plane_sem:
                            # Plane INLIERS painted in BLUE in SEM_SEG
                            sem_img_paint= DrawPoints(sem_img_paint,pcb_pts_projected_inliers,alpha=False, color = (0, 255, 0), thickness = thickness,radius = radius)

                            # Plane OUTLIERS painted in RED in SEM_SEG
                            sem_img_paint= DrawPoints(sem_img_paint,pcb_pts_projected_outliers,alpha=False, color = (0, 0, 255), thickness = thickness,radius = radius)

                            # PLANE SAMPLED POINTS painted in grey overlay in SEM_SEGq
                            sem_img_paint= DrawPoints(sem_img_paint,plane_points_projected_sem,alpha=[0.4],  color = (50,0,255), thickness = thickness,radius = radius)


                        # PLANE SAMPLED POINTS painted in grey overlay in SEM_SEG
                        # alpha >>>>>> imagen mas nitida

                        plot=False
                        if plot:
                            img_paint= DrawPoints(img_paint,plane_points_projected_img,alpha=[0.4],  color = (50,0,255), thickness = 1,radius = 1)

                            cv2.namedWindow("sem_img", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
                            cv2.imshow("sem_img", sem_img)# Show image

                            cv2.namedWindow("sem_img_paint", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
                            cv2.imshow("sem_img_paint", sem_img_paint)# Show image

                            cv2.namedWindow("img_paint", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
                            cv2.imshow("img_paint", img_paint)# Show image

                            cv2.waitKey(0)
                            cv2.destroyAllWindows()



                        #     pcd = o3d.io.read_point_cloud(lidar_path)

                        #     # LOAD SYNC
                        #     dataloader.dataset.dataset.
                        # # // Precompute partial matrix multiplication
                        # # cv::Mat presult = velo2cam.t() * P.t();
                        # Eigen::MatrixXd pts_3D_rect_hom = cart2hom(pts_3D_rect);
                        # Eigen::MatrixXd pts_2D = pts_3D_rect_hom * P.transpose();


                        """
                        P_lidar2img = np.matmul(np.transpose(self.velo2cam),np.transpose(self.P))
                        = 4x4 * 3x4
                        4x3= 4x4 * 4x3
                        result = point3D.t() * presult;
                        result = result / result.at<double>(0,2); -> From homography to euclidean space

                        """

                    self.logger.trace(bcolors.OKGREEN+'gps_file: ' +bcolors.ENDC+str(gps_num))

                    # LOAD GPS INFO
                    gps_path=os.path.join(dataloader.dataset.dataset.gps_dir,gps_num+".txt")
                    with open(gps_path, "r") as f_gps:
                        gpslines = f_gps.readlines()
                        assert(len(gpslines)==1) # GPS FILE SHOULD CONTAIN ONLY ONE LINE
                        gpslines=gpslines[0].split()

                        # oxts_INS_filtered contents:
                        # UTM_E,  UTM_N, altitude, ns, ns, ns, ns, heading, pitch, roll, dT
                        UTM_E=float(gpslines[0])
                        UTM_N=float(gpslines[1])
                        altitude=float(gpslines[2])
                        heading=float(gpslines[7])
                        yaw=heading
                        pitch=float(gpslines[8])
                        roll=float(gpslines[9])
                        assert(len(gpslines)==11)# The number of atributes in the GPS file should be 11

                        self.logger.trace(bcolors.OKGREEN+'UTM_E: ' +bcolors.ENDC+str(UTM_E))
                        self.logger.trace(bcolors.OKGREEN+'UTM_N: ' +bcolors.ENDC+str(UTM_N))

                        self.logger.trace(bcolors.OKGREEN+'altitude: ' +bcolors.ENDC+str(altitude))
                        self.logger.trace(bcolors.OKGREEN+'pitch: ' +bcolors.ENDC+str(pitch*180/math.pi))
                        self.logger.trace(bcolors.OKGREEN+'heading: ' +bcolors.ENDC+str(heading*180/math.pi))
                        self.logger.trace(bcolors.OKGREEN+'roll: ' +bcolors.ENDC+str(roll*180/math.pi))

                    if not gps_init:
                        gps_init=True


                        UTM_E0=  dataloader.dataset.dataset.UTM_E0
                        UTM_N0=  dataloader.dataset.dataset.UTM_N0

                        """
                        UTM_E0=0
                        UTM_N0=0

                        """


                        # Set for next iteration
                        UTM_E_prev=UTM_E0
                        UTM_N_prev=UTM_N0

                        TV_UTM_E = UTM_E - UTM_E0
                        TV_UTM_N = UTM_N - UTM_N0

                        dataloader.dataset.dataset.UTM_E=UTM_E
                        dataloader.dataset.dataset.UTM_N=UTM_N

                        dataloader.dataset.dataset.tf_tree.add_transform_data("map", "gps", TV_UTM_E, TV_UTM_N, altitude,   0.0, pitch, heading)
                        dataloader.dataset.dataset.tf_tree.add_transform_data("map", "gps_prev", TV_UTM_E, TV_UTM_N, altitude,   0.0, pitch, heading)


                        # Initialize the tf with the Identity and no translation
                        lidar_prev2lidar_tf_old=np.ones((3,3))
                        lidar_prev2lidar_tf_old=np.append(lidar_prev2lidar_tf_old,np.zeros((3,1)),axis=1)
                        lidar_prev2lidar_tf_old=np.append(lidar_prev2lidar_tf_old,np.array([0,0,0,1]).reshape(1,-1),axis=0)
                        self.logger.trace(bcolors.OKGREEN+'lidar_prev2lidar_tf_old:\n' +bcolors.ENDC+str(lidar_prev2lidar_tf_old))

                    else:
                        TV_UTM_E = UTM_E - UTM_E0
                        TV_UTM_N = UTM_N - UTM_N0

                        # Obtain data for calculation of tf for change of axis
                        dE=UTM_E-UTM_E_prev
                        dN=UTM_N-UTM_N_prev

                        dDist=math.sqrt(dE**2+dN**2) # Modulus
                        UTM_E_prev=UTM_E
                        UTM_N_prev=UTM_N


                        self.logger.debug(bcolors.OKGREEN+'TV_UTM_E: ' +bcolors.ENDC+str(TV_UTM_E))

                        # UPDATE TRANSFORM LIDAR2GPS
                        dataloader.dataset.dataset.tf_tree.update_transform_data("map", "gps", TV_UTM_E, TV_UTM_N, altitude,   0.0, pitch, heading)
                        lidar_prev2lidar_tf=dataloader.dataset.dataset.tf_tree.lookup_transform("lidar", "lidar_prev")


                        translation_x=lidar_prev2lidar_tf[0,3]
                        translation_y=lidar_prev2lidar_tf[1,3]
                        tf_thr=10 # this should be set in accordance to the frame rate and so


                        # Calculation of angle
                        rotationMatrix = Rotation.from_dcm(lidar_prev2lidar_tf[:3, :3])
                        rotEuler=rotationMatrix.as_euler('zyx', degrees=True)
                        dHeading=rotEuler[0]
                        dPitch=rotEuler[1]
                        dRoll=rotEuler[2]

                        self.logger.trace(bcolors.OKGREEN+'dHeading (degrees):' +bcolors.ENDC+str(dHeading))
                        self.logger.trace(bcolors.OKGREEN+'dPitch (degrees):' +bcolors.ENDC+str(dPitch))
                        self.logger.trace(bcolors.OKGREEN+'dDist: ' +bcolors.ENDC+str(dRoll))
                        self.logger.trace(bcolors.OKGREEN+'translation_x: ' +bcolors.ENDC+str(translation_x))
                        self.logger.trace(bcolors.OKGREEN+'translation_y: ' +bcolors.ENDC+str(translation_y))

                        if (abs(translation_x)>tf_thr or abs(translation_y)>tf_thr): # either 1 of the 2 surpasses the threshold
                            self.logger.trace(bcolors.OKGREEN+'lidar_prev2lidar_tf (temp):\n' +bcolors.ENDC+str(lidar_prev2lidar_tf))
                            lidar_prev2lidar_tf=lidar_prev2lidar_tf_old
                            # breakpoint()

                        self.logger.trace(bcolors.OKGREEN+'lidar_prev2lidar_tf (def):\n' +bcolors.ENDC+str(lidar_prev2lidar_tf))
                        lidar_prev2lidar_tf_old=lidar_prev2lidar_tf

                        # lidar_prev2lidar_tf=dataloader.dataset.dataset.tf_tree.lookup_transform("lidar_prev", "lidar")


                        x_sample_min=0+lidar_prev2lidar_tf[0,-1]
                        x_sample_max=50+lidar_prev2lidar_tf[0,-1]

                        x_samples=np.arange(x_sample_min,x_sample_max,1)
                        # tracking_predictions=predict_equation(old_tracked_coeffs,dHeading,dDist,x_samples)
                        # tracking_predictions=predict_equation_V2(old_tracked_coeffs,dHeading,dPitch,dDist,x_samples)
                        tracking_predictions= predict_equation_V3(old_tracked_coeffs,lidar_prev2lidar_tf, x_samples)
                        tracking_predictions_tags=np.empty((len(tracking_predictions),3))* np.nan # if not, it initializes it with kind of random values
                        "tags, score "
                        dim=0
                        other_dim=1

                        for idx_i in range(len(tracking_predictions)):
                            points=tracking_predictions[idx_i] # Lane points / Lidar coordinate system
                            self.logger.debug(bcolors.OKGREEN+'points: \n' +bcolors.ENDC+str(points))
                            tracking_predictions_tags[idx_i,score_col]=points[0,other_dim]

                        sort_idxs=np.argsort(tracking_predictions_tags[:,score_col])
                        tracking_predictions=[tracking_predictions[sort_idx] for sort_idx in sort_idxs]
                        # old_tracked_coeffs=[old_tracked_coeffs[sort_idx] for sort_idx in sort_idxs]
                        tracking_predictions_tags=tracking_predictions_tags[sort_idxs,:]
                        tracking_predictions_tags[:,pred_idx_col]=np.arange(tracking_predictions_tags.shape[0]) # idx_i is set to be the order of predictions, which are as well ordered by the tracking score


                        dataloader.dataset.dataset.tf_tree.update_transform_data("map", "gps_prev", TV_UTM_E, TV_UTM_N, altitude,   0.0, pitch, heading)
                        dataloader.dataset.dataset.dDist=dDist
                        dataloader.dataset.dataset.dHeading=dHeading
                        dataloader.dataset.dataset.dPitch=dPitch
                        dataloader.dataset.dataset.dRoll=dRoll
                        dataloader.dataset.dataset.UTM_E=UTM_E
                        dataloader.dataset.dataset.UTM_N=UTM_N


            #     dataloader.dataset.dataset.tf_tree
            # self.tf_tree


                pred_lanes=[]
                pred_lanes_ordering=[]

                if prediction is not None:

                    """
                  img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img_paint, pred=prediction[0],\
                        save_predictions=save_predictions,bins=False)
                    cv2.imshow('pred', img)
                    cv2.waitKey(0)
                    breakpoint()
                    """

                    self.logger.trace(bcolors.OKGREEN+'prediction (len): ' +bcolors.ENDC+str(len(prediction)))
                    prediction_=prediction[0] # label infdo
                    if len(prediction_)>0 and initialization:

                        " def __init__(self, n_lanes, proc_noise_scale, meas_noise_scale)"
                        # laneTracker = LaneTracker(len(prediction_), 0.1, 500)
                        initialization=False # being initialized


                    trackedLanes=np.empty((3,3))* np.nan # if not, it initializes it with kind of random values
                    if not tracking_init:
                        old_trackedLanes=copy.deepcopy(trackedLanes)


                    # No point cloud -> Using the same plane to project into Lidar as before
                    # it is just to break perspective and improve tracking

                    if not point_cloud_exist:
                        plane_points_projected_img=np.array(plane_points_projected_img_old,copy=True)
                        plane_points=np.array(plane_points_old,copy=True)


                    self.logger.trace(bcolors.OKGREEN+'Iteration n: ' +bcolors.ENDC+str(idx))
                    indexes_3d=dataloader.dataset.Detectionto3d(plane_points_projected_img,img_paint,\
                    sem_img,copy.deepcopy(prediction_),plot=True)


                    # 0. RETRIEVE LIDAR 3D DETECTIONS
                    predictions_3d=[]
                    for index_3d in indexes_3d:
                        prediction_3d=plane_points[index_3d[:,0],:3]# xyz (no ones)
                        prediction_3d=np.append(prediction_3d,index_3d[:,1].reshape(-1,1),axis=1) # npts x 4
                        if prediction_3d.shape[0]>0: # only predictions with data points
                            predictions_3d.append(prediction_3d)



                    trackedLanes=np.empty((len(predictions_3d),3))* np.nan # if not, it initializes it with kind of random values

                    if len(predictions_3d)==0:
                        """HARDCODED"""
                        # breakpoint()

                        continue

                    # 1. RETRIEVE INTERVAL OF DATA
                    # lidar2img relationships y(img)-> x(lidar)
                    dim=0
                    other_dim=1
                    self.logger.trace(bcolors.OKGREEN+'predictions_3d:\n' +bcolors.ENDC+str(predictions_3d))

                    for idx_i,lane in enumerate(predictions_3d):
                        points=predictions_3d[idx_i]

                        if idx_i==0:
                            intervals=pyinterval[points[0,dim],points[-1,dim]]
                        else:
                            intervals = intervals & pyinterval[points[0,dim],points[-1,dim]]

                        self.logger.trace(bcolors.OKGREEN+'intervals: ' +bcolors.ENDC+str(intervals))


                    # 2. GET METRICS TO ORDER POLYLINES FROM RIGHT TO LEFT
                    if len(intervals)>0:
                        middle_point=0.5*(intervals[0][1]+intervals[0][0])
                        self.logger.trace(bcolors.OKGREEN+'middle_point: ' +bcolors.ENDC+str(middle_point))

                    for idx_i in range(len(predictions_3d)):
                        # pred_lanes.append(prediction_[idx_i].points) # van ordenados de menor a mayor en y / de mas lejos en pantalla a mas cerca en pantalla
                        points=predictions_3d[idx_i] # Lane points / Lidar coordinate system
                        self.logger.trace(bcolors.OKGREEN+'points: \n' +bcolors.ENDC+str(points))

                        if len(intervals)>0:
                            min_idx=np.argmin(np.absolute(points[:,dim]-middle_point))
                            trackedLanes[idx_i,score_col]=points[min_idx,other_dim]
                        else:
                            trackedLanes[idx_i,score_col]=np.mean(points[:,other_dim])

                        trackedLanes[idx_i,score_col]=points[0,other_dim]

                # 3. ORDER LANES -> according to their tracking score (y)
                    sort_idxs=np.argsort(trackedLanes[:,score_col])
                    self.logger.trace(bcolors.OKGREEN+'sort_idxs: ' +bcolors.ENDC+str(sort_idxs))

                    predictions_3d=[predictions_3d[sort_idx] for sort_idx in sort_idxs]
                    trackedLanes=trackedLanes[sort_idxs,:] #tags, idx_i, score
                    trackedLanes[:,pred_idx_col]=np.arange(trackedLanes.shape[0]) # idx_i is set to be the order of predictions, which are as well ordered by the tracking score

                    self.logger.trace(bcolors.OKGREEN+'pred_lanes_ordering: ' +bcolors.ENDC+str(pred_lanes_ordering))
                    self.logger.info(bcolors.OKGREEN+'label: ' +bcolors.ENDC+str(label))


                    # GET TRACKING ORDER
                    if not tracking_init:
                        tracking_init=True
                        old_trackedLanes=copy.deepcopy(trackedLanes)

                        # Tags shall be initialized
                        for idx_i in range(len(predictions_3d)):
                            old_trackedLanes[idx_i,tag_col]=idx_i
                            old_trackedLanes[idx_i,pred_idx_col]=idx_i

                        # tag, idx_i, track_id, score,pred_counter,track_counter
                        tracking_LUT=copy.deepcopy(old_trackedLanes)
                        tracking_LUT = np.insert(tracking_LUT, 2, np.nan, axis=1)

                        tracking_LUT = np.append(tracking_LUT, np.zeros((tracking_LUT.shape[0],1)),axis=1)
                        tracking_LUT = np.append(tracking_LUT, np.zeros((tracking_LUT.shape[0],1)),axis=1)

                    self.logger.trace(bcolors.OKGREEN+'old_trackedLanes:\n' +bcolors.ENDC+str(old_trackedLanes))
                    self.logger.trace(bcolors.OKGREEN+'trackedLanes:\n' +bcolors.ENDC+str(trackedLanes))


                    self.logger.trace(bcolors.OKGREEN+'old_trackedLanes (before): ' +bcolors.ENDC+str(old_trackedLanes))
                    # HARDCODED
                    # if cam_num==1653480040601:
                    #     breakpoint()

                    trackedLanes=associate_elements(old_trackedLanes, trackedLanes,2.0)


                    # RESET TRACK_ID COL OF tracking_LUT
                    tracking_LUT[:,2]=np.nan


                    # UPDATE THE INFORMATION OF OLD_TRACKEDLANES TO INCLUDE OLD DETECTIONS WITH THE NEW ONES
                    for idx_i in range(trackedLanes.shape[0]):
                        tag=trackedLanes[idx_i,tag_col]
                        # self.logger.trace(bcolors.OKGREEN+'tag (important): ' +bcolors.ENDC+str(tag))
                        find_idx=np.where(old_trackedLanes[:,tag_col]==tag)[0]

                        if find_idx.shape[0]>0:
                            """
                            TrackedLanes ->  # tag,pred_idx,score
                            tracking_LUT -> # tag, idx_i, track_id, score,pred_counter,track_counter
                            """

                            old_trackedLanes[find_idx,:]=trackedLanes[idx_i,:]

                            tracking_LUT[find_idx,tag_col]=trackedLanes[idx_i,tag_col] # update tag
                            tracking_LUT[find_idx,pred_idx_col]=trackedLanes[idx_i,pred_idx_col] # update pred_idx
                            tracking_LUT[find_idx,3]=trackedLanes[idx_i,2] # update score

                        else:
                            old_trackedLanes=np.append(old_trackedLanes,trackedLanes[idx_i,:].reshape(1,-1),axis=0)

                            new_row=[trackedLanes[idx_i,tag_col],trackedLanes[idx_i,pred_idx_col],np.nan,
                            trackedLanes[idx_i,score_col],1,0]
                            tracking_LUT=np.append(tracking_LUT,np.array(new_row).reshape(1,-1),axis=0)

                    self.logger.trace(bcolors.OKGREEN+'old_trackedLanes:\n' +bcolors.ENDC+str(old_trackedLanes))
                    self.logger.trace(bcolors.OKGREEN+'trackedLanes:\n' +bcolors.ENDC+str(trackedLanes))


                    for row_id,tag in enumerate(trackedLanes[:,tag_col]): # tag, track_id, score
                        if not np.isnan(tag):
                            old_idx=np.where(tracking_LUT[:,tag_col]==tag)[0][0] # should be in oldtracked
                            tracking_LUT[old_idx,2]=-1 # track_id



                    """
                    Track_id column:
                    -1 -> polyline detected; do not use tracking
                    number -> use tracking
                    nan -> There is no info for that poly either way (its impossible btw )
                    """

                    # Determine which info might be filled using the tracking
                    if "tracking_predictions_tags" in locals():
                        tracking_predictions_tags=associate_elements(old_trackedLanes, tracking_predictions_tags,2.0)
                        self.logger.trace(bcolors.OKGREEN+'tracking_predictions_tags:\n' +bcolors.ENDC+str(tracking_predictions_tags))

                        ids=[]

                        for row_id,tag in enumerate(tracking_predictions_tags[:,tag_col]): # tag, track_id, score
                            old_idxs=np.where(old_trackedLanes[:,tag_col]==tag)[0]
                            current_idxs=np.where(trackedLanes[:,tag_col]==tag)[0]
                            # Filter tracking_prediction_tags to instances that are
                            # present in old_trackedLanes but not in TrackedLanes

                            if len(old_idxs)>0 and len(current_idxs)==0:
                                ids.append(tag)
                                old_idx=old_idxs[0] # only one instance
                                tracking_LUT[old_idx,2]= tracking_predictions_tags[row_id,1] # tracking id column
                                self.logger.trace(bcolors.OKGREEN+'tracking_LUT:\n' +bcolors.ENDC+str(tracking_LUT))

                        self.logger.trace(bcolors.OKGREEN+'tags: ' +bcolors.ENDC+str(ids))

                    self.logger.trace(bcolors.OKGREEN+'tracking_LUT:\n' +bcolors.ENDC+str(tracking_LUT))


                    if not point_cloud_exist:

                        """
                        If there is no point cloud, then trackingLUT will only have the information of the tracking,
                        as there are no detections
                        """

                    """
                    trackedLanes=associate_elements(array_ref, array_query,2.0)
                    array_query has values but do not have tag for their values
                    """


                    # pred_lanes_ordering.append(points[-1,0]) # Pegado al coche
                    # pred_lanes_ordering.append(points[0,0]) # Lejos del coche


                    """
                    if trackedLanes is not None:
                        laneTracker.update(trackedLanes)

                        self.logger.trace(bcolors.OKGREEN+'pred_lanes_angle: ' +bcolors.ENDC+str(pred_lanes_angle))
                        self.logger.trace(bcolors.OKGREEN+'pred_lanes_ordering: ' +bcolors.ENDC+str(pred_lanes_ordering))
                    """

                    # pred_lanes_angle=np.array(pred_lanes_angle)
                    # sort_idxs=np.argsort(pred_lanes_angle).astype("int")

                    # IF PREDICTION IS NOT NONE

                    # POLYLINES2LANES
                    # 0. Resample each of the detections
                    tracked_coeffs={} # dictionary of tags
                    polylines_with_tracking=[]

                    for idx_i in range(len(tracking_LUT)):
                        # check whether to pick from tracking or predictions
                        if tracking_LUT[idx_i,2]==-1: # pick from predictions
                            pred_idx=int(tracking_LUT[idx_i,1])
                            new_poly,tracked_coeff= resample_laneline(predictions_3d[pred_idx],dims=["ref","num","num","cat"])

                        elif np.isnan(tracking_LUT[idx_i,2]): # there is a gap that cannot be covered by anyone
                            # Use previous coeffs
                            tracked_coeff=old_tracked_coeffs[tracking_LUT[idx_i,tag_col]]
                            self.logger.trace(bcolors.OKGREEN+'x_samples: \n' +bcolors.ENDC+str(x_samples))
                            new_poly=resample_laneline_with_coeffs(tracked_coeff , np.arange(5,50,1))

                        else: # pick from tracking
                            """"
                            En lugar de pillar los coeffs, pillar las tracking_predictions directamente ?
                            """
                            track_idx=int(tracking_LUT[idx_i,2])
                            # 1653479931900
                            self.logger.trace(bcolors.OKGREEN+'tracking_prediction: \n' +bcolors.ENDC+str(tracking_predictions[track_idx]))
                            new_poly,tracked_coeff= resample_laneline(tracking_predictions[track_idx],dims=["ref","num","num"])
                            self.logger.trace(bcolors.OKGREEN+'new_poly: \n' +bcolors.ENDC+str(new_poly))

                            """
                            tracking_predictions= predict_equation_V3(old_tracked_coeffs,lidar_prev2lidar_tf, x_samples)
                            already contains tracking polylines.
                            new_poly is re-calculated the coeffs, and resampled in another coeffs
                            """
                        tracked_coeffs[tracking_LUT[idx_i,tag_col]]=tracked_coeff
                        # tracked_coeffs.append(tracked_coeff)
                        polylines_with_tracking.append(new_poly)

                    old_tracked_coeffs=copy.deepcopy(tracked_coeffs)

                    # 1. ASSOCIATE POLYLINES TOGETHER, WITH TRACKING INFORMATION
                    # Filter polylines that have been at least detected a certain number of times
                    robust_detection_thr=3

                    # pred_counter
                    cand_idx=np.where(tracking_LUT[:,4]>=robust_detection_thr)[0]
                    self.logger.trace(bcolors.OKGREEN+'cand_idx: ' +bcolors.ENDC+str(cand_idx))

                    polyline_candidates=[polylines_with_tracking[idx_i] for idx_i in cand_idx]
                    tracking_LUT_cand=tracking_LUT[cand_idx,:]

                    lanes=associate_polylines_with_tracking(polyline_candidates,tracking_LUT_cand,thr_dist=[1.8,5.0], dims=["ref","query","num","cat"])

                    # lanes=associate_polylines_with_tracking(polylines_with_tracking,tracking_LUT,thr_dist=[2.5,5.0], dims=["ref","query","num","cat"])
                    self.logger.trace(bcolors.OKGREEN+'lanes: ' +bcolors.ENDC+str(lanes))

                    # Save plane points for next iteration
                    plane_points_projected_img_old=np.array(plane_points_projected_img,copy=True)
                    plane_points_old=np.array(plane_points,copy=True)


                    for idx_i in range(tracking_LUT.shape[0]):
                        if tracking_LUT[idx_i,2]==-1: # pick from predictions
                            tracking_LUT[idx_i,4]+=1 # pred_counter
                            tracking_LUT[idx_i,5]=0 # tracking counter (set to 0)
                        else:
                            tracking_LUT[idx_i,5]+=1 # tracking tracking

                    self.logger.trace(bcolors.OKGREEN+'tracking_LUT:\n' +bcolors.ENDC+str(tracking_LUT))


                    if save_local_info:
                        dataloader.dataset.save_local_info(idx, predictions_3d,trackedLanes,lanes)


                    if (save_global_info) & (point_cloud_exist):
                        predictions_3d_map=[]
                        transform = dataloader.dataset.dataset.tf_tree.lookup_transform("map", "lidar")

                        self.logger.trace(bcolors.OKGREEN+'gps_file: ' +bcolors.ENDC+str(gps_num))
                        self.logger.trace(bcolors.OKGREEN+'transform (lidar2map):\n' +bcolors.ENDC+str(transform))


                        for idx_i in range(len(predictions_3d)):
                            lidar_coords=np.append(predictions_3d[idx_i][:,0:3],np.ones((predictions_3d[idx_i].shape[0],1)),axis=1).T # 4x npts (h)
                            map_coords=np.matmul(transform,lidar_coords).T # (4x4)

                            self.logger.debug(bcolors.OKGREEN+'map_coords:\n' +bcolors.ENDC+str(map_coords))
                            map_coords=Homography2Cart(map_coords)
                            self.logger.debug(bcolors.OKGREEN+'map_coords:\n' +bcolors.ENDC+str(map_coords))

                            map_coords=np.append(map_coords,predictions_3d[idx_i][:,3:],axis=1)
                            predictions_3d_map.append(map_coords)

                        dataloader.dataset.save_global_info(idx, predictions_3d_map,tracking_LUT,lanes)
                        """tracking_LUT -> # tag, idx_i, track_id, score,pred_counter,track_counter"""

                    if (point_cloud_exist) and ( (save_point_cloud) or (save_csv)) :
                        point_cloud_iter=np.empty((0,len(pcb_fields)))

                        """ pcb_fields=["x","y","z","type","poly_id","tracking_id","iter","weight"]"""
                        for idx_i in range(len(predictions_3d_map)):
                            pcb_points=predictions_3d_map[idx_i] # npts, 4

                            poly_id_col=np.ones((predictions_3d_map[idx_i].shape[0],1))*poly_id
                            pcb_points=np.append(pcb_points,poly_id_col,axis=1)

                            # tag_col
                            tracking_id_col=np.ones((predictions_3d_map[idx_i].shape[0],1))*tracking_LUT[idx_i,tag_col]
                            pcb_points=np.append(pcb_points,tracking_id_col,axis=1)

                            iter_col=np.ones((predictions_3d_map[idx_i].shape[0],1))*iter_n # iteration
                            pcb_points=np.append(pcb_points,iter_col,axis=1)

                            # This metric could be re-measured (cause predictions are made in distinct intervals)
                            # weighting should be local lidar x coordinate

                            # weight_col=np.arange(0,predictions_3d_map[idx_i].shape[0],1).reshape(-1,1)
                            weight_col=(predictions_3d[idx_i][:,0]).reshape(-1,1)
                            pcb_points=np.append(pcb_points,weight_col,axis=1)
                            poly_id+=1

                            point_cloud_iter=np.append(point_cloud_iter,pcb_points,axis=0)

                    if save_csv:
                        dataloader.dataset.save_csv(idx, point_cloud_iter,pcb_fields)


                    # Save point cloud in each iteration
                    if save_point_cloud:
                        """ pcb_fields=["x","y","z","type","poly_id","tracking_id","iter","weight"]"""

                        # Accumulate point cloud across iterations and save
                        # time and resource consumming
                        point_cloud=np.append(point_cloud,point_cloud_iter,axis=0)

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])

                        """
                        POINT CLOUD COLORS -------------------------
                        for col_idx in range(colors.shape[1]):
                            colors[:,col_idx]=255*(colors[:,col_idx]-np.min(colors[:,col_idx]))/(np.max(colors[:,col_idx])-np.min(colors[:,col_idx]))

                        colors = o3d.utility.Vector3dVector(colors)

                        # colors=np.append(colors,np.ones((colors.shape[0],1)),axis=1)
                        pcd.colors=colors
                        """

                        output_dir=dataloader.dataset.dataset.saving_dir
                        os.makedirs(output_dir, exist_ok=True)
                        pcb_path=os.path.join(output_dir,"point_cloud.ply")
                        self.logger.trace (bcolors.OKGREEN + "Saving point cloud in : " + bcolors.ENDC+ pcb_path)

                        # Save the point cloud to a file
                        o3d.io.write_point_cloud(pcb_path, pcd)

                        df = pd.DataFrame(point_cloud, columns=pcb_fields)
                        pd_path=os.path.join(output_dir,"point_cloud.csv")
                        df.to_csv(pd_path)


                    # dataloader.dataset.save_global_info(idx, predictions_3d,trackedLanes,lanes)
                    # update color dictionary
                    for lane in lanes:
                        if not lane[0] in colors_by_lane:
                            colors_by_lane[lane[0]]=list(np.random.choice(range(256), size=3))

                    if video:
                        img_paint=(np.array(img,copy=True)*255).astype("uint8") # Torch to numpy array
                        img_paint=np.swapaxes(img_paint, 0, 1)
                        img_paint=np.swapaxes(img_paint, 1,2)

                        # tracking_predictions existe? si pero ya esta unida con polylines_with tracking
                        dataloader.dataset.save_frames_with_tracking(idx,img_paint,polylines_with_tracking,tracking_LUT,\
                            lanes,colors_by_lane,tracked_polys_lidar=tracking_predictions,point_cloud_exist=point_cloud_exist)



                    # ERASE TRACKING INSTANCES THAT HAVE NO PREDICTIONS IN LONG TIME
                    noInfo_thr=6
                    idxs=np.where(tracking_LUT[:,5]>=noInfo_thr)[0]
                    tags_del=tracking_LUT[idxs,tag_col]
                    self.logger.trace(bcolors.OKGREEN+'idxs: ' +bcolors.ENDC+str(idxs))
                    self.logger.trace(bcolors.OKGREEN+'tags_del: ' +bcolors.ENDC+str(tags_del))

                    for tag in tags_del:
                        old_tracked_coeffs.pop(tag) # delete those tracking predictions
                        old_idx=np.where(tracking_LUT[:,tag_col]==tag)[0][0] # should be in oldtracked
                        tracking_LUT = np.delete(tracking_LUT, (old_idx), axis=0)


                        old_idx=np.where(old_trackedLanes[:,tag_col]==tag)[0][0] # should be in oldtracked
                        old_trackedLanes = np.delete(old_trackedLanes, (old_idx), axis=0)




                    self.logger.trace(bcolors.OKGREEN+'tracking_LUT:\n' +bcolors.ENDC+str(tracking_LUT))

                    """
                        TrackedLanes ->  # tag,pred_idx,score
                        tracking_LUT -> # tag, idx_i, track_id, score,pred_counter,track_counter
                    """






        #             if self.view:
        #                 img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        #                 img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0],\
        #                     save_predictions=save_predictions,bins=bins)
        #                 if self.view == 'mistakes' and fp == 0 and fn == 0:
        #                     continue

        #                 if not save_predictions:
        #                     cv2.imshow('pred', img)
        #                     cv2.waitKey(0)

        # if save_predictions:
        #     with open('predictions.pkl', 'wb') as handle:
        #         pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)

    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset('train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.cfg['batch_size'],
                                                   shuffle=True,
                                                   num_workers=8,
                                                   worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=8,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
