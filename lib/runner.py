import pickle
import random
import logging
import os,glob,sys
import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
from utils.openlane_utils import bcolors,FormatAxes, get_cmap, create_trace_loglevel
from utils.tracking import LaneTracker
from utils.projections_utils import SampleFromPlane,Homography2Cart

from tfmatrix import transformations
import math
from interval import interval as pyinterval


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

        with torch.no_grad():
            for idx, (images, _, _) in enumerate(tqdm(dataloader)):

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


                dataset_name=dataloader.dataset.dataset_name
                self.logger.trace(bcolors.OKGREEN+'dataset_name: ' +bcolors.ENDC+str(dataset_name))
                gps_init=False


                if dataset_name=="bosch":
                    lidar_path = dataloader.dataset.dataset.sequence_path + "/pc_filtered/semantic/camera_0/" + str(filename) + ".pcd"
                    self.logger.trace(bcolors.OKGREEN+'lidar_path: ' +bcolors.ENDC+str(lidar_path))

                    pcd = o3d.io.read_point_cloud(lidar_path)
                    pcb_pts=np.array(pcd.points)
                    pcb_pts=np.append(pcb_pts,np.ones((pcb_pts.shape[0],1)),axis=1)


                    # Filter by height
                    pcb_pts_filtered_idxs=np.where(pcb_pts[:,2]<=-1.5)[0] # Points near floor; tupple size 1
                    pcb_pts_filtered=pcb_pts[pcb_pts_filtered_idxs,:]

                    filtered_pcd = o3d.geometry.PointCloud()
                    filtered_pcd.points = o3d.utility.Vector3dVector(pcb_pts_filtered[:,0:3])


                    # Obtain plane from the road
                    plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.01,
                                                            ransac_n=3,
                                                            num_iterations=1000)




                        # ax + by + cz +d=0
                        # z=(d-ax-by)/c

                    # plane_points=SampleFromPlane(plane_model,[0,50,-15,15],sampling_period=1) -> MARTA
                    plane_points=SampleFromPlane(plane_model,[0,103,-15,15],sampling_period=0.1)


                    breakpoint()
                    [a, b, c, d] = plane_model
                    print(f"Lidar plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

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


                    plane_points_projected=np.matmul(plane_points_camera,np.transpose(dataloader.dataset.dataset.P)) # dim,nPts *presult (4*3)= npts,3
                    plane_points_projected=Homography2Cart(plane_points_projected)
                    plane_points_projected=plane_points_projected[:,[0,1]]

                    # plane_points_projected=np.matmul(plane_points,dataloader.dataset.dataset.P_lidar2img) # dim,nPts *presult (4*3)= npts,3
                    # plane_points_projected[:,0]=plane_points_projected[:,0]/plane_points_projected[:,2]
                    # plane_points_projected[:,1]=plane_points_projected[:,1]/plane_points_projected[:,2]
                    # plane_points_projected=plane_points_projected[:,:2] # 0,1 -> x,y


                    breakpoint()
                    pcb_pts_projected=np.matmul(pcb_pts_filtered,dataloader.dataset.dataset.P_lidar2img) # dim,nPts *presult (4*3)= npts,3
                    pcb_pts_projected=Homography2Cart(pcb_pts_projected)


                    self.logger.debug(bcolors.OKGREEN+'pcb_pts_projected: ' +bcolors.ENDC+str(pcb_pts_projected))
                    outliers = np.setdiff1d(np.arange(len(filtered_pcd.points)), inliers)

                    pcb_pts_projected_inliers=pcb_pts_projected[inliers]
                    pcb_pts_projected_outliers=pcb_pts_projected[outliers]


                    # Visualize the point cloud
                    cam_num=dataloader.dataset.dataset.UseSyncFile("lidar","cam0",filename) # From lidar file to gps
                    sem_image_path = dataloader.dataset.dataset.sequence_path + "/semantic_images/camera_0/" + str(cam_num) + ".png"
                    self.logger.trace(bcolors.OKGREEN+'sem_image_path: ' +bcolors.ENDC+str(sem_image_path))

                    sem_img=cv2.imread(sem_image_path)
                    sem_img_paint=cv2.imread(sem_image_path)

                    idxs=np.where(  (pcb_pts_projected[:,0]>0)& (pcb_pts_projected[:,0]<sem_img.shape[0])  &
                    (pcb_pts_projected[:,1]>0)& (pcb_pts_projected[:,1]>sem_img.shape[1]))[0] # Points near floor; tupple size 1
                    # self.logger.trace(bcolors.OKGREEN+'idxs: ' +bcolors.ENDC+str(idxs))

                    # Radius of circle
                    radius = 5
                    # Line thickness of 2 px
                    thickness = 5

                    # pts, dim
                    color = (0, 255, 0) # Blue color in BGR
                    for x,y in pcb_pts_projected_inliers:
                        x=int(x)
                        y=int(y)
                        self.logger.debug(bcolors.OKGREEN+'x: ' +bcolors.ENDC+str(x)+
                        bcolors.OKGREEN+'y: ' +bcolors.ENDC+str(y))

                        sem_img_paint = cv2.circle(sem_img_paint, (x,y), radius, color, thickness)


                    color = (0,0,255) # Blue color in BGR
                    # pts, dim
                    for x,y in pcb_pts_projected_outliers:
                        x=int(x)
                        y=int(y)
                        self.logger.debug(bcolors.OKGREEN+'x: ' +bcolors.ENDC+str(x)+
                        bcolors.OKGREEN+'y: ' +bcolors.ENDC+str(y))

                        sem_img_paint = cv2.circle(sem_img_paint, (x,y), radius, color, thickness)

                    overlay = sem_img_paint.copy()

                    color = (50,50,50) # Blue color in BGR
                    # pts, dim

                    for x,y in plane_points_projected:
                        x=int(x)
                        y=int(y)
                        self.logger.debug(bcolors.OKGREEN+'x: ' +bcolors.ENDC+str(x)+
                        bcolors.OKGREEN+'y: ' +bcolors.ENDC+str(y))

                        overlay = cv2.circle(overlay, (x,y), radius, color, thickness)


                    alpha = 0.4  # Transparency factor.
                    sem_img_paint = cv2.addWeighted(overlay, alpha, sem_img_paint, 1 - alpha, 0) # Following line overlays transparent rectangle over the image


                    cv2.namedWindow("sem_img", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
                    cv2.imshow("sem_img", sem_img)# Show image

                    cv2.namedWindow("sem_img_paint", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
                    cv2.imshow("sem_img_paint", sem_img_paint)# Show image

                    cv2.waitKey(0)
                    cv2.destroyAllWindows()




                    breakpoint()
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


                    gps_file=dataloader.dataset.dataset.UseSyncFile("lidar","gps",filename) # From lidar file to gps
                    self.logger.trace(bcolors.OKGREEN+'gps_file: ' +bcolors.ENDC+str(gps_file))
                    breakpoint()

                    # LOAD GPS INFO
                    gps_path=os.path.join(dataloader.dataset.dataset.gps_dir,gps_file+".txt")
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

                    if not gps_init:
                        UTM_E0=UTM_E
                        UTM_N0=UTM_N
                        gps_init=True

                        dE=0
                        dN=0
                        dataloader.dataset.dataset.tf_tree.add_transform_data("map", "gps", dE, dN, altitude,   0.0, pitch, heading)
                    else:
                        dE = UTM_E - UTM_E0
                        dN = UTM_N - UTM_N0

                        # UPDATE TRANSFORM LIDAR2GPS
                        dataloader.dataset.dataset.tf_tree.update_transform_data("map", "gps", dE, dN, altitude,   0.0, pitch, heading)



            #     dataloader.dataset.dataset.tf_tree
            # self.tf_tree


            #         breakpoint()


                pred_lanes=[]
                pred_lanes_angle=[]
                pred_lanes_x=[]
                trackedLanes=[]


                if prediction is not None:
                    self.logger.trace(bcolors.OKGREEN+'prediction (len): ' +bcolors.ENDC+str(len(prediction)))
                    prediction_=prediction[0] # label infdo
                    if len(prediction_)>0 and initialization:
                        laneTracker = LaneTracker(len(prediction_), 0.1, 500)
                        initialization=False # being initialized

                    for idx_i,lane in enumerate(prediction_):
                        points = lane.points

                        if idx_i==0:
                            intervals=pyinterval[points[0,1],points[-1,1]]
                        else:
                            intervals = intervals & pyinterval[points[0,1],points[-1,1]]

                        self.logger.trace(bcolors.OKGREEN+'intervals: ' +bcolors.ENDC+str(intervals))

                    if len(intervals)>0:
                        middle_point=0.5*(intervals[0][1]+intervals[0][0])
                        self.logger.trace(bcolors.OKGREEN+'middle_point: ' +bcolors.ENDC+str(middle_point))
                    else:
                        middle_point=0

                    for idx_i,lane in enumerate(prediction_):

                        points = lane.points
                        pred_lanes.append(points) # van ordenados de menor a mayor en y / de mas lejos en pantalla a mas cerca en pantalla
                        self.logger.trace(bcolors.OKGREEN+'points: ' +bcolors.ENDC+str(points))


                        # 1. CALCULATE ANGLE OF POINTS
                        points_diff=np.gradient(points,axis=0)
                        spline_diff_mean=np.mean(points,axis=0)
                        median_angle=np.arctan2(spline_diff_mean[1], spline_diff_mean[0])
                        median_angle=median_angle*180/math.pi # in degrees
                        pred_lanes_angle.append(median_angle) # punto mas alejado al coche

                        lane_tracking=[]
                        lane_tracking.append(points[0,:])

                        if len(intervals)>0:
                            min_idx=np.argmin(np.absolute(points[:,1]-middle_point))
                            pred_lanes_x.append(points[min_idx,0])
                            lane_tracking.append(points[min_idx,:])
                        else:
                            pred_lanes_x.append(np.mean(points[:,0]))
                            lane_tracking.append(points[int(points.shape[0]/2),:]) # Middle of the polyline


                        trackedLanes.append(lane_tracking)

                            # pred_lanes_x.append(points[-1,0]) # Pegado al coche
                            # pred_lanes_x.append(points[0,0]) # Lejos del coche



                if trackedLanes is not None:
                    laneTracker.update(trackedLanes)

                    self.logger.trace(bcolors.OKGREEN+'pred_lanes_angle: ' +bcolors.ENDC+str(pred_lanes_angle))
                    self.logger.trace(bcolors.OKGREEN+'pred_lanes_x: ' +bcolors.ENDC+str(pred_lanes_x))

                # pred_lanes_angle=np.array(pred_lanes_angle)
                # sort_idxs=np.argsort(pred_lanes_angle).astype("int")

                sorted_array = sorted(pred_lanes_x)
                bins = [sorted_array.index(value) for value in pred_lanes_x]


                    # breakpoint()

                if filename==1653476296100:
                    breakpoint()

                self.logger.info(bcolors.OKGREEN+'label: ' +bcolors.ENDC+str(label))

                if self.view:
                    img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0],\
                        save_predictions=save_predictions,bins=bins)
                    if self.view == 'mistakes' and fp == 0 and fn == 0:
                        continue

                    if not save_predictions:
                        cv2.imshow('pred', img)
                        cv2.waitKey(0)

        if save_predictions:
            with open('predictions.pkl', 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)

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
