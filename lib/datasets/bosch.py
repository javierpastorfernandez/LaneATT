import glob,os,sys
import logging
from .lane_dataset_loader import LaneDatasetLoader
from utils.openlane_utils import bcolors,getRotation
import numpy as np
import math
import cv2
import pandas as pd
from tfmatrix import transformations
from utils.projections_utils import SampleFromPlane,Homography2Cart,DrawPoints,rescale_projection


class Bosch(LaneDatasetLoader):
    def __init__(self,split="test",img_org_size=(2000,4000),img_resize_size=(2000,4000),sem_size=(2000,4000), max_lanes=None, root=None, img_ext='.jpg', **_):
        """Use this loader if you want to test a model on an image without annotations or implemented loader."""
        self.logger = logging.getLogger(__name__)
        img_size=img_resize_size

        self.root = root

        self.dHeading=0
        self.dDist=0
        self.dDist=0
        self.dHeading=0
        self.dPitch=0
        self.dRoll=0

        if root is None:
            raise Exception('Please specify the root directory')

        self.split = split
        self.img_w, self.img_h =img_size[1],img_size[0]
        self.img_org_w, self.img_org_h = img_org_size[1],img_org_size[0]
        self.sem_w, self.sem_h =sem_size[1],sem_size[0]

        self.img_ext = img_ext


        # Force max_lanes, used when evaluating testing with models trained on other datasets
        # On NoLabelDataset, always force it
        self.max_lanes = max_lanes

        chunks=root.split("/")
        self.logger.trace (bcolors.OKGREEN + "chunks:" + bcolors.ENDC+ str(chunks))
        self.tf_tree = transformations.StaticTransformer()

        calib_path=""
        sync_path=""

        for i in range(1,len(chunks)-1):
            chunk=chunks[i]
            sync_path+="/"+chunk
            if (i<=len(chunks)-4):
                calib_path+="/"+chunk
        sequence_path=sync_path

        calib_path+="/"+"calibs"
        sync_path+="/"+"sync.txt"

        self.calib_path=calib_path
        self.sync_path=sync_path
        self.sequence_path=sequence_path
        self.gps_dir=sequence_path+"/oxts_INS_filtered"
        self.saving_dir=sequence_path+"/lanemarkings/workspace/lanemarking_detection"

        os.makedirs(self.saving_dir,exist_ok=True)

        self.logger.trace (bcolors.OKGREEN + "saving_dir:" + bcolors.ENDC+ str(self.saving_dir))
        self.logger.trace (bcolors.OKGREEN + "Bosch sequence path:" + bcolors.ENDC+ str(root))
        self.logger.trace (bcolors.OKGREEN + "calib_path" + bcolors.ENDC+ str(self.calib_path))
        self.logger.trace (bcolors.OKGREEN + "sync_path" + bcolors.ENDC+ str(self.sync_path))
        self.logger.trace (bcolors.OKGREEN + "sequence_path" + bcolors.ENDC+ str(self.sequence_path))
        self.logger.trace (bcolors.OKGREEN + "gps_dir" + bcolors.ENDC+ str(self.gps_dir))

        self.ReadSyncFile()



        # RETRIEVE CALIBRATION MATRICES FROM SEQUENCE
        # 1. Intrinsics
        instrinsics_path=os.path.join(self.calib_path,"calib_cam_to_cam0.txt")
        with open(instrinsics_path,"r") as f:
            line=f.readline()

            # REFERNCE CAMERA COORD -> RECTIFIED CAMERA COORD -> IMAGE2 COORD

            # P_rect_00::  Retrieve the ROTATION from reference camera coord to rect camera coord
            # P_rect_02::  Retrieve the projection matrix from rect camera coord to image2 coord
                # intrinsecos -->

            line=f.readline().split(": ")[1].split()
            self.logger.trace (bcolors.OKGREEN + "line:" + bcolors.ENDC+ str(line))
            cam_intrinsics=np.array([float(value)for value in line]).reshape(-1,4) # 3x4

        self.cam_intrinsics=cam_intrinsics # 3x4
        self.logger.trace (bcolors.OKGREEN + "cam_intrinsics:\n" + bcolors.ENDC+ str(cam_intrinsics))

        P=cam_intrinsics
        self.P_img=np.array(P,copy=True)


        P_img_resize=np.array(P,copy=True)
        P_img_resize=rescale_projection(img_org_size,img_size,P_img_resize)
        self.P_img_resize=P_img_resize

        P_sem=np.array(P,copy=True)
        P_sem=rescale_projection(img_org_size,sem_size,P_sem)
        self.P_sem=P_sem

        self.logger.trace (bcolors.OKGREEN + "P_img:\n" + bcolors.ENDC+ str(self.P_img))
        self.logger.trace (bcolors.OKGREEN + "P_img_resize:\n" + bcolors.ENDC+ str(self.P_img_resize))
        self.logger.trace (bcolors.OKGREEN + "P_sem:\n" + bcolors.ENDC+ str(self.P_sem))

        """
        cv::Mat presult = velo2cam.t() * P.t();
        result = point3D.t() * presult;
        result = result / result.at<double>(0,2);

        cv::Point point2D;
        point2D = cv::Point((int)result.at<double>(0,0),(int)result.at<double>(0,1));  //Point corresponding to semantic image

        """

        # 2. VeltoCam -> LIDAR TO REFERENCE CAMERA COORDINATE
        vel2cam_path=os.path.join(self.calib_path,"calib_velo_to_cam0.txt")
        with open(vel2cam_path,"r") as f:
            line=f.readline().split(": ")[1].split()
            vel2cam_r=np.array([float(value)for value in line]).reshape(3,3)
            self.logger.trace (bcolors.OKGREEN + "vel2cam (rotation):\n" + bcolors.ENDC+ str(vel2cam_r))

            line=f.readline().split(": ")[1].split()
            vel2cam_t=np.array([float(value)for value in line]).reshape(3,1)
            self.logger.trace (bcolors.OKGREEN + "vel2cam (translation):\n" + bcolors.ENDC+ str(vel2cam_t))

            vel2cam_M=np.append(vel2cam_r,vel2cam_t,axis=1)
            vel2cam_M=np.append(vel2cam_M,[[0,0,0,1]],axis=0) # 4x4 -> Esto esta bien!
            self.logger.trace (bcolors.OKGREEN + "vel2cam_M:\n" + bcolors.ENDC+ str(vel2cam_M))
        self.vel2cam_M=vel2cam_M # 4x4

        # 4x4 * 4*3 = 4*3
        P_lidar2img = np.matmul(np.transpose(self.vel2cam_M),np.transpose(self.P_img))
        # P_lidar2img_resize=np.matmul(self.P_lidar2img,np.transpose(scale_factor))d
        self.P_lidar2img=P_lidar2img


        P_lidar2img_resize = np.matmul(np.transpose(self.vel2cam_M),np.transpose(self.P_img_resize))
        self.P_lidar2img_resize=P_lidar2img_resize

        P_lidar2sem= np.matmul(np.transpose(self.vel2cam_M),np.transpose(self.P_sem))
        self.P_lidar2sem=P_lidar2sem


        self.logger.trace (bcolors.OKGREEN + "P_lidar2img:\n" + bcolors.ENDC+ str(self.P_lidar2img))
        self.logger.trace (bcolors.OKGREEN + "P_lidar2img_resize:\n" + bcolors.ENDC+ str(self.P_lidar2img_resize))
        self.logger.trace (bcolors.OKGREEN + "P_lidar2sem:\n" + bcolors.ENDC+ str(self.P_lidar2sem))


        # 2. gps2lidar
        gps2lidar_path=os.path.join(self.calib_path,"calib_gps_lidar.txt")

        if os.path.isfile(gps2lidar_path):
            self.logger.trace (bcolors.OKGREEN + "Adding calib (lidar2gps) from file" + bcolors.ENDC+str(gps2lidar_path))
            self.tf_tree.add_transform_from_file("gps", "lidar",gps2lidar_path )
            self.tf_tree.add_transform_from_file("gps_prev", "lidar_prev",gps2lidar_path )

        else:
            self.logger.trace (bcolors.OKGREEN + "Adding calib (lidar2gps) manually" + bcolors.ENDC)
            self.tf_tree.add_transform_data("gps", "lidar",-1.64584,0.0,-0.8,   0.0, -0.08, -0.08)
            self.tf_tree.add_transform_data("gps_prev", "lidar_prev",-1.64584,0.0,-0.8,   0.0, -0.08, -0.08)


        transform = self.tf_tree.lookup_transform("gps_prev", "lidar_prev")
        self.logger.trace (bcolors.OKGREEN + "tf (lidar_prev2gps_prev):\n" + bcolors.ENDC+ str(transform))

        transform = self.tf_tree.lookup_transform("gps", "lidar")
        self.logger.trace (bcolors.OKGREEN + "tf (lidar2gps):\n" + bcolors.ENDC+ str(transform))


        self.cam_height=0.5
        self.cam_roll=25

        R_g2c=getRotation((90-abs(self.cam_roll)),axis="roll",units="degrees")
        R_g2c=getRotation((5),axis="roll",units="degrees")

        self.H_g2im = np.matmul(cam_intrinsics[0:3,0:3], np.concatenate([R_g2c[:, 0:2], [[0], [self.cam_height], [0]]], 1))
        self.H_im2g = np.linalg.inv(self.H_g2im)

        top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])


        # The original dimensions of the PAPER / METHOD
        self.ipm_h = 208 # height of the original image
        self.ipm_w = 128 # width of the original image

        self.H_ipm2g = cv2.getPerspectiveTransform(np.float32([[0, 0],
                                                            [self.ipm_w-1, 0],
                                                            [0, self.ipm_h-1],
                                                            [self.ipm_w-1, self.ipm_h-1]]),
                                                                np.float32(top_view_region))


        self.H_im2ipm = np.linalg.inv(np.matmul(self.H_g2im, self.H_ipm2g))


        # LOAD ANNOTATIONS
        self.annotations = []
        self.load_annotations()


    def get_img_heigth(self, _):
        return self.img_h

    def get_img_width(self, _):
        return self.img_w

    def get_metrics(self, lanes, _):
        return 0, 0, [1] * len(lanes), [1] * len(lanes)

    def load_annotations(self):
        self.annotations = []
        pattern = '{}/**/*{}'.format(self.root, self.img_ext)
        print('Looking for image files with the pattern', pattern)


        # 1. RETRIEVE GPS FILENAMES AND SORT THEM
        files = glob.glob(pattern, recursive=True)
        files.sort()


        # 2. USE FIRST FILE TO SET UTM_E0 AND UTM_NO
        # First file in the sync

        gps_num=self.sync.iloc[0]["gps"]
        gps_path=os.path.join(self.gps_dir,str(gps_num)+".txt")
        self.logger.trace(bcolors.OKGREEN+' First gps filename: ' +bcolors.ENDC+str(gps_num))
        self.logger.trace(bcolors.OKGREEN+' First gps gps_path: ' +bcolors.ENDC+str(gps_path))

        # LOAD GPS INFO
        with open(gps_path, "r") as f_gps:
            gpslines = f_gps.readlines()
            assert(len(gpslines)==1) # GPS FILE SHOULD CONTAIN ONLY ONE LINE
            gpslines=gpslines[0].split()

            UTM_E=float(gpslines[0])
            UTM_N=float(gpslines[1])

            self.UTM_E0=UTM_E
            self.UTM_N0=UTM_N



        """"
        #3. FILE TO WHICH START ->  HARDCODED

        file=1653480010001
        file=1653480029200
        # file=1653480040601

        files_filtered=sorted(os.listdir(self.root))
        files_filtered=np.array([int(file.split(".")[0]) for file in files_filtered])
        idx=np.where(files_filtered==file)[0]
        idx=idx-7
        # print("files_filtered: ",files_filtered)
        files=files[int(idx):]

        """

        # files=files[6730:]
        files=files[3700:]


        for file in files:
            self.annotations.append({'lanes': [], 'path': file})




    def ReadSyncFile(self):
        sync_path=self.sync_path
        # Contains the sync text id between files
        sync=["lidar","cam0","cam1","cam2","gps_old","radar","gps"]
        self.sync = pd.read_csv(sync_path, sep=" ", names=sync)
        print(self.sync.head())


    def UseSyncFile(self,from_file,to_file,file,option="hard"):
        row=self.sync.loc[self.sync[from_file] ==float(file)]


        if option=="soft":
            """Some frames contain incomplete information such as point clouds. We would like to search """
            if len(row)==0:
                row_idx=np.argmin(self.sync[from_file]-float(file))

                if row_idx>=2:
                    row=self.sync.iloc[(row_idx-2):(row_idx+2),:]  # Print information around found index
                else:
                    row=self.sync.iloc[0:(row_idx+2),:]  # Print information around found index

                self.logger.trace (bcolors.OKGREEN + "row:\n" + bcolors.ENDC+ str(row.head()))
                output_file='{:010d}'.format(self.sync.iloc[row_idx][to_file])
                self.logger.trace (bcolors.OKGREEN + "Input file:" + bcolors.ENDC+ str(file)+
                bcolors.OKGREEN + " output_file:" + bcolors.ENDC+ str(output_file))

                return output_file

        else:
            if len(row)==0: # empty frame for some reason; stop
                output_file=None
            else:
                output_file='{num:010d}'.format(num=row[to_file].values[0])


        return output_file

    def save_predictions(self,predictions,output_basedir):
        self.logger.trace (bcolors.OKGREEN + "output_basedir:" + bcolors.ENDC+ str(output_basedir))
        self.logger.trace (bcolors.OKGREEN + "nº of predictions:" + bcolors.ENDC+ str(len(predictions)))

        for idx_i in range(len(predictions)):
            prediction=predictions[idx_i]
            org_path=self.annotations[idx_i]["path"]
            self.logger.trace(bcolors.OKGREEN + "Image path: " + bcolors.ENDC+ str(org_path))


            filename=org_path.split("/")[-1]
            path=os.path.join(output_basedir,filename.split(".")[0]+".txt")
            self.logger.trace(bcolors.OKGREEN + "Prediction path: " + bcolors.ENDC+ str(path))

            # np.savetxt("points.txt", points, delimiter=",")

            # Write the list to a txt file

            with open(path, "w") as file:
                line="path: "+org_path
                file.write(line + "\n")

                for idx_j in range(len(prediction)):
                    lane=prediction[idx_j]


                    for coordinates in lane:
                        file.write(str(coordinates[0])+" "+str(coordinates[1])  + ", ")

                    file.write("\n")

                    self.logger.debug (bcolors.OKGREEN + "lane points (type) " +bcolors.ENDC+ str(type(lane.points)))
                    self.logger.debug (bcolors.OKGREEN + "lane points (shape) "+bcolors.ENDC+ str(len(lane.points)))
                    self.logger.debug (bcolors.OKGREEN + "lane points: " + bcolors.ENDC+ str(lane.points))
                # for attribute, value in vars(lane).items():
                #     self.logger.trace (bcolors.OKGREEN + str(attribute)+": " + bcolors.ENDC+ str(value))


    def eval(self, _, __, ___, ____, _____):
        return "", None

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
