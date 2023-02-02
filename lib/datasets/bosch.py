import glob,os,sys
import logging
from .lane_dataset_loader import LaneDatasetLoader
from utils.openlane_utils import bcolors,getRotation
import numpy as np
import math
import cv2
import pandas as pd
from tfmatrix import transformations

class Bosch(LaneDatasetLoader):
    def __init__(self,split="test", img_h=720, img_w=1280, max_lanes=None, root=None, img_ext='.jpg', **_):
        """Use this loader if you want to test a model on an image without annotations or implemented loader."""
        self.logger = logging.getLogger(__name__)

        self.root = root
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

        resize_ratio=2
        scale_factor_projection=np.array([
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0,   0, 1]])

        P=cam_intrinsics
        P = np.matmul(scale_factor_projection,P)
        self.logger.trace (bcolors.OKGREEN + "P:\n" + bcolors.ENDC+ str(P))

        P=cam_intrinsics
        P /= resize_ratio
        P[2,2]*=resize_ratio

        self.logger.trace (bcolors.OKGREEN + "P:\n" + bcolors.ENDC+ str(P))
        breakpoint()

        self.P=P


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
        P_lidar2img = np.matmul(np.transpose(self.vel2cam_M),np.transpose(self.P))
        self.P_lidar2img=P_lidar2img




        # 2. gps2lidar
        gps2lidar_path=os.path.join(self.calib_path,"calib_gps_lidar.txt")

        if os.path.isfile(gps2lidar_path):
            self.logger.trace (bcolors.OKGREEN + "Adding calib (lidar2gps) from file" + bcolors.ENDC)
            self.tf_tree.add_transform_from_file("gps", "lidar",gps2lidar_path )
        else:
            self.logger.trace (bcolors.OKGREEN + "Adding calib (lidar2gps) manually" + bcolors.ENDC)
            self.tf_tree.add_transform_data("gps", "lidar",-1.64584,0.0,-0.8,   0.0, -0.08, -0.08)


        transform = self.tf_tree.lookup_transform("gps", "lidar")
        self.logger.trace (bcolors.OKGREEN + "tf (lidar2gps):\n" + bcolors.ENDC+ str(transform))



        breakpoint()



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



        if root is None:
            raise Exception('Please specify the root directory')

        self.split = split
        self.img_w, self.img_h = img_w, img_h
        self.img_ext = img_ext
        self.annotations = []
        self.load_annotations()


        # Force max_lanes, used when evaluating testing with models trained on other datasets
        # On NoLabelDataset, always force it
        self.max_lanes = max_lanes

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
        for file in glob.glob(pattern, recursive=True):
            self.annotations.append({'lanes': [], 'path': file})

    def ReadSyncFile(self):
        sync_path=self.sync_path
        # Contains the sync text id between files
        sync=["lidar","cam0","cam1","cam2","gps_old","radar","gps"]
        self.sync = pd.read_csv(sync_path, sep=" ", names=sync)
        print(self.sync.head())


    def UseSyncFile(self,from_file,to_file,file):
        row=self.sync.loc[self.sync[from_file] ==float(file)]
        # print("\n Row type ",type(row))
        # print("\ndf row: ",row)
        # print("\nDesired text file: ",row[to_file])

        # print(type(row[to_file]))
        # print(type(row[to_file].values))
        # print((row[to_file].values))

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
