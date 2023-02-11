import cv2
import numpy as np
import glob,os,sys
import argparse
from tqdm import tqdm
import pandas as pd

# python video_create.py --dir=/raid/sequences/extracted_boden_seq2/1/data/lanemarkings/workspace/lanemarking_detection/frames --output_dir=/raid/sequences/extracted_boden_seq2/1/data/lanemarkings/workspace/lanemarking_detection/frames --extension=png
# python video_create.py --dir=/home/javpasto/Documents/LaneDetection/LaneATT/experiments/laneatt_r18_tusimple/results/predictions --output_dir=/home/javpasto/Documents/LaneDetection/LaneATT/experiments/laneatt_r18_tusimple/results --extension=png

# export OUTPUT_DIR="/home/javpasto/Documents/LaneDetection/LaneATT/documentation/tracking/tracking_inf/prueba_01"
# export OUTPUT_DIR="/raid/sequences/extracted_boden_seq2/0/data/lanemarkings/workspace/lanemarking_detection/frames"
# echo ${OUTPUT_DIR}
# python video_create.py --dir="${OUTPUT_DIR}" --output_dir="${OUTPUT_DIR}" --extension="png"


def parse_args():
    parser = argparse.ArgumentParser(description="Create a video from its frames ")
    # parser.add_argument("dir", choices=["train", "test"], help="Train or test?")
    parser.add_argument("--dir", type=str, help="Directory where frames are stored", required=True)
    parser.add_argument("--output_dir", type=str,help="Ouput directory of video", required=True)
    parser.add_argument("--extension",type=str, choices=["jpg", "png"], default="jpg", help="Extension of files")
    args = parser.parse_args()
    return args

args=parse_args()
output_dir=args.output_dir
os.makedirs(output_dir, exist_ok=True)
video_path=os.path.join(output_dir,"video.avi")



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

dir=args.dir
extension=args.extension

dir_query=os.path.join(dir,"*."+extension)
print("dir_query: ",dir_query)

filenames=sorted(glob.glob(dir_query))

# Retrieve image dimensions
img=cv2.imread(filenames[0])
width=img.shape[1]
height=img.shape[0]
print("width: ",width)
print("height: ",height)


# # resize
# width=1280
# height=720


video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))

for filename in tqdm(filenames):
    # print(filename)
    img = cv2.imread(filename)
    img = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)
    video.write(img)

cv2.destroyAllWindows()
video.release()
