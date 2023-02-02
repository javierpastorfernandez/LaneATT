import cv2
import numpy as np
import glob,os,sys
import argparse
from tqdm import tqdm

# python video_create.py --dir=/home/javpasto/Documents/LaneDetection/LaneATT/experiments/laneatt_r18_tusimple/results/predictions --output_dir=/home/javpasto/Documents/LaneDetection/LaneATT/experiments/laneatt_r18_tusimple/results --extension=png
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
