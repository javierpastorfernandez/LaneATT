import os
import json
import random
import logging
from tqdm import tqdm
import numpy as np
from utils.tusimple_metric import LaneEval
from utils.openlane_utils import bcolors,FormatAxes, get_cmap, create_trace_loglevel
from .lane_dataset_loader import LaneDatasetLoader

# ADDITIONAL DEPENDENCIES (OPENLANE)
import pandas as pd
import glob

print("ENTER IN OPENLANE IMPORTS")

class OpenLane(LaneDatasetLoader):
    def __init__(self, split='train',version=1.2, max_lanes=None, root=None,compute_dataset_meta=False):
        self.split = split
        self.version = version
        self.root = root
        self.logger = logging.getLogger(__name__)

        self.logger.trace (bcolors.OKGREEN + "data_split: " + bcolors.ENDC+ str(self.split))
        self.logger.trace (bcolors.OKGREEN + "version: " + bcolors.ENDC+ str(self.version))
        self.logger.trace (bcolors.OKGREEN + "compute_dataset_meta: " + bcolors.ENDC+ str(compute_dataset_meta))
        self.view_cols=["filename","segment","img_folder","data_split","case","version"]

        if root is None:
            raise Exception('Please specify the root directory')

        self.img_w, self.img_h = 1280, 720
        self.annotations = []

        # LOAD OPENLANE DATASET

        data_split=self.loadOpenLane_V2(compute=compute_dataset_meta)
        self.load_annotations(data_split)

        # Force max_lanes, used when evaluating testing with models trained on other datasets
        if max_lanes is not None:
            self.max_lanes = max_lanes


    def get_img_heigth(self, _):
        return 720

    def get_img_width(self, _):
        return 1280

    def get_metrics(self, lanes, idx):
        label = self.annotations[idx]
        org_anno = label['old_anno']
        pred = self.pred2lanes(org_anno['path'], lanes, org_anno['y_samples'])
        _, fp, fn, matches, accs, _ = LaneEval.bench(pred, org_anno['org_lanes'], org_anno['y_samples'], 0, True)
        return fp, fn, matches, accs

    def pred2lanes(self, path, pred, y_samples):
        ys = np.array(y_samples) / self.img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.get_img_width(path)).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes


    def loadOpenLane_V2(self,compute=True):
            save_path=self.root
            dataset_dir=self.root


            def image_search(dict_,dataset_dir,split,segment,filename):
                for name in tqdm(glob.glob(dict_[split]["images"])):
                    dir_=os.path.join(dataset_dir,name) # Get a list of segments
                    if not os.path.isfile(dir_):
                        if segment in os.listdir(dir_):
                            segment_img_dir=os.path.join(dir_,segment)
                            if filename in os.listdir(segment_img_dir):
                                image_path=os.path.join(segment_img_dir,image)

                                return image_path
                return -1


            output_dir=os.path.join(save_path,"dataloader")
            self.logger.debug (bcolors.OKGREEN + "Output directory: " + bcolors.ENDC+ str(output_dir))

            if not os.path.exists(output_dir):
                # Create a new directory because it does not exist
                os.makedirs(output_dir)
                self.logger.debug (bcolors.OKGREEN + "Output directory created! " + bcolors.WHITE)

            os.chdir(dataset_dir)


            if compute:
                rows=list()
                lane_info_paths=["lane3d_1000_v1.2/lane3d_1000/training/","lane3d_1000_v1.2/lane3d_1000/validation/","lane3d_1000_v1.2/lane3d_1000/test/","lane3d_300/training/","lane3d_300/validation/","lane3d_300/test/"]
                splits=["training","validation","test","training","validation","test"]
                versions=[1.2,1.2,1.2,1.0,1.0,1.0]
                col_names=["img_path","json_path","filename","segment","img_folder","data_split","case","version"]


                for idx in range(len(lane_info_paths)):
                    lane_info_path_=os.path.join(dataset_dir,lane_info_paths[idx])
                    self.logger.trace(bcolors.OKGREEN+"Folder name: "+bcolors.ENDC+ str(lane_info_path_))

                    version=versions[idx]
                    split=splits[idx]

                    if split=="test":
                        cases=os.listdir(lane_info_path_)
                        cases=[case for case in cases if not os.path.isfile(os.path.join(lane_info_path_,case))]
                        self.logger.trace(bcolors.OKGREEN+"cases: "+bcolors.ENDC+ str(cases))

                    else:
                        cases=["None"]

                    for case in cases:
                        self.logger.trace(bcolors.OKGREEN+"case: "+bcolors.ENDC+ str(case))

                        if case!="None":
                            lane_info_path=os.path.join(lane_info_path_,case)
                        else:
                            lane_info_path=lane_info_path_


                        segments=os.listdir(lane_info_path)
                        segments=[segment for segment in segments if not os.path.isfile(os.path.join(lane_info_path,segment))]
                        self.logger.trace(bcolors.OKGREEN+"segments: "+bcolors.ENDC+ str(segments))

                        for segment in segments:
                            self.logger.trace(bcolors.OKGREEN+"segment: "+bcolors.ENDC+ str(segment))

                            target_folders_=glob.glob("images_*")
                            self.logger.trace(bcolors.OKGREEN+"target_folders: "+bcolors.ENDC+ str(target_folders_))

                            target_folders=[]
                            for folder in tqdm(target_folders_):
                                path=os.path.join(dataset_dir,folder)
                                self.logger.trace(bcolors.OKGREEN+"folder: "+bcolors.ENDC+ str(path))
                                if not os.path.isfile(path):
                                    target_folders.append(folder)
                                    self.logger.trace(bcolors.OKGREEN+"isFile: "+bcolors.ENDC+ str(os.path.isfile(path)))

                            for folder_name in tqdm(target_folders):
                                self.logger.trace(bcolors.OKGREEN+" Searching for segment in folder: "+bcolors.ENDC+ str(folder_name))
                                dir=os.path.join(dataset_dir,folder_name)
                                self.logger.trace(bcolors.OKGREEN+"dir: "+bcolors.ENDC+ str(dir))

                                segments_=os.listdir(dir)
                                if segment in segments_:
                                    self.logger.trace(bcolors.OKGREEN+"Found segment: "+bcolors.ENDC+ str(segment)+
                                    bcolors.OKGREEN+" in target folder: "+bcolors.ENDC+ str(folder_name))

                                    img_dir=os.path.join(dir,segment) # Image Segment Directory
                                    # Search for images within Json Segment Directory
                                    json_dir=os.path.join(lane_info_path,segment)

                                    jsons= os.listdir(json_dir)
                                    # Preserve only files (avoid directories)
                                    jsons=[json for json in jsons if os.path.isfile(os.path.join(json_dir,json))]

                                    for json in jsons:
                                        filename=json.split(".json")[0]
                                        # if filename=="151676164912912900":
                                        #     breakpoint()

                                        image=filename+".jpg"
                                        json_path=os.path.join(json_dir,json)
                                        image_path=os.path.join(img_dir,image)


                                        if (os.path.isfile(image_path))and(os.path.isfile(json_path)):
                                            rows.append([image_path,json_path,filename,segment,
                                            folder_name,split,case,version])



                rows=np.array(rows)
                data = pd.DataFrame(rows, columns=col_names)

                # Print dataframe types
                self.logger.trace(bcolors.OKGREEN+"Dataframe\n"+bcolors.ENDC+str(data.loc[:,self.view_cols].head(20)))
                data[col_names] = data[col_names].astype(str) # From object to string
                data["version"] = data["version"].astype(float) # From string to float
                # df['column_name'] = pd.to_numeric(df['column_name'], errors='coerce')
                self.logger.trace(bcolors.OKGREEN+"Dataframe types: "+bcolors.ENDC+str(data.dtypes))
                data.to_csv(output_dir+'OpenLane_dataset.csv', index=False)

                """
                data_v=data.loc[(data["version"]==1.0)]
                self.logger.trace(bcolors.OKGREEN+"Dataframe (v1.0) \n"+bcolors.ENDC+str(data_v.loc[:,self.view_cols].head(20)))

                data_v=data.loc[(data["version"]==1.2)]
                self.logger.trace(bcolors.OKGREEN+"Dataframe (v1.2) \n"+bcolors.ENDC+str(data_v.loc[:,self.view_cols].head(20)))


                data_duplicated=data.loc[data.filename.duplicated(keep=False)].sort_values("filename")
                data_duplicated_grouped=data_duplicated.groupby("filename")


                # Iterate over the groups and print the group name and the rows in the group
                for name, group in data_duplicated_grouped:
                    self.logger.trace(bcolors.OKGREEN+"name: "+bcolors.ENDC+str(name)+
                    bcolors.OKGREEN+" group:\n"+bcolors.ENDC+str(group.loc[:,self.view_cols])+"\n")


                self.logger.trace(bcolors.OKGREEN+"Nº of duplicated elements: "+bcolors.ENDC+str(len(data_duplicated_grouped)))
                self.logger.trace(bcolors.OKGREEN+"data_duplicated \n"+bcolors.ENDC+str(data_duplicated.loc[:,self.view_cols].head(20)))

                data_f=data.loc[(data["filename"]=="151676164912912900")]
                self.logger.trace(bcolors.OKGREEN+"data_f \n"+bcolors.ENDC+str(data_f.loc[:,self.view_cols].head(20)))

                """


            else:
                filename=output_dir+'OpenLane_dataset.csv'
                data = pd.read_csv(filename)
                self.logger.debug(bcolors.OKGREEN+"data\n"+bcolors.ENDC+str(data.head(20)))



            # SLICE ARRAY WITH DATA SPLIT, AND DATAFRAME VERSION
            data_version=data.loc[(data["version"]==1.2)]

            if self.split=="train+val":
                data_split=data_version.loc[ (data_version["data_split"]=="training")|(data_version["data_split"]=="validation")  ]
            elif self.split=="train":
                data_split=data_version.loc[(data_version["data_split"]=="training")]

            elif self.split=="val":
                data_split=data_version.loc[(data_version["data_split"]=="validation")]

            elif self.split=="test":
                data_split=data_version.loc[(data_version["data_split"]=="test")]


            # Getting rid of duplicates:
            data_split=data_split.loc[data_split.filename.duplicated(keep=False)].sort_values("filename")

            # Get a subsample of the dataset
            data_split=data_split.loc[data_split.index[0:1000]]


            self.logger.trace(bcolors.OKGREEN+"data_split (split & version) \n"+bcolors.ENDC+str(data_split.loc[:,self.view_cols].head(20)))
            return data_split


    def debug_json_info(self,info_dict,calib=False):
        lane_lines=info_dict["lane_lines"]
        gt_lanes_2d_pts=[]


        for idx in range(len(lane_lines)):
            lane=lane_lines[idx]
            # 2D
            lane_pts_2d=np.array(lane["uv"]).T # npts, Dim (2)

            """
            lane_pts_2d_laneatt=[]
            for x,y in lane_pts_2d:
                lane_pts_2d_laneatt.append((x,y))
            """

            self.logger.debug("Shape 2d lane: "+str(lane_pts_2d.shape))
            self.logger.debug(bcolors.FAIL+"lane_pts_2d"+bcolors.ENDC+str(lane_pts_2d.shape))

            gt_lanes_2d_pts.append(lane_pts_2d)

        return gt_lanes_2d_pts


    def load_annotations(self,data):
        self.logger.info('Loading OpenLane annotations...')
        self.annotations = []
        max_lanes = 0


        segments=list(data["segment"].unique())
        self.logger.info(bcolors.OKGREEN+"Nº of segments: "+bcolors.ENDC+str(len(segments)))

        for segment_idx in tqdm(range(len(segments)), bar_format=bcolors.FAIL+'{l_bar}{bar}{r_bar}'+bcolors.ENDC):
        # for segment_idx in tqdm(range(len(segments))):
            segment=segments[segment_idx]
            self.logger.trace(bcolors.WARNING+"segment: "+bcolors.ENDC+str(segment))

            # 1. Retrieve images segment by segment (sot by segment )
            data_segment = data.loc[data["segment"]==segment].sort_values("filename")
            self.logger.trace(bcolors.WARNING+"data_segment:\n"+bcolors.ENDC+str(data_segment.loc[:,self.view_cols].head(20)))

            # 2. Retrieve data_segment indexes
            data_segment_idxs=list(data_segment.index)
            self.logger.info(bcolors.OKGREEN+"Nº of images in segment: "+bcolors.ENDC+str(len(data_segment_idxs)))


            for idx in tqdm(range(len(data_segment_idxs))): # HARDCODED
                idx=data_segment_idxs[idx]
                self.logger.trace(bcolors.OKGREEN+"Dataset idx: "+bcolors.ENDC+str(idx))

                # 3. Retrieve paths
                json_path = data.loc[idx,"json_path"]
                image_path = data.loc[idx,"img_path"]

                self.logger.trace(bcolors.WARNING+"image_path: "+bcolors.ENDC+"\n"+str(image_path))
                self.logger.trace(bcolors.WARNING+"json_path: "+bcolors.ENDC+"\n"+str(json_path))

                if (os.path.isfile(json_path)):
                    with open(json_path, 'r') as f:
                        info_dict = json.load(f)
                        self.logger.debug(bcolors.WARNING+"info_dict: "+bcolors.ENDC+"\n"+str(info_dict))

                        gt_lanes_2d=self.debug_json_info(info_dict,calib=False)
                        max_lanes = max(max_lanes, len(gt_lanes_2d))

                        # I might have to transform this into anchor format
                        self.annotations.append({
                            'path': image_path,
                            'org_path': image_path, # image_file
                            'lanes': gt_lanes_2d,
                            'aug': False,
                        })

                    self.logger.trace(bcolors.WARNING+"N of lanes (2d): "+bcolors.ENDC+str(len(gt_lanes_2d)))


                    # 'path': os.path.join(self.root, data['raw_file']),
                    # 'org_path': data['raw_file'],
                    # 'org_lanes': gt_lanes,
                    # 'y_samples': y_samples

        if self.split == 'train':
            random.shuffle(self.annotations)

        self.max_lanes = max_lanes
        self.logger.info('%d annotations loaded, with a maximum of %d lanes in an image.', len(self.annotations),
                        self.max_lanes)



    def transform_annotations(self, transform):
        self.annotations = list(map(transform, self.annotations))

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.annotations[idx]['old_anno']['org_path']
        h_samples = self.annotations[idx]['old_anno']['y_samples']
        lanes = self.pred2lanes(img_name, pred, h_samples)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions, runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def eval_predictions(self, predictions, output_basedir, runtimes=None):
        pred_filename = os.path.join(output_basedir, 'tusimple_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        result = json.loads(LaneEval.bench_one_submit(pred_filename, self.anno_files[0]))
        table = {}
        for metric in result:
            table[metric['name']] = metric['value']

        return table

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
