import glob,os,sys
import logging
from .lane_dataset_loader import LaneDatasetLoader
from utils.openlane_utils import bcolors

class NoLabelDataset(LaneDatasetLoader):
    def __init__(self,split="test", img_h=720, img_w=1280, max_lanes=None, root=None, img_ext='.jpg', **_):
        """Use this loader if you want to test a model on an image without annotations or implemented loader."""
        self.root = root
        if root is None:
            raise Exception('Please specify the root directory')

        self.split = split
        self.img_w, self.img_h = img_w, img_h
        self.img_ext = img_ext
        self.annotations = []
        self.load_annotations()
        self.logger = logging.getLogger(__name__)


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
