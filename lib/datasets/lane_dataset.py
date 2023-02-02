import logging
import pdb
import os, glob, sys


import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d,lagrange
import math

import cv2
from skimage.morphology import medial_axis

import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from imgaug.augmentables.lines import LineString, LineStringsOnImage

from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset

from utils.openlane_utils import bcolors,FormatAxes, get_cmap, create_trace_loglevel,\
getRotation,getRotation_2d,getRigidTransformation
from lib.lane import Lane
from .culane import CULane
from .tusimple import TuSimple
from .llamas import LLAMAS
from .nolabel_dataset import NoLabelDataset
from .openlane import OpenLane
from .bosch import Bosch

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class LaneDataset(Dataset):
    def __init__(self,
                 S=72,
                 dataset='tusimple',
                 augmentations=None,
                 normalize=False,
                 img_size=(360, 640),
                 aug_chance=1.,
                 **kwargs):
        super(LaneDataset, self).__init__()

        self.dataset_name=dataset
        if dataset == 'tusimple':
            self.dataset = TuSimple(**kwargs)
        elif dataset == 'culane':
            self.dataset = CULane(**kwargs)
        elif dataset == 'llamas':
            self.dataset = LLAMAS(**kwargs)
        elif dataset == 'nolabel_dataset':
            self.dataset = NoLabelDataset(**kwargs)
        elif dataset == 'openlane':
            self.dataset = OpenLane(**kwargs)
        elif dataset == 'bosch':
            self.dataset = Bosch(**kwargs)

        else:
            raise NotImplementedError()
        self.n_strips = S - 1
        self.n_offsets = S
        self.normalize = normalize
        self.img_h, self.img_w = img_size
        self.strip_size = self.img_h / self.n_strips
        self.logger = logging.getLogger(__name__)

        # y at each x offset
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.transform_annotations()

        if augmentations is not None:
            # add augmentations
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in augmentations]  # add augmentation
        else:
            augmentations = []

        transformations = iaa.Sequential([Resize({'height': self.img_h, 'width': self.img_w})])
        self.to_tensor = ToTensor()
        self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=aug_chance), transformations])
        self.max_lanes = self.dataset.max_lanes

        if self.dataset_name=="bosch":
            # Belongs to Bosch class
            self.logger.trace (bcolors.OKGREEN + "H_im2ipm:\n" + bcolors.ENDC+ str(self.dataset.H_im2ipm))




    @property
    def annotations(self):
        return self.dataset.annotations

    def transform_annotations(self):
        self.logger.info("dataset name: "+str(self.dataset_name))
        self.logger.info("Transforming annotations to the model's target format...")

        if self.dataset_name=="openlane":
            self.dataset.annotations = np.array(list(map(self.transform_annotation_openlane, self.dataset.annotations)))
        else:
            self.dataset.annotations = np.array(list(map(self.transform_annotation, self.dataset.annotations)))

        self.logger.info('Done.')

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane


    def unique_y_values(self,lane):
        assert lane[-1,1] <= lane[0,1]

        seen_y = set()
        unique_lane = []
        for x, y in lane:
            if y not in seen_y:
                unique_lane.append([x, y])
                seen_y.add(y)

        unique_lane=np.array(unique_lane)
        return unique_lane


    def lane2anchor(self, points, sample_ys):
        self.logger.trace (bcolors.OKGREEN + "sample_ys: " + bcolors.ENDC+ str(sample_ys)) # img_h -> 0

        # this function expects the points to be sorted
        if not np.all(points[1:, 1] < points[:-1, 1]): # Should be ordenados de mayor a menor
            raise Exception('Annotation points have to be sorted ')

        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1

        domain_min_y = y.min()
        domain_max_y = y.max()

        # Resampling options
        y_sample=np.arange(domain_min_y,domain_max_y,5)

        resample_option="polyfit"
        if(resample_option=="interp1d"):
            interp = interp1d(y[::-1], x[::-1], k=min(3, len(points) - 1)) # ordenados de menor a mayor
            interp_xs = interp(y_sample)

        elif(resample_option=="InterpolatedUnivariateSpline"):
            interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1)) # ordenados de menor a mayor
            interp_xs = interp(y_sample)

        elif(resample_option=="polyfit"):
            coeff = np.polyfit(y[::-1],x[::-1],2)
            interp_xs = np.polyval(coeff, y_sample)   # evaluate the polynomial



        interp_data= np.append(interp_xs.reshape(-1,1), y_sample.reshape(-1,1),axis=1)
        self.logger.trace (bcolors.OKGREEN + "interp_data : " + bcolors.ENDC+ str(interp_data)) # img_h -> 0

        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0

        if(resample_option=="polyfit"):
            interp_xs = np.polyval(coeff, sample_ys_inside_domain)   # evaluate the polynomial
        else:
            interp_xs = interp(sample_ys_inside_domain)


        interp_data= np.append(interp_xs.reshape(-1,1), sample_ys_inside_domain.reshape(-1,1),axis=1)


        # we dont have an array that holds which points are vissible and which dont
        # we only interpolate in the range the lane does actually exist

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]

        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        self.logger.trace (bcolors.OKGREEN + "interp_xs&ys  : " + bcolors.ENDC+ str(interp_data)) # img_h -> 0
        self.logger.trace (bcolors.OKGREEN + "extrap_ys: " + bcolors.ENDC+ str(extrap_ys)) # img_h -> 0
        self.logger.trace (bcolors.OKGREEN + "extrap_xs : " + bcolors.ENDC+ str(extrap_xs)) # img_h -> 0


        self.logger.trace (bcolors.OKGREEN + "Anchor (all) : " + bcolors.ENDC+ str(all_xs)) # img_h -> 0

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        self.logger.trace (bcolors.OKGREEN + "xs_inside_image  : " + bcolors.ENDC+ str(xs_inside_image)) # img_h -> 0
        self.logger.trace (bcolors.OKGREEN + "xs_outside_image  : " + bcolors.ENDC+ str(xs_outside_image)) # img_h -> 0

        return xs_outside_image, xs_inside_image



    def transform_annotation_openlane(self, anno, img_wh=None):
        if img_wh is None:
            img_h = self.dataset.get_img_heigth(anno['path'])
            img_w = self.dataset.get_img_width(anno['path'])
        else:
            img_w, img_h = img_wh

        self.logger.trace (bcolors.OKGREEN + "img_w: " + bcolors.ENDC+ str(img_w))
        self.logger.trace (bcolors.OKGREEN + "img_h: " + bcolors.ENDC+ str(img_h))


        old_lanes = anno['lanes']
        self.logger.trace (bcolors.OKGREEN + "old_lanes: " + bcolors.ENDC+ str(old_lanes))


        # CREATE TRANFORMED ANNOTATIONS
        lanes = np.ones((self.dataset.max_lanes, 2 + 1 + 1 + 1 + self.n_offsets),
                        dtype=np.float32) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 length, S+1 coordinates [self.n_offsets]
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0



        for idx in range(len(old_lanes)):
            lane=old_lanes[idx] # access data

            self.logger.trace (bcolors.OKGREEN + "lane (old): " + bcolors.ENDC+ str(lane))

            # 0. REMOVING LANES WITH LESS THAN 2 POINTS
            if len(lane)<=1: # At least 2 points
                continue

            # 1. SORT IN Y (I do not know if this is compatible with openlane!?)
            lane_sort_idxs=np.argsort(lane[:,1])
            lane=lane[lane_sort_idxs,:]
            lane=lane[::-1,:]

            self.logger.trace (bcolors.OKGREEN + "lane (sorted): " + bcolors.ENDC+ str(lane))
            lane=self.unique_y_values(lane)
            self.logger.trace (bcolors.OKGREEN + "lane (sorted & unique): " + bcolors.ENDC+ str(lane))

            # 2. NORMALIZE THE ANNOTATION COORDINATES
            # img_w -> resolution of the input image -> self.img_w: transformed coordinates
            # img_h -> resolution of the input image -> self.img_w: transformed coordinates

            lane[:,0]=lane[:,0]* self.img_w / float(img_w)
            lane[:,1]=lane[:,1]* self.img_h / float(img_h)


            self.logger.trace (bcolors.OKGREEN + "lane (sorted & unique & normalized): " + bcolors.ENDC+ str(lane))

            # LANE2ANCHOR
            # xs_outside_image, xs_inside_image = self.lane2anchor(lane, self.offsets_ys)

            try:
                xs_outside_image, xs_inside_image = self.lane2anchor(lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) == 0:
                continue


            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[idx, 0] = 0
            lanes[idx, 1] = 1
            lanes[idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[idx, 3] = xs_inside_image[0]
            lanes[idx, 4] = len(xs_inside_image)
            lanes[idx, 5:5 + len(all_xs)] = all_xs

        new_anno = {'path': anno['path'], 'label': lanes, 'old_anno': anno}
        return new_anno





        # create tranformed annotations
        lanes = np.ones((self.dataset.max_lanes, 2 + 1 + 1 + 1 + self.n_offsets),
                        dtype=np.float32) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 length, S+1 coordinates [self.n_offsets]
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0


        # Lanes to anchors

        for lane_idx, lane in enumerate(old_lanes):
            try:
                xs_outside_image, xs_inside_image = self.lane2anchor(lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) == 0:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[lane_idx, 3] = xs_inside_image[0]
            lanes[lane_idx, 4] = len(xs_inside_image)
            lanes[lane_idx, 5:5 + len(all_xs)] = all_xs

        new_anno = {'path': anno['path'], 'label': lanes, 'old_anno': anno}
        return new_anno


    def transform_annotation(self, anno, img_wh=None):
        if img_wh is None:
            img_h = self.dataset.get_img_heigth(anno['path'])
            img_w = self.dataset.get_img_width(anno['path'])
        else:
            img_w, img_h = img_wh

        old_lanes = anno['lanes']

        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [[[x * self.img_w / float(img_w), y * self.img_h / float(img_h)] for x, y in lane]
                     for lane in old_lanes]
        # create tranformed annotations
        lanes = np.ones((self.dataset.max_lanes, 2 + 1 + 1 + 1 + self.n_offsets),
                        dtype=np.float32) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 length, S+1 coordinates [self.n_offsets]
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0


        for lane_idx, lane in enumerate(old_lanes):
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) == 0:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[lane_idx, 3] = xs_inside_image[0]
            lanes[lane_idx, 4] = len(xs_inside_image)
            lanes[lane_idx, 5:5 + len(all_xs)] = all_xs

        new_anno = {'path': anno['path'], 'label': lanes, 'old_anno': anno}
        return new_anno

    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def label_to_lanes(self, label):
        lanes = []
        for l in label:
            if l[1] == 0:
                continue
            xs = l[5:] / self.img_w
            ys = self.offsets_ys / self.img_h
            start = int(round(l[2] * self.n_strips))
            length = int(round(l[4]))
            xs = xs[start:start + length][::-1]
            ys = ys[start:start + length][::-1]
            xs = xs.reshape(-1, 1)
            ys = ys.reshape(-1, 1)
            points = np.hstack((xs, ys))

            lanes.append(Lane(points=points))
        return lanes




    def draw_annotation_DavidMataix(self, idx, label=None, pred=None, img=None, semantic=None):
        # Get image if not provided
        if img is None:
            # print(self.annotations[idx]['path'])
            img, label, _ = self.__getitem__(idx)
            label = self.label_to_lanes(label)
            img = img.permute(1, 2, 0).numpy()
            if self.normalize:
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = (img * 255).astype(np.uint8)
        else:
            _, label, _ = self.__getitem__(idx)
            label = self.label_to_lanes(label)
        img = cv2.resize(img, (self.img_w, self.img_h))
        semantic = cv2.resize(semantic, (self.img_w, self.img_h))
        img_h, _, _ = img.shape
        # Pad image to visualize extrapolated predictions
        pad = 0
        if pad > 0:
            img_pad = np.zeros((self.img_h + 2 * pad, self.img_w + 2 * pad, 3), dtype=np.uint8)
            img_pad[pad:-pad, pad:-pad, :] = img
            img = img_pad
        data = [(None, None, label)]
        if pred is not None:
            # print(len(pred), 'preds')
            fp, fn, matches, accs = self.dataset.get_metrics(pred, idx)
            # print('fp: {} | fn: {}'.format(fp, fn))
            # print(len(matches), 'matches')
            # print(matches, accs)
            assert len(matches) == len(pred)
            data.append((matches, accs, pred))
        else:
            fp = fn = None
        for matches, accs, datum in data:
            #-------------------------------------------------------------------------------------------------
            #Extact semantic images
            c1 = np.asarray([37, 193, 255])   #
            c2 = np.asarray([37, 193, 255])   #

            mask1 = cv2.inRange(semantic,c1, c2)

            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(4,4))
            temp = cv2.dilate(mask1,kernel)
            eroded = cv2.erode(temp,kernel2)
            skeleton = medial_axis(eroded).astype(np.uint8)

            skeleton_points=np.argwhere((skeleton[:,:]==1))
            skeleton_points_idx=np.where((skeleton[:,:]==1))
            img = np.asarray(img)
            contours_cl,_= cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            #cv2.drawContours(img, contours_cl, -1, (255,255,0), 1)
            centroid_continous_line = []
            for contour in contours_cl:
                points_inside = []
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    #img = cv2.circle(img, (cx,cy), radius=3, color=(0, 0, 255), thickness=2)
                    #Dibujar los puntos dentro del contorno
                    centroid_continous_line.append([cx, cy])
                    for point in skeleton_points:
                        result =cv2.pointPolygonTest(contour, (point[1], point[0]), False)

                        if result != -1: #out
                            points_inside.append([point[1], point[0]])
                            centroid_continous_line.append([point[1], point[0]])
                    cont = 0
                    primero =True
                    tuple_fin_10 =-1
                    for point_aux in points_inside:
                        if point_aux[1] > int(self.img_h/2)-5:
                            if cont > 10:
                                img = cv2.circle(img, (point_aux[0],point_aux[1]), radius=2, color=(0, 255, 0), thickness=1)
                                cont =0
                                if not primero:
                                    img = cv2.line(img,tuple_fin_10,(point_aux[0],point_aux[1]),color=(0,0,0),thickness=1)
                                    tuple_fin_10 = (point_aux[0],point_aux[1])
                                else:
                                    tuple_fin_10 = (point_aux[0],point_aux[1])
                                    primero = False
                            cont +=1
                    cont = 0

            #Dashed line
            c1 = np.asarray([255, 0, 128])   #
            c2 = np.asarray([255, 0, 128])   #
            mask2 = cv2.inRange(semantic,c1, c2)

            #Skeleton
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(4,4))
            temp = cv2.dilate(mask2,kernel)
            eroded = cv2.erode(temp,kernel2)
            skeleton = medial_axis(eroded).astype(np.uint8)

            skeleton_points=np.argwhere((skeleton[:,:]==1))

            contours_dl,_= cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #cv2.drawContours(img, contours_dl, -1, (255,255,0), 1)
            centroid_dashed_line = []
            for contour in contours_dl:
                points_inside = []
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    centroid_dashed_line.append([cx,cy])
                    #img = cv2.circle(img, (cx,cy), radius=3, color=(0, 0, 255), thickness=2)
                    for point in skeleton_points:
                        result =cv2.pointPolygonTest(contour, (point[1], point[0]), False)

                        if result != -1: #out
                            points_inside.append([point[1], point[0]])
                            centroid_dashed_line.append([point[1], point[0]])
                    cont = 0
                    primero =True
                    tuple_fin_10 =-1
                    for point_aux in points_inside:
                        if point_aux[1] > int(self.img_h/2)-5:
                            if cont > 10:
                                img = cv2.circle(img, (point_aux[0],point_aux[1]), radius=2, color=(0, 255, 0), thickness=1)
                                cont =0
                                if not primero:
                                    img = cv2.line(img,tuple_fin_10,(point_aux[0],point_aux[1]),color=(0,0,0),thickness=1)
                                    tuple_fin_10 = (point_aux[0],point_aux[1])
                                else:
                                    tuple_fin_10 = (point_aux[0],point_aux[1])
                                    primero = False
                            cont +=1
                    cont = 0

            #POint of interest contonuous line and dashed line
            #centroid_continous_line
            #centroid_dashed_line

            for i, l in enumerate(datum):
                if matches is None:
                    color = GT_COLOR
                elif matches[i]:
                    color = PRED_HIT_COLOR
                else:
                    color = PRED_MISS_COLOR
                points = l.points
                points[:, 0] *= img.shape[1]
                points[:, 1] *= img.shape[0]
                points = points.round().astype(int)
                points += pad
                xs, ys = points[:, 0], points[:, 1]
                print("Nueva line")
                #print(int(len(points)/4))
                iter_num_puntos_ecuacion = int(len(points)/4)
                points_aux = []
                points_aux.append(points[0])
                points_aux.append(points[iter_num_puntos_ecuacion])
                points_aux.append(points[iter_num_puntos_ecuacion*2])
                points_aux.append(points[-1])
                # Three rectangles
                cnt = [[points_aux[1][0]-20,points_aux[1][1]],[points_aux[0][0]-20,points_aux[0][1]],[points_aux[0][0]+20,points_aux[0][1]],[points_aux[1][0]+20,points_aux[1][1]]]
                bbox1 = np.int0(cnt)
                cnt2 = [[points_aux[2][0]-20,points_aux[2][1]],[points_aux[1][0]-20,points_aux[1][1]],[points_aux[1][0]+20,points_aux[1][1]],[points_aux[2][0]+20,points_aux[2][1]]]
                bbox2 = np.int0(cnt2)
                cnt3 = [[points_aux[3][0]-20,points_aux[3][1]],[points_aux[2][0]-20,points_aux[2][1]],[points_aux[2][0]+20,points_aux[2][1]],[points_aux[3][0]+20,points_aux[3][1]]]
                bbox3 = np.int0(cnt3)


                '''cv2.drawContours(img, [bbox1], 0, (0,255,0), 3)
                cv2.drawContours(img, [bbox2], 0, (0,255,0), 3)
                cv2.drawContours(img, [bbox3], 0, (0,255,0), 3)'''
                #-------------------------------------------CONTINUOS LINE
                puntos_dentro = []
                if len(centroid_continous_line) > 0:
                    for centroid in centroid_continous_line:
                        if centroid[1]  > int(self.img_h/2)-5:
                            result =cv2.pointPolygonTest(bbox1, (centroid[0], centroid[1]), False)
                            if result != -1: #out
                                puntos_dentro.append(centroid)

                            result =cv2.pointPolygonTest(bbox2, (centroid[0], centroid[1]), False)
                            if result != -1: #out
                                puntos_dentro.append(centroid)

                            result =cv2.pointPolygonTest(bbox3, (centroid[0], centroid[1]), False)
                            if result != -1: #out
                                puntos_dentro.append(centroid)


                if len(centroid_dashed_line) > 0:
                    for centroid in centroid_dashed_line:
                        if centroid[1] > int(self.img_h/2)-5:
                            result =cv2.pointPolygonTest(bbox1, (centroid[0], centroid[1]), False)
                            if result != -1: #out
                                puntos_dentro.append(centroid)

                            result =cv2.pointPolygonTest(bbox2, (centroid[0], centroid[1]), False)
                            if result != -1: #out
                                puntos_dentro.append(centroid)
                            result =cv2.pointPolygonTest(bbox3, (centroid[0], centroid[1]), False)
                            if result != -1: #out
                                puntos_dentro.append(centroid)


                if len(puntos_dentro) >0: #Dashed line or COntinous line
                    aux_final_points_x  =[]
                    aux_final_points_y  =[]
                    for point in points:
                        if point[1] > int(self.img_h/2)-5:
                            aux_final_points_x.append(point[0])
                            aux_final_points_y.append(point[1])

                    for m in puntos_dentro:
                        aux_final_points_x.append(m[0])
                        aux_final_points_y.append(m[1])

                    #NUmpy values Aproximate polinomial
                    x = np.array(aux_final_points_x)
                    y = np.array(aux_final_points_y)
                    coeff = np.polyfit(x,y,3)
                    t = np.linspace(np.min(x), np.max(x), 50)
                    draw_y = np.polyval(coeff, t)   # evaluate the polynomial
                    print(np.mean(x))
                    for idx,elment in enumerate(draw_y):
                        img = cv2.circle(img, (int(t[idx]),int(elment)), radius=1, color=(0, 0, 255), thickness=-1)
                else: #No point inside:
                    for curr_p, next_p in zip(points_aux[:-1], points_aux[1:]):
                        img = cv2.circle(img, (int(curr_p[0]),int(curr_p[1])), radius=1, color=(0, 0, 255), thickness=-1)

                '''if len(centroid_continous_line) > 0:
                    puntos_dentro_cl = []
                    for centroid in centroid_continous_line:
                        result =cv2.pointPolygonTest(bbox1, (centroid[0], centroid[1]), False)
                        if result != -1: #out
                            puntos_dentro_cl.append(centroid)

                        result =cv2.pointPolygonTest(bbox2, (centroid[0], centroid[1]), False)
                        if result != -1: #out
                            puntos_dentro_cl.append(centroid)

                        result =cv2.pointPolygonTest(bbox3, (centroid[0], centroid[1]), False)
                        if result != -1: #out
                            puntos_dentro_cl.append(centroid)


                    #UNir puntos con los centroides:
                    print("------------")
                    aux_final_points_x  =[]
                    aux_final_points_y  =[]
                    for point in points:
                        aux_final_points_x.append(point[0])
                        aux_final_points_y.append(point[1])

                    for m in puntos_dentro_cl:
                        aux_final_points_x.append(m[0])
                        aux_final_points_y.append(m[1])

                    #NUmpy values
                    x = np.array(aux_final_points_x)
                    y = np.array(aux_final_points_y)
                    coeff = np.polyfit(x,y,3)
                    t = np.linspace(np.min(x), np.max(x), 50)
                    draw_y = np.polyval(coeff, t)   # evaluate the polynomial

                    for idx,elment in enumerate(draw_y):
                        print(int(t[idx]),int(elment))
                        img = cv2.circle(img, (int(t[idx]),int(elment)), radius=1, color=(255, 255, 255), thickness=-1)'''
                #-----------------------------------------------------Dashed line---------------------------------
                '''if len(centroid_dashed_line) > 0:
                    puntos_dentro = []
                    for centroid in centroid_dashed_line:
                        result =cv2.pointPolygonTest(bbox1, (centroid[0], centroid[1]), False)
                        if result != -1: #out
                            puntos_dentro.append(centroid)

                        result =cv2.pointPolygonTest(bbox2, (centroid[0], centroid[1]), False)
                        if result != -1: #out
                            print(centroid)
                            puntos_dentro.append(centroid)
                        result =cv2.pointPolygonTest(bbox3, (centroid[0], centroid[1]), False)
                        if result != -1: #out
                            puntos_dentro.append(centroid)

                    #UNir puntos con los centroides:
                    print("------------")
                    aux_final_points_x  =[]
                    aux_final_points_y  =[]
                    for point in points:
                        aux_final_points_x.append(point[0])
                        aux_final_points_y.append(point[1])

                    for m in puntos_dentro:
                        aux_final_points_x.append(m[0])
                        aux_final_points_y.append(m[1])

                    print(len(points))
                    print(type(points))
                    print(type(puntos_dentro))
                    #NUmpy values
                    x = np.array(aux_final_points_x)
                    y = np.array(aux_final_points_y)
                    coeff = np.polyfit(x,y,3)
                    t = np.linspace(np.min(x), np.max(x), 50)
                    draw_y = np.polyval(coeff, t)   # evaluate the polynomial

                    for idx,elment in enumerate(draw_y):
                        print(int(t[idx]),int(elment))
                        img = cv2.circle(img, (int(t[idx]),int(elment)), radius=1, color=(255, 255, 255), thickness=-1)'''
                #-----------------------------------------------------------------------------------------------------------

                '''for curr_p, next_p in zip(points_aux[:-1], points_aux[1:]):
                    tuple_origin = (curr_p[0]+20,curr_p[1])
                    tuple_fin = (next_p[0]+20,next_p[1])
                    tuple_origin_10 = (curr_p[0]-20,curr_p[1])
                    tuple_fin_10 = (next_p[0]-20,next_p[1])
                    #img = cv2.circle(img, (int(curr_p[0]),int(curr_p[1])), radius=2, color=(255, 0, 0), thickness=-1)
                    img = cv2.circle(img, (int(curr_p[0]+20),int(curr_p[1])), radius=2, color=(0, 255, 0), thickness=-1)
                    img = cv2.circle(img, (int(curr_p[0]-20),int(curr_p[1])), radius=2, color=(0, 255, 0), thickness=-1)
                    #LADO DERECHO
                    img = cv2.line(img,tuple_origin,tuple_fin,color=(0,0,0),thickness=1)
                    #LADO IZQUIERDO
                    img = cv2.line(img,tuple_origin_10,tuple_fin_10,color=(0,0,0),thickness=1)
                '''
                '''for curr_p, next_p in zip(points[:-1], points[1:]):
                    img = cv2.circle(img, (int(curr_p[0]),int(curr_p[1])), radius=1, color=(255, 0, 0), thickness=-1)'''
                '''for curr_p, next_p in zip(points[:-1], points[1:]):
                    #print(curr_p)
                    tuple_origin = (curr_p[0]+10,curr_p[1])
                    tuple_fin = (next_p[0]+10,next_p[1])
                    tuple_origin_10 = (curr_p[0]-10,curr_p[1])
                    tuple_fin_10 = (next_p[0]-10,next_p[1])
                    img = cv2.circle(img, (int(curr_p[0]),int(curr_p[1])), radius=2, color=(255, 0, 0), thickness=-1)'''
                    #img = cv2.circle(img, (int(curr_p[0]+10),int(curr_p[1])), radius=2, color=(0, 255, 0), thickness=-1)
                    #img = cv2.circle(img, (int(curr_p[0]-10),int(curr_p[1])), radius=2, color=(0, 255, 0), thickness=-1)
                    #img = cv2.line(img,tuple_origin,tuple_fin,color=(0,0,0),thickness=1)
                    #img = cv2.line(img,tuple_origin_10,tuple_fin_10,color=(0,0,0),thickness=1)
                    #img = cv2.line(img,tuple(curr_p),tuple(next_p),color=color,thickness=3 if matches is None else 3)
                # if 'start_x' in l.metadata:
                #     start_x = l.metadata['start_x'] * img.shape[1]
                #     start_y = l.metadata['start_y'] * img.shape[0]
                #     cv2.circle(img, (int(start_x + pad), int(img_h - 1 - start_y + pad)),
                #                radius=5,
                #                color=(0, 0, 255),
                #                thickness=-1)
                # if len(xs) == 0:
                #     print("Empty pred")
                # if len(xs) > 0 and accs is not None:
                #     cv2.putText(img,
                #                 '{:.0f} ({})'.format(accs[i] * 100, i),
                #                 (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad)),
                #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #                 fontScale=0.7,
                #                 color=color)
                #     cv2.putText(img,
                #                 '{:.0f}'.format(l.metadata['conf'] * 100),
                #                 (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad - 50)),
                #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #                 fontScale=0.7,
                #                 color=(255, 0, 255))
        return img, fp, fn







    def draw_annotation(self, idx, label=None, pred=None, img=None,save_predictions=True,bins=False):
        # Get image if not provided
        if img is None:
            # print(self.annotations[idx]['path'])
            img, label, _ = self.__getitem__(idx)
            label = self.label_to_lanes(label)
            img = img.permute(1, 2, 0).numpy()
            if self.normalize:
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = (img * 255).astype(np.uint8)
        else:
            _, label, _ = self.__getitem__(idx)
            label = self.label_to_lanes(label)


        self.logger.trace(bcolors.OKGREEN+'idx: ' +bcolors.ENDC+str(idx))
        self.logger.trace(bcolors.OKGREEN+'label: ' +bcolors.ENDC+str(label))

        img_org = cv2.resize(img, (self.img_w, self.img_h))
        img=np.array(img_org,copy=True)

        """


        if self.dataset_name=="bosch":
            # IMAGE TO IPM

            im_ipm = cv2.warpPerspective(img.astype("float") / 255, np.squeeze(self.dataset.H_im2ipm), (self.dataset.ipm_w, self.dataset.ipm_h))
            self.logger.trace(bcolors.OKGREEN+'im_ipm\n: ' +bcolors.ENDC+str(im_ipm))

            # im_ipm = np.clip(im_ipm, 0, 1)


            cv2.namedWindow("img", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
            cv2.namedWindow("ipm", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions


            cv2.imshow("img", img)# Show image
            cv2.imshow("ipm", im_ipm)# Show image

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        """

        img_h, _, _ = img.shape
        # Pad image to visualize extrapolated predictions
        pad = 0
        if pad > 0:
            img_pad = np.zeros((self.img_h + 2 * pad, self.img_w + 2 * pad, 3), dtype=np.uint8)
            img_pad[pad:-pad, pad:-pad, :] = img
            img = img_pad
        data = [(None, None, label)]
        if pred is not None:
            # print(len(pred), 'preds')
            fp, fn, matches, accs = self.dataset.get_metrics(pred, idx)
            # print('fp: {} | fn: {}'.format(fp, fn))
            # print(len(matches), 'matches')
            # print(matches, accs)
            assert len(matches) == len(pred)
            data.append((matches, accs, pred))
        else:
            fp = fn = None

        img=np.array(img_org,copy=True)

        for matches, accs, datum in data:

            for i, l in enumerate(datum):
                if matches is None:
                    color = GT_COLOR
                elif matches[i]:
                    color = PRED_HIT_COLOR
                else:
                    color = PRED_MISS_COLOR
                points = l.points
                points[:, 0] *= img.shape[1]
                points[:, 1] *= img.shape[0]
                points = points.round().astype(int)
                points += pad
                xs, ys = points[:, 0], points[:, 1]


                for curr_p, next_p in zip(points[:-1], points[1:]):
                    img = cv2.line(img,
                                   tuple(curr_p),
                                   tuple(next_p),
                                   color=color,
                                   thickness=3 if matches is None else 3)


            if bins:
                fontScale=1.0
                fontThickness=3

                self.logger.trace(bcolors.OKGREEN+'len(bins): ' +bcolors.ENDC+str(len(bins)))
                self.logger.trace(bcolors.OKGREEN+'len(datum): ' +bcolors.ENDC+str(len(datum)))

                for i in range(len(datum)):
                    l= datum[i]
                    bin_=int(bins[i])
                    # bin_=np.where(bins==i)[0][0]

                    points = l.points
                    self.logger.trace(bcolors.OKGREEN+'points: ' +bcolors.ENDC+str(points))
                    self.logger.trace(bcolors.OKGREEN+'bin: ' +bcolors.ENDC+str(bin_))

                    labelSize=cv2.getTextSize(str(bin_),cv2.FONT_HERSHEY_SIMPLEX,fontScale,fontThickness)

                    if points.shape[0]>3:
                        pos_x=int(points[-3,0]-0.5*labelSize[0][0])
                        pos_y=int(points[-3,1])

                    else:
                        pos_x=int(points[-1,0]-0.5*labelSize[0][0])
                        pos_y=int(points[-1,1])


                    self.logger.trace(bcolors.OKGREEN+'pos_x: ' +bcolors.ENDC+str(pos_x)+
                    bcolors.OKGREEN+'pos_y: ' +bcolors.ENDC+str(pos_y))

                    # Print lane side with respect to ego (below)
                    img=cv2.putText(img, str(bin_), (pos_x,pos_y), cv2.FONT_HERSHEY_SIMPLEX ,fontScale, (250,250,250), fontThickness, cv2.LINE_AA)


        if save_predictions:

            item = self.dataset[idx]
            img_path=item['path']
            self.logger.trace (bcolors.OKGREEN + "img_path: " + bcolors.ENDC+ str(img_path))
            filename=img_path.split('/')[-1]
            self.logger.trace (bcolors.OKGREEN + "filename: " + bcolors.ENDC+ str(filename))

            output_dir="/home/javpasto/Documents/LaneDetection/LaneATT/experiments/laneatt_r18_tusimple/results"
            output_dir=os.path.join(output_dir,"predictions")

            os.makedirs(output_dir, exist_ok=True)
            self.logger.trace (bcolors.OKGREEN + "writting image in: " + bcolors.ENDC+ str(os.path.join(output_dir,filename)))
            cv2.imwrite(os.path.join(output_dir,filename), img)



                # if 'start_x' in l.metadata:
                #     start_x = l.metadata['start_x'] * img.shape[1]
                #     start_y = l.metadata['start_y'] * img.shape[0]
                #     cv2.circle(img, (int(start_x + pad), int(img_h - 1 - start_y + pad)),
                #                radius=5,
                #                color=(0, 0, 255),
                #                thickness=-1)
                # if len(xs) == 0:
                #     print("Empty pred")
                # if len(xs) > 0 and accs is not None:
                #     cv2.putText(img,
                #                 '{:.0f} ({})'.format(accs[i] * 100, i),
                #                 (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad)),
                #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #                 fontScale=0.7,
                #                 color=color)
                #     cv2.putText(img,
                #                 '{:.0f}'.format(l.metadata['conf'] * 100),
                #                 (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad - 50)),
                #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #                 fontScale=0.7,
                #                 color=(255, 0, 255))
        return img, fp, fn

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __getitem__(self, idx):
        item = self.dataset[idx]
        path=item['path']
        img_org = cv2.imread(path)

        line_strings_org = self.lane_to_linestrings(item['old_anno']['lanes'])
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)
        for i in range(30):
            img, line_strings = self.transform(image=img_org.copy(), line_strings=line_strings_org)
            line_strings.clip_out_of_image_()
            new_anno = {'path': item['path'], 'lanes': self.linestrings_to_lanes(line_strings)}
            try:
                label = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']
                break
            except:
                if (i + 1) == 30:
                    self.logger.critical('Transform annotation failed 30 times :(')
                    exit()

        img = img / 255.
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.to_tensor(img.astype(np.float32))
        # return (img, label, idx,path)
        return (img, label, idx)

    def __len__(self):
        return len(self.dataset)
