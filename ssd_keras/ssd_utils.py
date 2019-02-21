from __future__ import division
import numpy as np 

import tensorflow as tf 
import keras.backend as K
from keras.layers import Layer
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image

import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from PIL import Image
# from scipy.misc import imread, imresize # deprecated
from imageio import imread
import skimage
from skimage import img_as_float32
from skimage.transform import resize
from imgaug import augmenters as iaa
import os
import sys
import h5py
import pickle
from random import shuffle
from bs4 import BeautifulSoup

class BBoxUtil(object):

    def __init__(self, num_classes, priors=None, overlap_threshold=0.5, nms_thresh=0.45, top_k=200):
        
        self.num_classes = num_classes
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._num_thresh = nms_thresh
        self._top_k = top_k
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores, self._top_k, iou_threshold=self._num_thresh)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':0}))

    @property
    def nms_thresh(self):
        return self._nms_thresh
    
    @nms_thresh.setter
    def nms_thresh(self, value):
        self._nms_thresh = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores, self._top_k, iou_threshold=self._nms_thresh)

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, value):
        self._top_k = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores, self._top_k, iou_threshold=self._nms_thresh)

    def iou(self, box):

        """Compute intersection over union for the box with all priors.
        # Arguments
            box: Box, numpy tensor of shape (4,).
        # Return
            iou: Intersection over union,
                numpy tensor of shape (num_priors).
        """

        # Compute intersection of union of all priors
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        
        # Compute union
        area_pred = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.priors[:, 2] - self.priors[:, 0])
        area_gt *= (self.priors[:, 3] - self.priors[:, 1])
        union = area_pred + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        """Encode box for training, do it only for assigned priors
        # Arguments
            box: Box, numpy tensor of shape (4,).
            return_iou: Whether to concat iou to encoded values.
        # Return
            encoded_box: Tensor with encoded box
                numpy tensor of shape (num_priors, 4 + int(return_iou)).
        """

        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))
        assign_mask = iou > self.overlap_threshold

        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        
        assigned_priors = self.priors[assign_mask]
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]

        assigned_priors_center = 0.5 * (assigned_priors[:, :2] + assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] - assigned_priors[:, :2])

        # Encode variance
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]
        
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        """Assign boxes to priors for training.
        # Arguments
            boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes),
                num_classes without background.
        # Return
            assignment: Tensor with assigned boxes,
                numpy tensor of shape (num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                assignment[:, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                assignment[:, -7:] are all 0. See loss for more details.
        """

        assignment = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,
                                                         np.arange(assign_num),
                                                         :4]
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1

        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        """Convert bboxes from local predictions to shifted priors.
        # Arguments
            mbox_loc: Numpy array of predicted locations.
            mbox_priorbox: Numpy array of prior boxes.
            variances: Numpy array of variances.
        # Return
            decode_bbox: Shifted priors.
        """

        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]

        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_height * variances[:, 1]
        decode_bbox_center_y += prior_center_y

        decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_height

        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                        decode_bbox_ymin[:, None],
                                        decode_bbox_xmax[:, None],
                                        decode_bbox_ymax[:, None]
                                    ), axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)

        return decode_bbox
    
    def detection_out(self, predictions, background_label_id=0, keep_top_k=100, confidence_threshold=0.01):
        """Do non maximum suppression (nms) on prediction results.
        # Arguments
            predictions: Numpy array of predicted values.
            num_classes: Number of classes for prediction.
            background_label_id: Label of background class.
            keep_top_k: Number of total bboxes to be kept per image
                after nms step.
            confidence_threshold: Only consider detections,
                whose confidences are larger than a threshold.
        # Return
            results: List of predictions for every picture. Each prediction is:
                [label, confidence, xmin, ymin, xmax, ymax]
        """

        mbox_loc = predictions[:, :, :4]
        variances = predictions[:, :, -4:]
        mbox_priorbox = predictions[:, :, -8:-4]
        mbox_conf = predictions[:, :, 4:-8]

        results = []
        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox[i], variances[i])

            for c in range(self.num_classes):
                if c == background_label_id:
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]

                    feed_dict = {self.boxes: boxes_to_process,
                                self.scores: confs_to_process}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict)

                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes), axis=1)

                    results[-1].extend(c_pred)

                if len(results[-1]) > 0:
                    results[-1] = np.array(results[-1])
                    argsort = np.argsort(results[-1][:, 1])[::-1]
                    results[-1] = results[-1][argsort]
                    results[-1] = results[-1][:keep_top_k]

        return results
    
# Generate pickle file of priorboxes for network 
def generate_priorboxes(prior_boxes_path=None,
                        img_size=(300, 300),
                        feature_map_sizes=[(36, 36), (17, 17), (8, 8), (1, 1)], 
                        num_priors=[3, 6, 6, 6], 
                        min_scale = None,
                        max_scale = None,   
                        scales = [40, 100, 168, 222, 330],        
                        # scales=[100, 168, 222, 276, 330], 
                        variances=[0.1, 0.1, 0.2, 0.2],
                        aspect_ratios=[[2.0],
                                        [2.0, 3.0],
                                        [2.0, 3.0],
                                        [2.0, 3.0]],
                        flip=True,
                        clip=True):

    prior_boxes_tensors = []
    variances = np.array(variances)

    for idx in range(len(scales) - 1):

        min_size = scales[idx]
        if idx != 0:
            max_size = scales[idx + 1]
        else: max_size = None
        
        feature_map_size = feature_map_sizes[idx]
        num_prior = num_priors[idx]
        aspect_ratio = aspect_ratios[idx]

        if min_size <= 0:
            raise Exception('min_size must be positive.')

        aspect_ratios_ = [1.0]

        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            aspect_ratios_.append(1.0)

        if aspect_ratios:
            for ar in aspect_ratio:
                if ar in aspect_ratios_:
                    continue
                aspect_ratios_.append(ar)
                if flip:
                    aspect_ratios_.append(1.0 / ar)

        layer_width = feature_map_size[0]
        layer_height = feature_map_size[1]

        img_width = img_size[0]
        img_height = img_size[1]

        # define prior boxes shapes
        box_widths = []
        box_heights = []

        for ar in aspect_ratios_:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(min_size)
                box_heights.append(min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(min_size * max_size))
                box_heights.append(np.sqrt(min_size * max_size))
            elif ar != 1:
                box_widths.append(min_size * np.sqrt(ar))
                box_heights.append(min_size / np.sqrt(ar))

        # print(box_widths)
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        # define centers of prior boxes
        step_x = img_width / layer_width
        step_y = img_height / layer_height

        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width, dtype=np.float)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height, dtype=np.float)

        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        # define xmin, ymin, xmax, ymax of prior boxes
        num_priors_ = len(aspect_ratios_)

        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors_))

        # Normalize to 0-1
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)

        # clip to 0-1
        if clip:
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)

        # define variances
        num_boxes = len(prior_boxes)

        if len(variances) == 1:
            variances_ = np.ones((num_boxes, 4)) * variances[0]
        elif len(variances) == 4:
            variances_ = np.tile(variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')

        prior_boxes_tensor = np.concatenate((prior_boxes, variances_), axis=1)
        # prior_boxes_tensor = K.expand_dims(K.variable(prior_boxes), 0)

        # if K.backend() == 'tensorflow':
            # pattern = [tf.shape(x)[0], 1, 1]
            # pattern = [feature_map_size[0], 1, 1]
            # prior_boxes_tensor = tf.tile(prior_boxes_tensor, pattern)
        
        prior_boxes_tensors.append(prior_boxes_tensor)

    prior_boxes_tensors = np.concatenate(prior_boxes_tensors, axis=0)
    # pickle.dump(prior_boxes_tensors, open('priors_ssd300_modified.pkl', 'wb'))

    return prior_boxes_tensors
        
# Implement DataGenerator class
class DataGen(object):

    def __init__(self, gt, bbox_util,
                 batch_size, path_prefix,
                 train_keys, val_keys, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 count = 0,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3./4., 4./3.]):
        self.count = count
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range

    def augment(self,
                gaussian_blur_prob=0.5,
                bi_blur_prob=0.5,
                motion_blur_prob=0.25,
                flip_prob=0.5,
                brightness_prob=0.5,
                brightness_var=2,
                sharpen_prob=0.5,
                transform_prob=0.3,
                transform_var=0.05,
                constrast_var=1.2,
                crop_percent=((0.0, 0.1), (0.0, 0.1), (0.0, 0.1), (0.0, 0.1))):

        seq = iaa.Sequential([
    
            iaa.OneOf([
                iaa.Sometimes(gaussian_blur_prob, [
                    iaa.GaussianBlur(sigma=(0.0, 0.2))
                ]),
                iaa.Sometimes(bi_blur_prob, [
                    iaa.BilateralBlur(d=(1,2))
                ])
            ]),

            iaa.Sometimes(motion_blur_prob, [
                iaa.MotionBlur(k=(3,5))
            ]),

            iaa.SomeOf(flip_prob, [
                iaa.Fliplr(0.5)
            ]),

            iaa.Sometimes(brightness_prob, [
                iaa.Add((-brightness_var, brightness_var), per_channel=0.5)
            ]),

            iaa.Sometimes(sharpen_prob, [
                iaa.Sharpen()
            ]),

            iaa.SomeOf(transform_prob, [
                iaa.PerspectiveTransform(scale=(0.0, transform_var))
            ]),

            # iaa.GammaContrast(gamma=(0.005, constrast_var)),

            iaa.CropAndPad(percent=crop_percent)
        ], random_order=True)

        return seq
        
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var 
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
    
    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y
    
    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y
    
    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))     
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y+h, x:x+w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, new_targets
    
    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:            
                img_path = os.path.join(self.path_prefix, key)
                img = imread(img_path)
                img = img_as_float32(img)
                if img.ndim != 3:
                    continue

                y = self.gt[key].copy()
                    
                # if train and self.do_crop:
                    # img, y = self.random_sized_crop(img, y)
                # img = imresize(img, self.image_size).astype('float32')
                img = resize(img, self.image_size)
                img = img_as_float32(img)
                if train:
                    # shuffle(self.color_jitter)
                    # for jitter in self.color_jitter:
                        # img = jitter(img)
                    # if self.lighting_std:
                        # img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)

                    seg = self.augment(brightness_prob=0, flip_prob=0, transform_prob=0, crop_percent=0)
                    img = seg.augment_image(img) # without changing size/dimension/etc

                y = self.bbox_util.assign_boxes(y)

                if img.ndim != 3:
                    raise Exception('Incorrect shape', key)

                inputs.append(img)                
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    # yield preprocess_input(tmp_inp), tmp_targets # only when using pre-trained model like vgg-16 with weights
                    yield tmp_inp, tmp_targets

    # def __init__(self,
    #             hdf5_dataset_path=None,
    #             label_format=None, 
    #             filenames=None, 
    #             filenames_type=None, 
    #             labels=None, 
    #             image_dir=None, 
    #             image_ids=None, 
    #             labels_output_format=('xmin', 'ymin', 'xmax', 'ymax'), # np.hstack with one-hot class-id in the end
    #             verbose=True):

    #     '''
    #     Following ssd port of Keras of 2 repo: 
    #         https://github.com/pierluigiferrari/ssd_keras/blob/master/data_generator/object_detection_2d_data_generator.py
    #         https://github.com/rykov8/ssd_keras/blob/master/SSD_training.ipynb
        
    #     The generate() is from rykov8 and the rest functions are from pierluigi

    #     Initializes the data generator. You can either load a dataset directly here in the constructor,
    #     e.g. an HDF5 dataset, or you can use one of the parser methods to read in a dataset.
    #     Arguments:
    #         load_images_into_memory (bool, optional): If `True`, the entire dataset will be loaded into memory.
    #             This enables noticeably faster data generation than loading batches of images into memory ad hoc.
    #             Be sure that you have enough memory before you activate this option.
    #         hdf5_dataset_path (str, optional): The full file path of an HDF5 file that contains a dataset in the
    #             format that the `create_hdf5_dataset()` method produces. If you load such an HDF5 dataset, you
    #             don't need to use any of the parser methods anymore, the HDF5 dataset already contains all relevant
    #             data.
    #         filenames (string or list, optional): `None` or either a Python list/tuple or a string representing
    #             a filepath. If a list/tuple is passed, it must contain the file names (full paths) of the
    #             images to be used. Note that the list/tuple must contain the paths to the images,
    #             not the images themselves. If a filepath string is passed, it must point either to
    #             (1) a pickled file containing a list/tuple as described above. In this case the `filenames_type`
    #             argument must be set to `pickle`.
    #             Or
    #             (2) a text file. Each line of the text file contains the file name (basename of the file only,
    #             not the full directory path) to one image and nothing else. In this case the `filenames_type`
    #             argument must be set to `text` and you must pass the path to the directory that contains the
    #             images in `images_dir`.
    #         filenames_type (string, optional): In case a string is passed for `filenames`, this indicates what
    #             type of file `filenames` is. It can be either 'pickle' for a pickled file or 'text' for a
    #             plain text file.
    #         images_dir (string, optional): In case a text file is passed for `filenames`, the full paths to
    #             the images will be composed from `images_dir` and the names in the text file, i.e. this
    #             should be the directory that contains the images to which the text file refers.
    #             If `filenames_type` is not 'text', then this argument is irrelevant.
    #         labels (string or list, optional): `None` or either a Python list/tuple or a string representing
    #             the path to a pickled file containing a list/tuple. The list/tuple must contain Numpy arrays
    #             that represent the labels of the dataset.
    #         image_ids (string or list, optional): `None` or either a Python list/tuple or a string representing
    #             the path to a pickled file containing a list/tuple. The list/tuple must contain the image
    #             IDs of the images in the dataset.
    #         eval_neutral (string or list, optional): `None` or either a Python list/tuple or a string representing
    #             the path to a pickled file containing a list/tuple. The list/tuple must contain for each image
    #             a list that indicates for each ground truth object in the image whether that object is supposed
    #             to be treated as neutral during an evaluation.
    #         labels_output_format (list, optional): A list of five strings representing the desired order of the five
    #             items class ID, xmin, ymin, xmax, ymax in the generated ground truth data (if any). The expected
    #             strings are 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'.
    #         verbose (bool, optional): If `True`, prints out the progress for some constructor operations that may
    #             take a bit longer.
    #     '''

    #     self.labels_output_format = labels_output_format
    #     self.labels_format = label_format

    #     self.dataset_size = 0

    #     if not filenames is None:
    #         if isinstance(filenames, (list, tuple)):
    #             self.filenames = filenames
    #         elif isinstance(filenames, str):
    #             with open(filenames, 'rb') as f:
    #                 if filenames_type == 'pickle':
    #                     self.filenames = pickle.load(f)
    #                 elif filenames_type == 'text':
    #                     self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
    #                 else:
    #                     raise ValueError("`filenames_type` can be either 'text' or 'pickle'.")
    #         else:
    #             raise ValueError("`filenames` must be either a Python list/tuple or a string representing a filepath (to a pickled or text file). The value you passed is neither of the two.")
            
    #         self.dataset_size = len(self.filenames)
    #         self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
            
    #         self.images = []
    #         if verbose: it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
    #         else: it = self.filenames
    #         for filename in it:
    #             with Image.open(filename) as image:
    #                 self.images.append(np.array(image, dtype=np.uint8))
    #     else:
    #         self.filenames = None

    #     # In case ground truth is available, `self.labels` is a list containing for each image a list (or NumPy array)
    #     # of ground truth bounding boxes for that image.
    #     if not labels is None:
    #         if isinstance(labels, str):
    #             with open(labels, 'rb') as f:
    #                 self.labels = pickle.load(f)
    #         elif isinstance(labels, (list, tuple)):
    #             self.labels = labels
    #         else:
    #             raise ValueError("`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
    #     else:
    #         self.labels = None

    #     if not image_ids is None:
    #         if isinstance(image_ids, str):
    #             with open(image_ids, 'rb') as f:
    #                 self.image_ids = pickle.load(f)
    #         elif isinstance(image_ids, (list, tuple)):
    #             self.image_ids = image_ids
    #         else:
    #             raise ValueError("`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
    #     else:
    #         self.image_ids = None

    #     if not hdf5_dataset_path is None:
    #         self.hdf5_dataset_path = hdf5_dataset_path
    #         self.load_hdf5_dataset(verbose=verbose)
    #     else:
    #         self.hdf5_dataset = None

    # def create_hdf5_dataset(self, file_path='dataset.h5', resize=False, variable_image_size=True, verbose=True):

    #     '''
    #         Converts the currently loaded dataset into a HDF5 file. This HDF5 file contains all
    #         images as uncompressed arrays in a contiguous block of memory, which allows for them
    #         to be loaded faster. Such an uncompressed dataset, however, may take up considerably
    #         more space on your hard drive than the sum of the source images in a compressed format
    #         such as JPG or PNG.
    #         It is recommended that you always convert the dataset into an HDF5 dataset if you
    #         have enugh hard drive space since loading from an HDF5 dataset accelerates the data
    #         generation noticeably.
    #         Note that you must load a dataset (e.g. via one of the parser methods) before creating
    #         an HDF5 dataset from it.
    #         The created HDF5 dataset will remain open upon its creation so that it can be used right
    #         away.
    #         Arguments:
    #             file_path (str, optional): The full file path under which to store the HDF5 dataset.
    #                 You can load this output file via the `DataGenerator` constructor in the future.
    #             resize (tuple, optional): `False` or a 2-tuple `(height, width)` that represents the
    #                 target size for the images. All images in the dataset will be resized to this
    #                 target size before they will be written to the HDF5 file. If `False`, no resizing
    #                 will be performed.
    #             variable_image_size (bool, optional): The only purpose of this argument is that its
    #                 value will be stored in the HDF5 dataset in order to be able to quickly find out
    #                 whether the images in the dataset all have the same size or not.
    #             verbose (bool, optional): Whether or not prit out the progress of the dataset creation.
    #         Returns:
    #             None.
    #     '''

    #     self.hdf5_dataset_path = hdf5_dataset_path

    #     # dataset_size = len(self.filenames)
    #     # Change dataset file from list to dict
    #     dataset_dict = self.filenames
        
    #     # Create the hdf5 file
    #     hdf5_dataset = h5py.File(file_path, 'w')
        
    #     # Create a few attributes that tell us what this dataset contains.
    #     # The dataset will obviously always contain images, but maybe it will
    #     # also contain labels, image IDs, etc.
    #     hdf5_dataset.attrs.create(name='has_labels', data=False, shape=None, dtype=np.bool_)
    #     hdf5_dataset.attrs.create(name='has_image_ids', data=False, shape=None, dtype=np.bool_)
    #     hdf5_dataset.attrs.create(name='has_eval_neutral', data=False, shape=None, dtype=np.bool_)
    #     # It's useful to be able to quickly check whether the images in a dataset all
    #     # have the same size or not, so add a boolean attribute for that.
    #     if variable_image_size and not resize:
    #         hdf5_dataset.attrs.create(name='variable_image_size', data=True, shape=None, dtype=np.bool_)
    #     else:
    #         hdf5_dataset.attrs.create(name='variable_image_size', data=False, shape=None, dtype=np.bool_)

    #     # Create the dataset in which the images will be stored as flattened arrays.
    #     # This allows us, among other things, to store images of variable size.
    #     hdf5_images = hdf5_dataset.create_dataset(name='images',
    #                                                 shape=(dataset_size,),
    #                                                 maxshape=(None),
    #                                                 dtype=h5py.special_dtype(vlen=np.uint8))

    #     # Create the dataset that will hold the image heights, widths and channels that
    #     # we need in order to reconstruct the images from the flattened arrays later.
    #     hdf5_image_shapes = hdf5_dataset.create_dataset(name='image_shapes',
    #                                                     shape=(dataset_size, 3),
    #                                                     maxshape=(None, 3),
    #                                                     dtype=np.int32)

    #     if not (self.labels is None):

    #         # Create the dataset in which the labels will be stored as flattened arrays.
    #         hdf5_labels = hdf5_dataset.create_dataset(name='labels',
    #                                                     shape=(dataset_size,),
    #                                                     maxshape=(None),
    #                                                     dtype=h5py.special_dtype(vlen=np.int32))

    #         # Create the dataset that will hold the dimensions of the labels arrays for
    #         # each image so that we can restore the labels from the flattened arrays later.
    #         hdf5_label_shapes = hdf5_dataset.create_dataset(name='label_shapes',
    #                                                         shape=(dataset_size, 2),
    #                                                         maxshape=(None, 2),
    #                                                         dtype=np.int32)

    #         hdf5_dataset.attrs.modify(name='has_labels', value=True)

    #     if not (self.image_ids is None):

    #         hdf5_image_ids = hdf5_dataset.create_dataset(name='image_ids',
    #                                                         shape=(dataset_size,),
    #                                                         maxshape=(None),
    #                                                         dtype=h5py.special_dtype(vlen=str))

    #         hdf5_dataset.attrs.modify(name='has_image_ids', value=True)

    #     if not (self.eval_neutral is None):

    #         # Create the dataset in which the labels will be stored as flattened arrays.
    #         hdf5_eval_neutral = hdf5_dataset.create_dataset(name='eval_neutral',
    #                                                         shape=(dataset_size,),
    #                                                         maxshape=(None),
    #                                                         dtype=h5py.special_dtype(vlen=np.bool_))

    #         hdf5_dataset.attrs.modify(name='has_eval_neutral', value=True)

    #     if verbose:
    #         tr = trange(dataset_dict.items(), desc='Creating HDF5 dataset', file=sys.stdout)
    #     else:
    #         tr = range(dataset_size)

    #     # Iterate over all images in the dataset.
    #     for i in tr:

    #         # Store the image.
    #         with Image.open(self.filenames[i]) as image:

    #             image = np.asarray(image, dtype=np.uint8)

    #             # Make sure all images end up having three channels.
    #             if image.ndim == 2:
    #                 image = np.stack([image] * 3, axis=-1)
    #             elif image.ndim == 3:
    #                 if image.shape[2] == 1:
    #                     image = np.concatenate([image] * 3, axis=-1)
    #                 elif image.shape[2] == 4:
    #                     image = image[:,:,:3]

    #             if resize:
    #                 image = cv2.resize(image, dsize=(resize[1], resize[0]))

    #             # Flatten the image array and write it to the images dataset.
    #             hdf5_images[i] = image.reshape(-1)
    #             # Write the image's shape to the image shapes dataset.
    #             hdf5_image_shapes[i] = image.shape

    #         # Store the ground truth if we have any.
    #         if not (self.labels is None):

    #             labels = np.asarray(self.labels[i])
    #             # Flatten the labels array and write it to the labels dataset.
    #             hdf5_labels[i] = labels.reshape(-1)
    #             # Write the labels' shape to the label shapes dataset.
    #             hdf5_label_shapes[i] = labels.shape

    #         # Store the image ID if we have one.
    #         if not (self.image_ids is None):

    #             hdf5_image_ids[i] = self.image_ids[i]

    #         # Store the evaluation-neutrality annotations if we have any.
    #         if not (self.eval_neutral is None):

    #             hdf5_eval_neutral[i] = self.eval_neutral[i]

    #     hdf5_dataset.close()

    #     self.hdf5_dataset = h5py.File(file_path, 'r')
    #     self.hdf5_dataset_path = file_path
    #     self.dataset_size = len(self.hdf5_dataset['images'])
    #     self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32) # Instead of shuffling the HDF5 dataset, we will shuffle this index list.


    # def load_hdf5_dataset(self, verbose=True):
    #         '''
    #         Loads an HDF5 dataset that is in the format that the `create_hdf5_dataset()` method
    #         produces.
    #         Arguments:
    #             verbose (bool, optional): If `True`, prints out the progress while loading
    #                 the dataset.
    #         Returns:
    #             None.
    #         '''

    #         self.hdf5_dataset = h5py.File(self.hdf5_dataset_path, 'r')
    #         self.dataset_size = len(self.hdf5_dataset['images'])
    #         self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32) # Instead of shuffling the HDF5 dataset or images in memory, we will shuffle this index list.

    #         if self.load_images_into_memory:
    #             self.images = []
    #             if verbose: tr = trange(self.dataset_size, desc='Loading images into memory', file=sys.stdout)
    #             else: tr = range(self.dataset_size)
    #             for i in tr:
    #                 self.images.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))

    #         # TODO: change to dict to suit generate() and parse_xml()
    #         if self.hdf5_dataset.attrs['has_labels']:
    #             self.labels = []
    #             labels = self.hdf5_dataset['labels']
    #             label_shapes = self.hdf5_dataset['label_shapes']
    #             if verbose: tr = trange(self.dataset_size, desc='Loading labels', file=sys.stdout)
    #             else: tr = range(self.dataset_size)
    #             for i in tr:
    #                 self.labels.append(labels[i].reshape(label_shapes[i]))

    #         if self.hdf5_dataset.attrs['has_image_ids']:
    #             self.image_ids = []
    #             image_ids = self.hdf5_dataset['image_ids']
    #             if verbose: tr = trange(self.dataset_size, desc='Loading image IDs', file=sys.stdout)
    #             else: tr = range(self.dataset_size)
    #             for i in tr:
    #                 self.image_ids.append(image_ids[i])

    #         if self.hdf5_dataset.attrs['has_eval_neutral']:
    #             self.eval_neutral = []
    #             eval_neutral = self.hdf5_dataset['eval_neutral']
    #             if verbose: tr = trange(self.dataset_size, desc='Loading evaluation-neutrality annotations', file=sys.stdout)
    #             else: tr = range(self.dataset_size)
    #             for i in tr:
    #                 self.eval_neutral.append(eval_neutral[i])

    # # Following the repo of rykov8: https://github.com/rykov8/ssd_keras/blob/master/SSD_training.ipynb
    # def generate(self, train=True):

    #     while True:
    #         if train:
    #             shuffle(self.train_keys)
    #             keys = self.train_keys
    #         else:
    #             shuffle(self.val_keys)
    #             keys = self.val_keys
    #         inputs = []
    #         targets = []
    #         for key in keys:            
    #             img_path = self.path_prefix + key
    #             img = imread(img_path).astype('float32')
    #             y = self.gt[key].copy()
    #             if train and self.do_crop:
    #                 img, y = self.random_sized_crop(img, y)
    #             img = imresize(img, self.image_size).astype('float32')
    #             if train:
    #                 shuffle(self.color_jitter)
    #                 for jitter in self.color_jitter:
    #                     img = jitter(img)
    #                 if self.lighting_std:
    #                     img = self.lighting(img)
    #                 if self.hflip_prob > 0:
    #                     img, y = self.horizontal_flip(img, y)
    #                 if self.vflip_prob > 0:
    #                     img, y = self.vertical_flip(img, y)
    #             y = self.bbox_util.assign_boxes(y)
    #             inputs.append(img)                
    #             targets.append(y)
    #             if len(targets) == self.batch_size:
    #                 tmp_inp = np.array(inputs)
    #                 tmp_targets = np.array(targets)
    #                 inputs = []
    #                 targets = []
    #                 yield preprocess_input(tmp_inp), tmp_targets

def parse_xml(
            images_dirs,
            # image_set_filenames,
            annotations_dirs=[],
            classes=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car'
                    ,'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike'
                    ,'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
            labels_output_format=('xmin', 'ymin', 'xmax', 'ymax'), 
            included_classes=['person', 'cat', 'dog', 'bird', 'car', 'tvmonitor', 'chair', 'bottle', 'train', 'sofa'
                            ,'diningtable', 'bus', 'motorbike', 'boat', 'aeroplane', 'pottedplan', 'bicycle'],
            exclude_truncated=False,
            exclude_diffult=False,
            verbose=True):
        
        '''
            This is an XML parser for the Pascal VOC datasets. It might be applicable to other datasets with minor changes to
            the code, but in its current form it expects the data format and XML tags of the Pascal VOC datasets.
            Arguments:
                images_dirs (list): A list of strings, where each string is the path of a directory that
                    contains images that are to be part of the dataset. This allows you to aggregate multiple datasets
                    into one (e.g. one directory that contains the images for Pascal VOC 2007, another that contains
                    the images for Pascal VOC 2012, etc.).
                image_set_filenames (list): A list of strings, where each string is the path of the text file with the image
                    set to be loaded. Must be one file per image directory given. These text files define what images in the
                    respective image directories are to be part of the dataset and simply contains one image ID per line
                    and nothing else.
                annotations_dirs (list, optional): A list of strings, where each string is the path of a directory that
                    contains the annotations (XML files) that belong to the images in the respective image directories given.
                    The directories must contain one XML file per image and the name of an XML file must be the image ID
                    of the image it belongs to. The content of the XML files must be in the Pascal VOC format.
                classes (list, optional): A list containing the names of the object classes as found in the
                    `name` XML tags. Must include the class `background` as the first list item. The order of this list
                    defines the class IDs.
                include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                    are to be included in the dataset. If 'all', all ground truth boxes will be included in the dataset.
                exclude_truncated (bool, optional): If `True`, excludes boxes that are labeled as 'truncated'.
                exclude_difficult (bool, optional): If `True`, excludes boxes that are labeled as 'difficult'.
                ret (bool, optional): Whether or not to return the outputs of the parser.
                verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.
            Returns:
                None
        '''

        images_dirs = images_dirs
        annotations_dirs = annotations_dirs
        # image_set_filenames = image_set_filenames
        classes = classes
        include_classes = included_classes

        filenames = []
        image_ids = []
        labels = {}

        if not annotations_dirs:
            labels = None
            annotations_dirs = [None] * len(images_dirs)

        for images_dir, annotations_dir in zip(images_dirs, annotations_dirs):
        # for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):

            # with open(image_set_filename) as f:
            #     image_ids = [line.strip() for line in f] # Note: These are strings, not integers.
            #     image_ids += image_ids
            
            image_ids = os.listdir(annotations_dir)

            if verbose: it = tqdm(image_ids, desc="Processing image set {}".format(os.path.basename(annotations_dir)))
            else: it = image_ids
            
            for image_id in it:
                
                filename = '{}'.format(image_id.replace('.xml', '')) + '.jpg'
                filenames.append(os.path.join(images_dir, filename))

                if not annotations_dir is None:

                    # For dog-cat redux dataset format
                    with open(os.path.join(annotations_dir, image_id)) as f:
                        soup = BeautifulSoup(f, 'xml')
                    
                    folder = soup.folder.text

                    boxes = [] # store all boxes for this image here
                    one_hot_classes = []
                    # eval_neutr = [] # not implemented yet # We'll store whether a box is annotated as "difficult" here.
                    objects = soup.find_all('object')

                    size_tree = soup.find_all('size')
                    width = float(size_tree[0].width.text)
                    height = float(size_tree[0].height.text)
                    
                    for obj in objects:

                        class_name = obj.find('name', recursive=False).text
                        if not include_classes == 'all' and class_name not in include_classes:
                            continue

                        else:
                            one_hot_class = [0] * len(include_classes)
                            class_index = include_classes.index(class_name)
                            one_hot_class[class_index] = 1
                            
                            # truncated = int(obj.find('truncated', recursive=False).text)
                            # if exclude_truncated and (truncated == 1): continue

                            # exclude_diffult = int(obj.find('difficult', recursive=False).text)
                            # if exclude_diffult and (difficult == 1): continue

                            bndbox = obj.find('bndbox', recursive=False)
                            # Normalize xmin, ymin, xmax, ymax dimensions in float instead of int according to repo by rykov8 
                            xmin = float(bndbox.xmin.text)/width 
                            ymin = float(bndbox.ymin.text)/height
                            xmax = float(bndbox.xmax.text)/width
                            ymax = float(bndbox.ymax.text)/height

                            item_dict = {'folder': folder,
                                        'image_name': filename,
                                        'image_id': image_id,
                                        'class_name': class_name,
                                        # 'class_id': class_id,
                                        # 'pose': pose,
                                        # 'truncated': truncated,
                                        # 'difficult': difficult,
                                        'xmin': xmin,
                                        'ymin': ymin,
                                        'xmax': xmax,
                                        'ymax': ymax}

                            box = []
                            for item in labels_output_format:
                                box.append(item_dict[item])

                            boxes.append(box)
                            one_hot_classes.append(one_hot_class)

                    if not len(boxes) == 0:
                        boxes = np.asarray(boxes)
                        one_hot_classes = np.asarray(one_hot_classes)
                        image_label = np.hstack((boxes, one_hot_classes))
                        labels[filename] = image_label
            
        dataset_size = len(filenames)
        dataset_indices = np.arange(dataset_size, dtype=np.int32)

        pickle.dump(labels, open('gt_modified.pkl', 'wb'))

            # if self.load_images_into_memory:
            #     self.images = []
            #     if verbose: it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            #     else: it = self.filenames
            #     for filename in it:
            #         with Image.open(filename) as image:
            #             self.images.append(np.array(image, dtype=np.uint8))

            # if ret:
            #     return self.images, self.filenames, self.labels, self.image_ids, self.eval_neutral