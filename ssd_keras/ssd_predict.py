import tensorflow as tf 
import keras
from keras.models import load_model
from keras.preprocessing import image

from ssd_class_300 import SSD_300
from ssd_training import MultiBoxLoss
from ssd_utils import BBoxUtil, generate_priorboxes, DataGen, parse_xml
from ssd_layers import Normalize, PriorBox, Residual_Block

from imageio import imread
from skimage import img_as_float32
from skimage.transform import resize
import matplotlib.pyplot as plt 
import numpy as np 
import h5py
import pickle
import os, sys

sys.path.append(os.getcwd())

NUM_CLASSES = 17 + 1   # classes + 1 for background (e.g. for PASCAL07 it is 20 + 1)
BS = 32
NUM_EPOCH = 5
input_shape = (300, 300, 3)
BASE_LR = 3e-4

image_dir_VOC2012 = 'data/VOCdevkit/VOC2012/JPEGImages'
annotation_dir_VOC2012 = 'data/VOCdevkit/VOC2012/Annotations'

priors = generate_priorboxes()
bbox_util = BBoxUtil(NUM_CLASSES, priors=priors)

gt = pickle.load(open('gt_modified.pkl', 'rb'))
keys = sorted(gt.keys())

print('[INFO] All images:', len(keys))
NUM_TRAIN = int(round(0.8 * len(keys)))
print(NUM_TRAIN)
NUM_TRAIN = int(round( NUM_TRAIN * 0.01))

train_keys = keys[:NUM_TRAIN]
val_keys = keys[NUM_TRAIN:NUM_TRAIN * 2]
# NUM_VAL = len(val_keys)
# print('[INFO] Training images: {}'.format(NUM_TRAIN))
# print('[INFO] Validation images: {}'.format(NUM_VAL))

# train_gen = DataGen(gt, bbox_util, BS, image_dir_VOC2012, train_keys, val_keys, (input_shape[0], input_shape[1]))

inputs = []
images = []
key = sorted(val_keys)[0]
img_path = os.path.join(image_dir_VOC2012, key)
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
img = img_as_float32(img)
images.append(imread(img_path))
inputs.append(img.copy())
 
model = load_model('saved_model-01-4.0289.hdf5', custom_objects={'Normalize':Normalize, 'PriorBox':PriorBox, 'ResidualBlock':Residual_Block})

preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)

for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.get_cmap('Spectral') 
    colors = colors(np.linspace(0, 1, 4)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
#         label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    
    plt.show()