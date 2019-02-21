import os, sys
import pickle
from sklearn.externals import joblib
from ssd_utils import generate_priorboxes, parse_xml

# priors_ssd_modified = 'priors_ssd300_modified.pkl'

# print('[INFO] Generating priorboxes for model')
# priors = generate_priorboxes()
# pickle.dump(priors, open(priors_ssd_modified, 'wb'), protocol=4)
# joblib.dump(priors, priors_ssd_modified) 

print('[INFO] Generating groundtruth boxes')
# image_dir_VOC2007 = 'data/VOCdevkit/VOC2007/JPEGImages'
# annotation_dir_VOC2007 = 'data/VOCdevkit/VOC2007/Annotations'
image_dir_VOC2012 = 'data/VOCdevkit/VOC2012/JPEGImages'
annotation_dir_VOC2012 = 'data/VOCdevkit/VOC2012/Annotations'
# train_image_set_filenames = 'data/VOCdevkit/VOC2007/Annotations/test.txt'
# val_image_set_filenames = 'data/VOCdevkit/VOC2007/Annotations/trainval.txt'

parse_xml(images_dirs=[image_dir_VOC2012], annotations_dirs=[annotation_dir_VOC2012])
