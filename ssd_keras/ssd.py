import numpy as np 
import tensorflow as tf 

import keras
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing import image
# from keras.applications.imagenet_utils import preprocess_input

from ssd_class_300 import SSD_300
from ssd_training import MultiBoxLoss
from ssd_utils import BBoxUtil, generate_priorboxes, DataGen, parse_xml

import os, sys
import pickle
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())

NUM_CLASSES = 17 + 1   # classes + 1 for background (e.g. for PASCAL07 it is 20 + 1)
BS = 64
NUM_EPOCH = 200
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
train_keys = keys[:NUM_TRAIN]
val_keys = keys[NUM_TRAIN:]
NUM_VAL = len(val_keys)
print('[INFO] Training images: {}'.format(NUM_TRAIN))
print('[INFO] Validation images: {}'.format(NUM_VAL))

# train_gen = DataGen(gt, bbox_util, BS, image_dir_VOC2012, train_keys, val_keys, (input_shape[0], input_shape[1]))
 
model = SSD_300(input_shape, num_classes=NUM_CLASSES)
model.summary()

def schedule(epoch, decay=0.9):
    return BASE_LR * decay ** (epoch) 

filepath = "saved_model-{epoch:02d}-{val_loss:.4f}.hdf5"
callbacks = [ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1, save_weights_only=True),
            # LearningRateScheduler(schedule, verbose=1),
            EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
optimizer = Adam(lr=BASE_LR)

model.compile(optimizer, loss=MultiBoxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss, metrics=['accuracy'])
history = model.fit_generator(train_gen.generate(train=True), train_gen.train_batches // BS, 
                                NUM_EPOCH, verbose=1, callbacks=callbacks,
                                validation_data=train_gen.generate(train=False),
                                validation_steps=train_gen.val_batches // BS)

# Loss Curves
N = NUM_EPOCH
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("Curves_Plot.png")
plt.show()