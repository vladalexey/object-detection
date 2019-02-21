from __future__ import division
import numpy as np 

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Convolution2D, GlobalAveragePooling2D, Dense, ZeroPadding2D
from keras.layers import Reshape, Activation, concatenate, Flatten, add
from keras.regularizers import l2

from ssd_layers import PriorBox, Normalize, Residual_Block

     
def SSD_300(
    input_shape, 
    num_classes=17 + 1,
    min_scale = None,
    max_scale = None,
    aspect_ratios_per_layer= [[2.0],
                            [2.0, 3.0],
                            [2.0, 3.0],
                            [2.0, 3.0],
                            [2.0, 3.0],
                            [2.0, 3.0]],
    variances = [0.1, 0.1, 0.2, 0.2],
    scales = [30, 60, 114, 168, 222, 276, 330],
    # scales = [100, 168, 222, 276, 330],
    clip_boxes = True):


    '''
        Arguments:
            input_shape (tuple): The height and width and channel of the input images.
            min_scale (float): A float in [0, 1], the scaling factor for the size of the generated anchor boxes
                as a fraction of the shorter side of the input image.
            max_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            aspect_ratios_per_layer (list, optional): The list of aspect ratios for which default boxes are to be
                generated for this layer.
            clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
            variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
                its respective variance value.
    '''
    n_predictor_layers = 6 # The number of predictor conv layers in the network is 6 for the original SSD300.

    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))
        

    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    net = {}
    img_height, img_width, img_channels = input_shape[0], input_shape[1], input_shape[2]
    image_size = (input_shape[1], input_shape[0])
    # Block 1
    input_tensor = Input(shape=(img_height, img_width, img_channels))
    net['input'] = input_tensor
    net['conv1_1'] = Convolution2D(16, kernel_size=3, activation='relu', padding='same')(net['input'])
    net['conv1_2'] = Convolution2D(32, kernel_size=3, strides=2, activation='relu', padding='valid', name='conv1_2')(net['conv1_1'])
    
    # Block 2
    net['res2_1'] = Residual_Block(16, net['conv1_2'], name='res2_1')
    net['conv3_1'] = Convolution2D(64, kernel_size=3, activation='relu', padding='valid', strides=2, name='conv3_1')(net['res2_1'])

    # Block 3
    net['res4_1'] = Residual_Block(32, net['conv3_1'], name='res4_1')
    net['res4_2'] = Residual_Block(32, net['res4_1'], name='res4_2')
    net['conv4_3'] = Convolution2D(128, kernel_size=3, activation='relu', padding='valid', strides=2, name='conv4_3')(net['res4_2'])
    
    # Block 4
    net['res5_1'] = Residual_Block(64, net['conv4_3'], name='res5_1')
    net['res5_2'] = Residual_Block(64, net['res5_1'], name='res5_2')
    net['res5_3'] = Residual_Block(64, net['res5_2'], name='res5_3') 
    net['res5_4'] = Residual_Block(64, net['res5_3'], name='res5_4')
    net['conv5_5'] = Convolution2D(256, kernel_size=3, activation='relu', padding='valid', strides=2, name='conv5_5')(net['res5_4'])

    # Block 5
    net['res6_1'] = Residual_Block(128, net['conv5_5'], name='res6_1')
    net['res6_2'] = Residual_Block(128, net['res6_1'], name='res6_2')
    net['res6_3'] = Residual_Block(128, net['res6_2'], name='res6_3') # prediction from 6_3 layer 26
    net['res6_4'] = Residual_Block(128, net['res6_3'], name='res6_4')
    net['conv6_5'] = Convolution2D(512, kernel_size=3, activation='relu', padding='valid', strides=2, name='conv6_5')(net['res6_4'])

    # Block 6
    net['res7_1'] = Residual_Block(256, net['conv6_5'], name='res7_1')
    net['res7_2'] = Residual_Block(256, net['res7_1'], name='res7_2') # prediction from 7_2 layer 34
    
    # Last pool
    net['pool7_3'] = GlobalAveragePooling2D(name='pool7_3')(net['res7_2'])

    # Prediction from conv4_3
    net['conv4_3_norm'] = Normalize(20)(net['conv4_3'])
    num_priors = 3
    net['conv4_3_norm_mbox_loc'] = Convolution2D(num_priors * 4, kernel_size=3, padding='same')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc_flat'] = Flatten()(net['conv4_3_norm_mbox_loc'])
    
    net['conv4_3_norm_mbox_conf'] = Convolution2D(num_priors * num_classes, kernel_size=3, padding='same')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf_flat'] = Flatten()(net['conv4_3_norm_mbox_conf'])

    net['conv4_3_norm_mbox_priorbox'] = PriorBox(image_size, min_size=scales[0], aspect_ratios=aspect_ratios_per_layer[0], variances=variances)(net['conv4_3_norm'])
    
    # Prediction from conv5_5
    num_priors = 6
    net['conv5_5_mbox_loc'] = Convolution2D(num_priors * 4, kernel_size=3, padding='same')(net['conv5_5'])
    net['conv5_5_mbox_loc_flat'] = Flatten()(net['conv5_5_mbox_loc'])
    
    net['conv5_5_mbox_conf'] = Convolution2D(num_priors * num_classes, kernel_size=3, padding='same')(net['conv5_5'])
    net['conv5_5_mbox_conf_flat'] = Flatten()(net['conv5_5_mbox_conf'])

    net['conv5_5_mbox_priorbox'] = PriorBox(image_size, min_size=scales[1], max_size=scales[2], aspect_ratios=aspect_ratios_per_layer[1], variances=variances)(net['conv5_5'])

    # Prediction from conv6_3
    num_priors = 6
    net['conv6_3_mbox_loc'] = Convolution2D(num_priors * 4, kernel_size=3, padding='same')(net['res6_3'])
    net['conv6_3_mbox_loc_flat'] = Flatten()(net['conv6_3_mbox_loc'])
    
    net['conv6_3_mbox_conf'] = Convolution2D(num_priors * num_classes, kernel_size=3, padding='same')(net['res6_3'])
    net['conv6_3_mbox_conf_flat'] = Flatten()(net['conv6_3_mbox_conf'])

    net['conv6_3_mbox_priorbox'] = PriorBox(image_size, min_size=scales[2], max_size=scales[3], aspect_ratios=aspect_ratios_per_layer[2], variances=variances)(net['res6_3'])
    
    # Prediction from conv6_5
    num_priors = 6
    net['conv6_5_mbox_loc'] = Convolution2D(num_priors * 4, kernel_size=3, padding='same')(net['conv6_5'])
    net['conv6_5_mbox_loc_flat'] = Flatten()(net['conv6_5_mbox_loc'])
    
    net['conv6_5_mbox_conf'] = Convolution2D(num_priors * num_classes, kernel_size=3, padding='same')(net['conv6_5'])
    net['conv6_5_mbox_conf_flat'] = Flatten()(net['conv6_5_mbox_conf'])

    net['conv6_5_mbox_priorbox'] = PriorBox(image_size, min_size=scales[3], max_size=scales[4], aspect_ratios=aspect_ratios_per_layer[3], variances=variances)(net['conv6_5'])

    # Prediction from conv7_2
    num_priors = 6
    net['conv7_2_mbox_loc'] = Convolution2D(num_priors * 4, kernel_size=3, padding='same')(net['res7_2'])
    net['conv7_2_mbox_loc_flat'] = Flatten()(net['conv7_2_mbox_loc'])
    
    net['conv7_2_mbox_conf'] = Convolution2D(num_priors * num_classes, kernel_size=3, padding='same')(net['res7_2'])
    net['conv7_2_mbox_conf_flat'] = Flatten()(net['conv7_2_mbox_conf'])

    net['conv7_2_mbox_priorbox'] = PriorBox(image_size, min_size=scales[4], max_size=scales[5], aspect_ratios=aspect_ratios_per_layer[4], variances=variances)(net['res7_2'])
    
    # Prediction from pool7_3
    num_priors = 6
    net['pool7_3_mbox_loc_flat'] = Dense(num_priors * 4)(net['pool7_3'])
    
    net['pool7_3_mbox_conf_flat'] = Dense(num_priors * num_classes)(net['pool7_3'])

    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 512)
    else:
        target_shape = (512, 1, 1)
    
    net['pool7_3_reshaped'] = Reshape(target_shape)(net['pool7_3'])
    net['pool7_3_mbox_priorbox'] = PriorBox(image_size, min_size=scales[5], max_size=scales[6], aspect_ratios=aspect_ratios_per_layer[5], variances=variances)(net['pool7_3_reshaped'])
   
    # Combine predictions
    
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    net['mbox_loc'] = concatenate([net['conv4_3_norm_mbox_loc_flat'],
                            net['conv5_5_mbox_loc_flat'],
                            net['conv6_3_mbox_loc_flat'],
                            net['conv6_5_mbox_loc_flat'],
                            net['conv7_2_mbox_loc_flat'],
                            net['pool7_3_mbox_loc_flat']
                            ], axis=1)
    
    # We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    net['mbox_conf'] = concatenate([net['conv4_3_norm_mbox_conf_flat'],
                            net['conv5_5_mbox_conf_flat'],
                            net['conv6_3_mbox_conf_flat'],
                            net['conv6_5_mbox_conf_flat'],
                            net['conv7_2_mbox_conf_flat'],
                            net['pool7_3_mbox_conf_flat']
                            ], axis=1)
    
    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    net['mbox_prior'] = concatenate([net['conv4_3_norm_mbox_priorbox'],
                            net['conv5_5_mbox_priorbox'],
                            net['conv6_3_mbox_priorbox'],
                            net['conv6_5_mbox_priorbox'],
                            net['conv7_2_mbox_priorbox'],
                            net['pool7_3_mbox_priorbox']
                            ], axis=1)

    # Calculating number of boxes to isolate it using Reshape
    if hasattr(net['mbox_loc'], '_keras_shape'):
        num_boxes = net['mbox_loc']._keras_shape[-1] // 4
    elif hasattr(net['mbox_loc'], '_int_shape'):
        num_boxes = net['mbox_loc']._int_shape[-1] // 4 
    
    # Concatenate all predictions from different layers
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    
    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    net['mbox_loc'] = Reshape((num_boxes, 4))(net['mbox_loc'])

    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    net['mbox_conf'] = Reshape((num_boxes, num_classes))(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax')(net['mbox_conf'])

    net['predictions'] = concatenate([net['mbox_loc'],
                            net['mbox_conf'],
                            net['mbox_prior']
                            ], axis=2)
    model = Model(net['input'], net['predictions'])

    # model = Model(net['input'], net['pool7_3']) # for debugging

    return model