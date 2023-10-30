# Copyright (c) 2023, Stefan Toeberg.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# (http://opensource.org/licenses/BSD-3-Clause)
#
# __author__ = "Stefan Toeberg, LUH: IMR"
# __description__ = """
#                   architecture of the used unet model
#                   """

# packages
import absl.logging
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import layers, backend
import numpy as np
import os
import sys
import logging

# get rid of the tensorflow / tensorflow.python.keras console spam
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Keras outputs warnings using `print` to stderr, temporarily directed to devnull
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")

# from tensorflow.python.keras import layers, backend
# from tensorflow.python.keras.models import Model

# get rid of the tensorflow / tensorflow.python.keras console spam

logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logger = tf.get_logger()
logger.setLevel(logging.FATAL)


# keep the settings in the file as used for the training
config = tf.compat.v1.ConfigProto()
# dynamically grow the memory used on the GPU
# config.gpu_config.allow_growth = True
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
# set this TensorFlow session as the default session for Keras
# backend.set_session(sess)
_epsilon = tf.convert_to_tensor(backend.epsilon(), np.float32)


def unet_model(input_shape=(128, 128, 1), n_classes=2):
    # actually two inputs used for the training weightmap and image
    # but when performing the segmentation only the image is necessary
    ip = layers.Input(shape=input_shape)

    # the shape of the weight maps has to be such that it can be element-wise
    # multiplied to the softmax output
    # weight_ip = layers.Input(shape=input_shape[:2] + (n_classes,))

    # layers of the unet architecture
    conv1 = layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(ip)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    #  conv1 = layers.Dropout(0.1)(conv1)
    mpool1 = layers.MaxPool2D()(conv1)

    conv2 = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(mpool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    #  conv2 = layers.Dropout(0.1)(conv2) #0.2
    mpool2 = layers.MaxPool2D()(conv2)

    conv3 = layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(mpool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    #   conv3 = layers.Dropout(0.1)(conv3) #0.3
    mpool3 = layers.MaxPool2D()(conv3)

    conv4 = layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(mpool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Dropout(0.5)(conv4)
    mpool4 = layers.MaxPool2D()(conv4)

    conv5 = layers.Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(mpool4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Dropout(0.5)(conv5)

    up6 = layers.Conv2DTranspose(256, 2, strides=2, kernel_initializer="he_normal", padding="same"
                                 )(conv5)
    up6 = layers.BatchNormalization()(up6)
    conv6 = layers.Concatenate()([up6, conv4])
    conv6 = layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(conv6)
    #   conv6 = layers.Dropout(0.1)(conv6) #0.4

    up7 = layers.Conv2DTranspose(128, 2, strides=2, kernel_initializer="he_normal", padding="same"
                                 )(conv6)
    up7 = layers.BatchNormalization()(up7)
    conv7 = layers.Concatenate()([up7, conv3])
    conv7 = layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    #   conv7 = layers.Dropout(0.1)(conv7) #0.3

    up8 = layers.Conv2DTranspose(64, 2, strides=2, kernel_initializer="he_normal", padding="same"
                                 )(conv7)
    up8 = layers.BatchNormalization()(up8)
    conv8 = layers.Concatenate()([up8, conv2])
    conv8 = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(conv8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(conv8)
    conv8 = layers.BatchNormalization()(conv8)
    #  conv8 = layers.Dropout(0.1)(conv8) #0.2

    up9 = layers.Conv2DTranspose(32, 2, strides=2, kernel_initializer="he_normal", padding="same"
                                 )(conv8)
    up9 = layers.BatchNormalization()(up9)
    conv9 = layers.Concatenate()([up9, conv1])
    conv9 = layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(conv9)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal"
                          )(conv9)
    conv9 = layers.BatchNormalization()(conv9)
    #  conv9 = layers.Dropout(0.1)(conv9)

    c10 = layers.Conv2D(n_classes, 1, activation="softmax", kernel_initializer="he_normal"
                        )(conv9)

    model = Model(inputs=[ip], outputs=[c10])  # , weight_ip

    model.compile(optimizer=Adam(lr=3e-5), loss="categorical_crossentropy", metrics=["accuracy"]
                  )

    return model
