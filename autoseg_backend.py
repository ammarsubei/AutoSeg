from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping
from keras import backend as K
import tensorflow as tf
import os, random, math, string
import pickle
import numpy as np
import cv2
from enum import Enum

class Dataset:
    def __init__(self, name, num_classes, ignore_classes, data_dir, directory_structure):
        self.name = name
        self.num_classes = num_classes
        self.ignore_classes = ignore_classes
        self.data_dir = data_dir
        self.directory_structure = directory_structure

IGNORE_CLASSES = [0, 0, 0, 0, 0, 0, 0,  # ignore 'void' class
                  1, 1, 0, 0,           # ignore 'parking', 'rail track'
                  1, 1, 1, 0, 0, 0,     # ignore 'guard rail', 'bridge', 'tunnel'
                  1, 0,                 # ignore 'polegroup'
                  1, 1, 1, 1,           # 'object' class
                  1, 1, 1,              # 'nature' and 'sky' classes
                  1, 1,                 # 'human' class
                  1, 0, 0, 1, 1, 1]     # ignore 'caravan', 'trailer'

def label_to_onehot(label, num_classes):
    """Converts labels (e.g. 2) to one-hot vectors (e.g. [0,0,1,0]).
    Accepts and returns a numpy array."""
    return np.eye(num_classes)[label]

# Accepts and returns a numpy array.
def one_hot_to_label(one_hot):
    """Converts one-hot vectors (e.g. [0,0,1,0]) to labels (e.g. 2).
    Accepts and returns a numpy array."""
    return one_hot.argmax(2).astype('uint8')

def pixelwise_crossentropy(target, output):
    """Keras' default crossentropy function wasn't working, not sure why."""
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    return -tf.reduce_sum(target * IGNORE_CLASSES * tf.log(output))

def pixelwise_accuracy(y_true, y_pred):
    """Same as Keras' default accuracy function, but with axis=-1 changed to axis=2.
        These should be equivalent, I'm not sure why they result in different values."""
    return K.mean(K.cast(K.equal(K.argmax(y_true, axis=2),
                         K.argmax(y_pred, axis=2)),
                         K.floatx()))

    def get_callbacks(self, model_name='test.h5', patience=500, logdir='./logs/default'):
        """Returns a standard set of callbacks.
            Kept here mainly to avoid clutter in main.py"""
        checkpoint = ModelCheckpoint(
            model_name,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True)

        tb = TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            write_graph=True,
            write_grads=True,
            write_images=True)

        early = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

        return [checkpoint, tb]
