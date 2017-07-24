from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import os, random, math, string
import pickle
import numpy as np
import cv2
from enum import Enum

class Dataset(object):
    """
    Contains the parameters of a particular dataset, noteably the file lists.
    """
    def __init__(self, name, num_classes,
                 training_input_dirs, training_output_dirs,
                 validation_input_dirs, validation_output_dirs):
        self.name = name
        self.num_classes = num_classes
        self.training_inputs = self.get_file_list(training_input_dirs)
        self.training_outputs = self.get_file_list(training_output_dirs)
        self.validation_inputs = self.get_file_list(validation_input_dirs)
        self.validation_outputs = self.get_file_list(validation_output_dirs)

    def get_file_list(self, directories):
        """
        For each directory in a list, recursively find all files in it.
        Return a list of lists of files of the same length as directories.
        """
        file_list = []
        for directory in directories:
            contents = []
            for path, subdirs, files in os.walk(directory):
                for f in files:
                    contents.append(os.path.join(path,f))
            contents.sort()
            file_list.append(contents)

        return file_list

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
