import os

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import cv2

class Dataset(object):
    """
    Contains the parameters of a particular dataset, noteably the file lists.
    """
    def __init__(self, name, num_classes,
                 input_dirs, output_dirs):
        self.name = name
        self.num_classes = num_classes
        training_inputs = get_file_list(input_dirs[0])
        training_outputs = get_file_list(output_dirs[0])
        validation_inputs = get_file_list(input_dirs[1])
        validation_outputs = get_file_list(output_dirs[1])
        self.training_data = (training_inputs, training_outputs)
        self.validation_data = (validation_inputs, validation_outputs)

cityscapes = Dataset('Cityscapes', 34,
                     [['/cityscapes/images_left/train'],
                      ['/cityscapes/images_left/val']],
                     [['/cityscapes/labels_fine/train'],
                      ['/cityscapes/labels_fine/val']])

cityscapes_stereo = Dataset('Cityscapes Stereo', 34,
                            [['/cityscapes/images_left/train',
                              '/cityscapes/images_right/train'],
                             ['/cityscapes/images_left/val',
                              '/cityscapes/images_right/val']],
                            [['/cityscapes/labels_fine/train'],
                             ['/cityscapes/labels_fine/val']])

mapillary = Dataset('Mapillary', 66,
                    [['/Mapillary/training/images'],
                     ['/Mapillary/validation/images']],
                    [['/Mapillary/training/labels'],
                     ['/Mapillary/validation/labels']])

def get_file_list(directories):
    """
    For each directory in a list, recursively find all files in it.
    Return a list of lists of files of the same length as directories.
    """
    file_list = []
    for directory in directories:
        contents = []
        for path, subdirs, files in os.walk(directory):
            for f in files:
                contents.append(os.path.join(path, f))
        contents.sort()
        file_list.append(contents)

    return file_list

def pixelwise_crossentropy(target, output):
    """
    Keras' default crossentropy function wasn't working, not sure why.
    """
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    return -tf.reduce_sum(target * tf.log(output))

def pixelwise_accuracy(y_true, y_pred):
    """
    Same as Keras' default accuracy function, but with axis=-1 changed to axis=2.
    These should be equivalent, I'm not sure why they result in different values.
    """
    return K.mean(K.cast(K.equal(K.argmax(y_true, axis=2),
                                 K.argmax(y_pred, axis=2)),
                         K.floatx()))

def get_callbacks(model_name='test.h5', patience=500, logdir='./logs/default'):
    """
    Returns a standard set of callbacks.
    Kept here mainly to avoid clutter in train_model.py
    """
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

    return [checkpoint, tb]
