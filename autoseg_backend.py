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
                 input_dirs, output_dirs,
                 colors):
        self.name = name
        self.num_classes = num_classes
        training_inputs = get_file_list(input_dirs[0])
        training_outputs = get_file_list(output_dirs[0])
        validation_inputs = get_file_list(input_dirs[1])
        validation_outputs = get_file_list(output_dirs[1])
        self.training_data = (training_inputs, training_outputs)
        self.validation_data = (validation_inputs, validation_outputs)
        self.colors = colors

cityscapes = Dataset('Cityscapes', 34,
                     [['/cityscapes/images_left/train'],
                      ['/cityscapes/images_left/val']],
                     [['/cityscapes/labels_fine/train'],
                      ['/cityscapes/labels_fine/val']],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                      [0, 74, 111], [81, 0, 81], [128, 64, 128], [232, 35, 244],
                      [160, 170, 250], [140, 150, 230], [70, 70, 70],
                      [156, 102, 102], [153, 153, 190], [180, 165, 180],
                      [100, 100, 150], [90, 120, 150], [153, 153, 153],
                      [153, 153, 153], [30, 170, 250], [0, 220, 220],
                      [35, 142, 107], [152, 251, 152], [180, 130, 70],
                      [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0],
                      [100, 60, 0], [90, 0, 0], [110, 0, 0], [100, 80, 0],
                      [230, 0, 0], [32, 11, 119]])

cityscapes_stereo = Dataset('Cityscapes Stereo', 34,
                            [['/cityscapes/images_left/train',
                              '/cityscapes/images_right/train'],
                             ['/cityscapes/images_left/val',
                              '/cityscapes/images_right/val']],
                            [['/cityscapes/labels_fine/train'],
                             ['/cityscapes/labels_fine/val']],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                             [0, 74, 111], [81, 0, 81], [128, 64, 128], [232, 35, 244],
                             [160, 170, 250], [140, 150, 230], [70, 70, 70],
                             [156, 102, 102], [153, 153, 190], [180, 165, 180],
                             [100, 100, 150], [90, 120, 150], [153, 153, 153],
                             [153, 153, 153], [30, 170, 250], [0, 220, 220],
                             [35, 142, 107], [152, 251, 152], [180, 130, 70],
                             [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0],
                             [100, 60, 0], [90, 0, 0], [110, 0, 0], [100, 80, 0],
                             [230, 0, 0], [32, 11, 119]])

mapillary = Dataset('Mapillary', 66,
                    [['/Mapillary/training/images'],
                     ['/Mapillary/validation/images']],
                    [['/Mapillary/training/labels'],
                     ['/Mapillary/validation/labels']],
                    [[165, 42, 42], [0, 192, 0], [196, 196, 196],
                     [190, 153, 153], [180, 165, 180], [102, 102, 156],
                     [102, 102, 156], [128, 64, 255], [140, 140, 200],
                     [170, 170, 170], [250, 170, 160], [96, 96, 96],
                     [230, 150, 140], [128, 64, 128], [110, 110, 110],
                     [244, 35, 232], [150, 100, 100], [70, 70, 70],
                     [150, 120, 90], [220, 20, 60], [255, 0, 0],
                     [255, 0, 0], [255, 0, 0], [200, 128, 128],
                     [255, 255, 255], [64, 170, 64], [128, 64, 64],
                     [70, 130, 180], [255, 255, 255], [152, 251, 152],
                     [107, 142, 35], [0, 170, 30], [255, 255, 128],
                     [250, 0, 30], [0, 0, 0], [220, 220, 220],
                     [170, 170, 170], [222, 40, 40], [100, 170, 30],
                     [40, 40, 40], [33, 33, 33], [170, 170, 170],
                     [0, 0, 142], [170, 170, 170], [210, 170, 100],
                     [153, 153, 153], [128, 128, 128], [0, 0, 142],
                     [250, 170, 30], [192, 192, 192], [220, 220, 0],
                     [180, 165, 180], [119, 11, 32], [0, 0, 142],
                     [0, 60, 100], [0, 0, 142], [0, 0, 90],
                     [0, 0, 230], [0, 80, 100], [128, 64, 64],
                     [0, 0, 110], [0, 0, 70], [0, 0, 192], [32, 32, 32],
                     [0, 0, 0], [0, 0, 0]])

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

def get_callbacks(model_name='test.h5', logdir='./logs/DEFAULT'):
    """
    Returns a standard set of callbacks.
    Kept here mainly to avoid clutter in train_model.py
    """
    checkpoint = ModelCheckpoint(
        model_name,
        monitor='val_loss',
        verbose=0,
        save_best_only=True)

    tb = TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
        write_graph=True,
        write_grads=True,
        write_images=True)

    return [checkpoint, tb]
