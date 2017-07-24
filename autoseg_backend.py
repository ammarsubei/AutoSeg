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

class BackendHandler(object):
    """Handles data generation and hides callback creation."""
    def __init__(self, data_dir, num_classes, visualize_while_training=False):
        self.data_dir = os.getcwd() + data_dir
        self.num_classes = num_classes
        self.visualize_while_training = visualize_while_training
        self.image_path = self.data_dir + 'images_left/'
        self.label_path = self.data_dir + 'labels_fine/'
        self.disparity_path = self.data_dir + 'disparity/'
        self.cwd_contents = os.listdir(os.getcwd())
        self.training_file_list = self.get_file_list('train/')
        self.validation_file_list = self.get_file_list('val/')
        self.get_class_weights()

    def get_file_list(self, category='train/'):
        """Recursively get raw images and annotations in given dir.
            Return as tuple where first element is the path to the image
            and the second element is the path to the corresponding annotation."""
        file_dir = self.image_path + category
        allfiles = []
        for path, subdirs, files in os.walk(file_dir):
            for f in files:
                f = f.split('_')[0] + '/' + f
                allfiles.append(f)
        allfiles.sort()
        file_list = []
        for f in allfiles:
            input_output = (self.image_path + category + f,
                            (self.image_path + category + f).replace('left', 'right'),
                            self.label_path + category + \
                            f.replace('leftImg8bit', 'gtFine_labelIds'),
                            self.disparity_path + category + f.replace('leftImg8bit', 'disparity'))
            file_list.append(input_output)
        return file_list

    def generate_data(self, batch_size, validating=False,
                      horizontal_flip=True, vertical_flip=False,
                      adjust_brightness=0.1, rotate=5, zoom=0.1):
        """Replaces Keras' native ImageDataGenerator."""
        if validating:
            data = self.validation_file_list
        else:
            data = self.training_file_list
        random.shuffle(data)

        i = 0
        while True:
            image_batch = []
            image_batch_right = []
            label_batch = []
            for batch in range(batch_size):
                if i == len(data):
                    i = 0
                    random.shuffle(data)
                sample = data[i]
                i += 1
                image = cv2.imread(sample[0])
                image_right = cv2.imread(sample[1])
                label = remap_class(cv2.imread(sample[2], 0))

                '''
                # Data Augmentation
                if not validating:
                    # Horizontal flip.
                    if horizontal_flip and random.randint(0, 1):
                        cv2.flip(image, 1)
                        cv2.flip(label, 1)
                    # Vertical flip.
                    if vertical_flip and random.randint(0, 1):
                        cv2.flip(image, 0)
                        cv2.flip(label, 0)
                    # Randomly make image brighter or darker.
                    if adjust_brightness:
                        factor = 1 + abs(random.gauss(mu=0, sigma=adjust_brightness))
                        if random.randint(0, 1):
                            factor = 1 / factor
                        image = 255*((image/255)**factor)
                        image = np.array(image, dtype='uint8')
                    # Randomly rotate image within a certain range.
                    if rotate:
                        angle = random.gauss(mu=0, sigma=rotate)
                    else:
                        angle = 0
                    # Randomly zoom image.
                    if zoom:
                        scale = random.gauss(mu=1, sigma=zoom)
                    else:
                        scale = 1
                    # Apply rotation/zoom calculated above.
                    if rotate or zoom:
                        rows, cols = label.shape
                        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
                        image = cv2.warpAffine(image, M, (cols, rows))
                        label = cv2.warpAffine(label, M, (cols, rows))

                '''
                one_hot = label_to_onehot(label, self.num_classes)
                image_batch.append((image.astype(float) - 128) / 128)
                image_batch_right.append((image_right.astype(float) - 128) / 128)
                label_batch.append(one_hot)
            image_batch = np.array(image_batch)
            image_batch_right = np.array(image_batch_right)
            label_batch = np.array(label_batch)
            yield ([image_batch, image_batch_right], label_batch)

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
