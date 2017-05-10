from keras import backend as K
import os, sys, time
import numpy as np
import cv2

# Get lists: train_img, train_label, etc.
# Create a method that batch-process the categorical labels to a one-hot matrix and saves them in a database along with the images.
# Create a generator method that yields a tuple of two tensors of shape (batch_size, img_height, img_width, num_channels).
# This method should also imshow() the segmentation of the sample image after each batch.
# Create a method that unpacks the one-hot matrix into a displayable RGB image.

class SegGen(object):

    def __init__(self, data_dir, num_classes):
        self.data_dir = os.getcwd() + data_dir
        self.num_classes = num_classes
        self.file_list = os.listdir(self.data_dir + 'images/')

    def prepareDataset(self):
        for i in self.file_list:
            print(i)

    def labelToOneHot(self, label):
        start = time.clock()
        img_K = K.one_hot(label, 11)
        end = time.clock()
        print(end-start)
        start = time.clock()
        b = np.zeros((label.size, label.max()+1))
        b[np.arange(label.size),label] = 1
        end = time.clock()
        print(end-start)



sg = SegGen('/data/', 11)
img = cv2.imread('data/images/0016E5_08151.png')
sg.labelToOneHot(img)