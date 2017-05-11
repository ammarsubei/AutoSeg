from keras import backend as K
import os, random
import numpy as np
import cv2
import pickle

# Get lists: train_img, train_label, etc.
# Create a method that batch-process the categorical labels to a one-hot matrix and saves them in a database along with the images.
# Create a generator method that yields a tuple of two tensors of shape (batch_size, img_height, img_width, num_channels).
# This method should also imshow() the segmentation of the sample image after each batch.

# Accepts and returns a numpy array.
def labelToOneHot(label, num_classes):
    return np.eye(num_classes)[label]

# Accepts and returns a numpy array.
def oneHotToLabel(self,one_hot):
    return one_hot.argmax(2)

class SegGen(object):

    def __init__(self, data_dir, num_classes, reinitialize=False):
        self.data_dir = os.getcwd() + data_dir
        self.num_classes = num_classes
        self.image_path = self.data_dir + 'images/'
        self.label_path = self.data_dir + 'labels/'
        self.file_list = os.listdir(self.data_dir + 'images/')
        cwd_contents = os.listdir(os.getcwd())
        # Sort the data into training and validation sets, or load already sorted sets.
        if reinitialize or 'training_file_list.pickle' not in cwd_contents or 'validation_file_list.pickle' not in cwd_contents:
            print("Splitting training/validation data 80/20...")
            random.shuffle(self.file_list)
            breakpoint = int(len(self.file_list) * 0.8)
            self.training_file_list, self.validation_file_list = self.file_list[:breakpoint], self.file_list[breakpoint:]
            # pickle the files
            with open('training_file_list.pickle', 'wb') as f:
                pickle.dump(self.training_file_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open('validation_file_list.pickle', 'wb') as f:
                pickle.dump(self.validation_file_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Loading training/validation split...")
            with open('training_file_list.pickle', 'rb') as f:
                self.training_file_list =  pickle.load(f)
            with open('validation_file_list.pickle', 'rb') as f:
                self.validation_file_list = pickle.load(f)

    def generateData(self, batch_size, validating=False):
        if validating:
            data = self.validation_file_list
        else:
            data = self.training_file_list
        i = 0
        while True:
            image_batch = []
            label_batch = []
            for b in range(batch_size):
                if i == len(data):
                    i = 0
                sample = data[i]
                i += 1
                image = cv2.imread(self.image_path + sample) / 255
                label = cv2.imread(self.label_path + sample, 0)
                image = image.swapaxes(0,1)
                label = label.swapaxes(0,1)
                one_hot = labelToOneHot(label, self.num_classes)
                image_batch.append(image)
                label_batch.append(one_hot)
            image_batch = np.array(image_batch)
            label_batch = np.array(label_batch)
            yield (image_batch, label_batch)

    

    

#sg = SegGen('/data/', 11)
#print(next(sg.trainingGenerator(11)))
