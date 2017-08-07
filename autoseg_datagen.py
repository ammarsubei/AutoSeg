"""Module docstring"""

import os
import random
import numpy as np
import cv2
from PIL import Image

def label_to_onehot(label, num_classes):
    """Converts labels (e.g. 2) to one-hot vectors (e.g. [0,0,1,0]).
    Accepts and returns a numpy array."""
    return np.eye(num_classes)[label]

# Accepts and returns a numpy array.
def one_hot_to_label(one_hot):
    """Converts one-hot vectors (e.g. [0,0,1,0]) to labels (e.g. 2).
    Accepts and returns a numpy array."""
    return one_hot.argmax(2).astype('uint8')

def get_file_list(directories):
     """
     For each directory in a list, recursively find all files in it.
     Return a list of lists of files of the same length as directories.
     """
     file_list = []
     for directory in directories:
         contents = []
         for path, subdirs, files in os.walk(os.getcwd() + directory):
             for f in files:
                 if directory.find('labels_fine') == -1 or f.find('labelIds') != -1:
                    contents.append(os.path.join(path, f))
         contents.sort()
         file_list.append(contents)

     return file_list

def merge_file_lists(input_files, output_files):
     inputs = []
     for i in range(len(input_files[0])):
         inp = []
         for j in range(len(input_files)):
             inp.append(input_files[j][i])
         inputs.append(inp)
     outputs = []
     for i in range(len(output_files[0])):
         outp = []
         for j in range(len(output_files)):
             outp.append(output_files[j][i])
         outputs.append(outp)

     data = []
     for i in range(len(inputs[0])):
         data.append((inputs[i], outputs[i]))
     return data

#def get_random_crop(img, size):



def generate_cityscapes_data(batch_size, augment_data=False, validating=True):
    """Replaces Keras' native ImageDataGenerator."""
    if not validating:
        training_input_files = get_file_list(['/cityscapes_768/images_left/train'])
        training_output_files = get_file_list(['/cityscapes_768/labels_fine/train'])
        data = merge_file_lists(training_input_files, training_output_files)
    else:
        validation_input_files = get_file_list(['/cityscapes_768/images_left/val'])
        validation_output_files = get_file_list(['/cityscapes_768/labels_fine/val'])
        data = merge_file_lists(validation_input_files, validation_output_files)


    random.shuffle(data)

    i = 0
    while True:
        input_batch = []
        output_batch = []
        for b in range(batch_size):
            if i == len(data):
                i = 0
                random.shuffle(data)
            sample = data[i]
            i += 1

            inputs = []
            for inp in sample[0]:
                inputs.append((cv2.imread(inp).astype(float) - 128) / 128)

            outputs = []
            for outp in sample[1]:
                outputs.append(label_to_onehot(cv2.imread(outp, 0), 34))

            input_batch.append(inputs)
            print(inputs)
            output_batch.append(outputs)


        yield (np.array(input_batch), np.array(output_batch))
