"""Module docstring"""

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


def generate_cityscapes_data(dataset, data, batch_size, num_classes, augment_data=False):
    """Replaces Keras' native ImageDataGenerator."""
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
            if dataset == 'Mapillary':
                for outp in sample[1]:
                    outputs.append(label_to_onehot(Image.open(outp), num_classes))
            else:
                for outp in sample[1]:
                    outputs.append(label_to_onehot(cv2.imread(outp, 0), num_classes))

            input_batch.append(inputs)
            output_batch.append(outputs)


        input_batch = np.array(input_batch)
        output_batch = np.array(output_batch)
        yield (input_batch, output_batch)
