"""Module docstring"""

import random
import numpy as np
import cv2

def label_to_onehot(label, num_classes):
    """Converts labels (e.g. 2) to one-hot vectors (e.g. [0,0,1,0]).
    Accepts and returns a numpy array."""
    return np.eye(num_classes)[label]

# Accepts and returns a numpy array.
def one_hot_to_label(one_hot):
    """Converts one-hot vectors (e.g. [0,0,1,0]) to labels (e.g. 2).
    Accepts and returns a numpy array."""
    return one_hot.argmax(2).astype('uint8')

def generate_data(data, batch_size, num_classes, augment_data=False):
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
            for outp in sample[1]:
                outputs.append(label_to_onehot(cv2.imread(outp, 0), num_classes))

            input_batch.append(inputs)
            output_batch.append(outputs)


        input_batch = np.array(input_batch)
        output_batch = np.array(output_batch)
        yield (input_batch, output_batch)
