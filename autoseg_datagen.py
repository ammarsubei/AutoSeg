"""Module docstring"""

import os
import numpy as np
import cv2

class Datagen(object):
    """Handles data generation and hides callback creation."""
    def __init__(self, dataset):
        self.dataset = dataset


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
