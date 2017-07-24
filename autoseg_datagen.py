"""Module docstring"""

import os
import numpy as np
import cv2

class Datagen(object):
    """Handles data generation."""
    def __init__(self,
                 root_data_dir,
                 train_input_dir_list,
                 train_output_dir_list,
                 validate=True,
                 val_input_dir_list,):

        path, subdirs, images = os.walk(path_to_train_images)
