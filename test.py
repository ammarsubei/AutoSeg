import numpy as np
import cv2, os

img = cv2.imread(os.getcwd() + '/sample_label.png', 0)
print(np.unique(img))