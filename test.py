import numpy as np
import cv2

label = cv2.imread('../../labels/munich_000188_000019_gtFine_labelIds.png')
print(np.unique(label))