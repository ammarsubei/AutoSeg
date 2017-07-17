import numpy as np
import cv2

img = cv2.imread('cityscapes_orig/disparity/train/aachen/aachen_000017_000019_disparity.png', 0)
print(img)
cv2.imshow('Hi', img)
cv2.waitKey(5000)
