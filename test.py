import numpy as np
import cv2, random

img = cv2.imread('../../img.jpg')
img = cv2.resize(img, (800,600))
rows,cols = img.shape[:2]
while True:
    angle = random.gauss(mu=0, sigma=4)
    print(angle)
    scale = random.gauss(mu=1, sigma=0.1)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,scale)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow('Original', img)
    cv2.imshow('Rotated', dst)
    cv2.moveWindow('Original', 10,10)
    cv2.moveWindow('Rotated', 700,10)
    cv2.waitKey(1000)