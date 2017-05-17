<<<<<<< HEAD
import cv2, time, os

image = cv2.imread('./cityscapes/images/val/munster/munster_000097_000019_leftImg8bit.png')
label = cv2.imread('./cityscapes/labels_fine/val/munster/munster_000097_000019_gt_Fine_labelIds.png')
start = time.clock()
image = cv2.resize(image, (480,240))
label = cv2.resize(image, (480,240))
end = time.clock()
print(2500*(end - start))
=======
import numpy as np
import cv2, random

<<<<<<< HEAD
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
=======
label = cv2.imread('../../labels/munich_000188_000019_gtFine_labelIds.png')
print(np.unique(label))
>>>>>>> 7d49b792332f3f8179a95829e2830a107014cb84
>>>>>>> c14cca7c74e9df13616785f846ef1b531e51d7d9
