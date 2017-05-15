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
import cv2

label = cv2.imread('../../labels/munich_000188_000019_gtFine_labelIds.png')
print(np.unique(label))
>>>>>>> 7d49b792332f3f8179a95829e2830a107014cb84
