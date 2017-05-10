import cv2
print(cv2.__version__)
sample_img = cv2.imread('data/images/0016E5_08151.png')
cv2.imshow('Help', sample_img)
cv2.waitKey(0)