import numpy as np
import cv2

<<<<<<< HEAD
weights = [1,2]

target = np.array([ [[0.0,1.0],[1.0,0.0]],
                    [[0.0,1.0],[1.0,0.0]]])

output = np.array([ [[0.5,0.5],[0.9,0.1]],
                    [[0.9,0.1],[0.4,0.6]]])

pixelwise_accuracy = np.mean(np.equal(np.argmax(target, axis=2), np.argmax(output, axis=2)).astype(float))

print(pixelwise_accuracy)
=======
img = cv2.imread('cityscapes_orig/disparity/train/aachen/aachen_000017_000019_disparity.png', 0)
print(img)
cv2.imshow('Hi', img)
cv2.waitKey(5000)
>>>>>>> d2bcc4f914477ba905552e7cdd9ddf87361936aa
