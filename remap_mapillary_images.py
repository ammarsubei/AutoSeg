import cv2, os, time
import numpy as np
from PIL import Image

def remap_class(arr):
    simple = arr.copy()
    simple[arr == 0] = 20
    simple[arr == 1] = 20
    simple[arr == 2] = 17
    simple[arr == 3] = 8
    simple[arr == 4] = 16
    simple[arr == 5] = 8
    simple[arr == 6] = 8
    simple[arr == 7] = 3
    simple[arr == 8] = 2
    simple[arr == 9] = 17
    simple[arr == 10] = 3
    simple[arr == 11] = 3
    simple[arr == 12] = 3
    simple[arr == 13] = 2
    simple[arr == 14] = 3
    simple[arr == 15] = 3
    simple[arr == 16] = 18
    simple[arr == 17] = 8
    simple[arr == 18] = 19
    simple[arr == 19] = 7
    simple[arr == 20] = 7
    simple[arr == 21] = 7
    simple[arr == 22] = 7
    simple[arr == 23] = 2
    simple[arr == 24] = 4
    simple[arr == 25] = 10
    simple[arr == 26] = 10
    simple[arr == 27] = 14
    simple[arr == 28] = 21
    simple[arr == 29] = 10
    simple[arr == 30] = 11
    simple[arr == 31] = 15
    simple[arr == 32] = 0
    simple[arr == 33] = 8
    simple[arr == 34] = 8
    simple[arr == 35] = 9
    simple[arr == 36] = 0
    simple[arr == 37] = 0
    simple[arr == 38] = 8
    simple[arr == 39] = 8
    simple[arr == 40] = 8
    simple[arr == 41] = 2
    simple[arr == 42] = 8
    simple[arr == 43] = 2
    simple[arr == 44] = 9
    simple[arr == 45] = 9
    simple[arr == 46] = 13
    simple[arr == 47] = 9
    simple[arr == 48] = 12
    simple[arr == 49] = 13
    simple[arr == 50] = 13
    simple[arr == 51] = 8
    simple[arr == 52] = 6
    simple[arr == 53] = 6
    simple[arr == 54] = 6
    simple[arr == 55] = 6
    simple[arr == 56] = 6
    simple[arr == 57] = 6
    simple[arr == 58] = 6
    simple[arr == 59] = 6
    simple[arr == 60] = 6
    simple[arr == 61] = 6
    simple[arr == 62] = 6
    simple[arr == 63] = 6
    simple[arr == 64] = 1
    simple[arr == 65] = 0

    return simple

start = time.clock()
i = 0
for path, subdirs, files in os.walk('./mapillary/labels'):
    os.makedirs(path.replace('labels', 'labels_simple'), exist_ok=True)

for path, subdirs, files in os.walk('./mapillary/labels'):
    for f in files:
        big_file = os.path.join(path, f)
        new_file = big_file.replace('labels', 'labels_simple')
        if big_file.endswith('.png'):
            img = np.array(Image.open(big_file))
            img = remap_class(img)
            cv2.imwrite(new_file, img)
            i += 1
            print(str(i) + " of ~10,000")
