import cv2, os, time

start = time.clock()
for path, subdirs, files in os.walk('./cityscapes_orig'):
    for f in files:
        big_file = os.path.join(path, f)
        new_file = big_file.replace('cityscapes_orig', 'cityscapes')
        if big_file.endswith('.png'):
            img = cv2.imread(big_file)
            img = cv2.resize(img, (480,240))
            cv2.imwrite(new_file, img)
            print(new_file)
