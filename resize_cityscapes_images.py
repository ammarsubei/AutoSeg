import cv2, os, time

start = time.clock()
i = 0
for path, subdirs, files in os.walk('./cityscapes_orig'):
    os.makedirs(path.replace('cityscapes_orig', 'cityscapes_512'), exist_ok=True)

for path, subdirs, files in os.walk('./cityscapes_orig'):
    for f in files:
        big_file = os.path.join(path, f)
        new_file = big_file.replace('cityscapes_orig', 'cityscapes_512')
        if big_file.endswith('.png'):
            img = cv2.imread(big_file)
            img = cv2.resize(img, (512,256))
            cv2.imwrite(new_file, img)
            i += 1
            print(str(i) + " of 139024")
