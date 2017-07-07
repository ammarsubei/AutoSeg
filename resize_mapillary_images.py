import cv2, os, time

start = time.clock()
i = 0
for path, subdirs, files in os.walk('./Mapillary'):
    os.makedirs(path.replace('Mapillary', 'Mapillary_800'), exist_ok=True)

for path, subdirs, files in os.walk('./Mapillary'):
    for f in files:
        big_file = os.path.join(path, f)
        new_file = big_file.replace('Mapillary', 'Mapillary_800')
        if big_file.endswith('.png') or big_file.endswith('.jpg'):
            img = cv2.imread(big_file)
            img = cv2.resize(img, (800,600))
            cv2.imwrite(new_file, img)
            i += 1
            print(str(i) + " of ~40,000")
