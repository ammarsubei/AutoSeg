import numpy as np
import os, cv2
import pickle

colors = [None]*34
for f in os.listdir('../../labels/'):
    f = f[:27]
    print(f)
    print(colors)
    lbl = cv2.imread('../../labels/' + f + '_labelIds.png', 0)
    col = cv2.imread('../../color/' + f + '_color.png')
    if lbl is not None and col is not None:
        for u in np.unique(lbl):
            if colors[u] is None:
                # find an element [x][y] in lbl with the value u
                for y in range(lbl.shape[0]):
                    for x in range(lbl.shape[1]):
                        if lbl[y][x] == u:
                            colors[u] = col[y][x].tolist()
                            break
            
    if None not in colors:
        break
print("")
print(colors)
with open('cityscapes_color_mappings.pickle', 'wb') as f:
    pickle.dump(colors, f, protocol=pickle.HIGHEST_PROTOCOL)