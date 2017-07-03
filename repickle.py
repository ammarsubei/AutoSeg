import pickle

with open('cityscapes_color_mappings.pickle', 'rb') as f:
    obj = pickle.load(f)
with open('cityscapes_color_mappings.pickle', 'wb') as f:
    pickle.dump(obj, f, protocol=0)
