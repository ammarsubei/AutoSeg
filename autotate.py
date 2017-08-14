import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
import os, sys, time
import autoseg_models
import cv2

os.environ["CUDA_VISIBLE_DEVICES"]=""

num_classes = 22
data_dir = '/mapillary/'
img_height = 720
img_width = 1280
visualize_while_training = False
dropout_rate = 0.4
weight_decay=0.0002
img_size = (img_width, img_height)
input_shape = (img_height, img_width, 3)
batch_size = 1
epochs = 1000
model_name= 'SQ.h5'

model = autoseg_models.get_SQ(input_shape=input_shape,
                              num_classes=num_classes,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay,
                              batch_norm=True)

if model_name in os.listdir(os.getcwd()):
    model.load_weights(model_name, by_name=True)
    for layer in model.layers:
        layer.trainable = False
        if layer.name == "concatenate_8":
            break

img = (cv2.resize(cv2.imread(sys.argv[1]), img_size).astype('float32') - 256.0) / 256.0
filename = sys.argv[1].split('/')[-1]
save_dir = sys.argv[2]

AUTOSEG_COLORS = [[0, 0, 0],        # void
                  [0, 0, 0],        # ego vehicle
                  [128, 64, 128],   # road / driveable area
                  [244, 35, 232],   # sidewalk / shoulder / paved undriveable area
                  [255, 255, 255],  # white lane marking
                  [255, 255, 0],    # yellow lane marking
                  [0, 0, 142],      # vehicle
                  [220, 20, 60],    # human
                  [70, 70, 70],     # building
                  [153, 153, 153],  # pole
                  [152, 251, 152],  # terrain
                  [107, 142, 35],   # vegetation
                  [250, 170, 30],   # traffic light
                  [220, 220, 0],    # traffic sign
                  [70, 130, 180],   # sky
                  [0, 170, 30],     # water
                  [180, 165, 180],  # guard rail
                  [102, 102, 156],  # barrier
                  [150, 100, 100],  # bridge
                  [150, 120, 90],   # tunnel
                  [0, 192, 0],      # animal
                  [255, 255, 255]]  # snow

def oneHotToLabel(one_hot):
    return one_hot.argmax(2).astype('uint8')

def makeLabelPretty(label):
    prettyLabel = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    colors = AUTOSEG_COLORS

    for i in range(num_classes):
        prettyLabel[np.where( (label==[i]) )] = colors[i]

    return prettyLabel


output = model.predict(np.array([img])).squeeze(0)
gray = oneHotToLabel(output).astype('uint8')
color = makeLabelPretty(gray)[...,::-1]

cv2.imwrite(save_dir + 'gray_' + filename, gray)
cv2.imwrite(save_dir + 'color_' + filename, color)
