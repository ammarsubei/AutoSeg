import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
import os, sys, time, string, random, pickle
import cv2
import skvideo.io
import autoseg_models
from autoseg_backend import BackendHandler, pixelwise_crossentropy, pixelwise_accuracy, MAPILLARY_COLORS

MAPILLARY_COLORS = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153], [180, 165, 180], [128, 64, 128],[128, 64, 128], [128, 64, 128], [128, 64, 128], [170, 170, 170], [250, 170, 160], [96, 96, 96], [230, 150, 140], [128, 64, 128], [110, 110, 110], [128, 64, 128], [107, 142, 35], [70, 70, 70], [150, 120, 90], [220, 20, 60], [255, 0, 0], [255, 0, 0], [255, 0, 0], [200, 128, 128], [255, 255, 255], [64, 170, 64], [128, 64, 64], [70, 130, 180], [255, 255, 255], [152, 251, 152], [107,
    142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30], [0, 0, 0], [220, 220, 220], [170, 170, 170], [222, 40, 40], [100, 170, 30], [40, 40, 40], [33, 33, 33], [170, 170, 170], [0, 0, 142], [170, 170, 170], [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 142], [250, 170, 30], [192, 192, 192], [220, 220, 0], [180, 165, 180], [119, 11, 32], [0, 0, 142], [107, 142, 35], [0, 0, 142], [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0, 70], [0, 0, 192], [32, 32, 32], [0, 0, 0], [0, 0, 0]]
'''
temp = np.zeros((100,100,3))
for c in MAPILLARY_COLORS:
    if True:
        temp[:] = c
        print(c)
        cv2.imshow('Temp', temp.astype('uint8')[...,::-1])
        cv2.waitKey(1000)
'''
os.environ["CUDA_VISIBLE_DEVICES"]="1"

train_encoder = True
num_classes = 66
data_dir = '/cityscapes_800/'
img_height = 720
img_width = 1280
visualize_while_training = True
dropout_rate = 0.4
weight_decay=0.0002
img_size = (img_width, img_height)
mask_size = img_size
input_shape = (img_height, img_width, 3)
batch_size = 3
epochs = 10000000
model_name= 'visualized_model.h5'

model = autoseg_models.get_SQ(input_shape=input_shape,
                                num_classes=num_classes,
                                dropout_rate=dropout_rate,
                                weight_decay=weight_decay,
                                batch_norm=True)

if model_name in os.listdir(os.getcwd()):
    model.load_weights(sys.argv[1], by_name=True)
    if not train_encoder:
        for layer in model.layers:
            layer.trainable = False
            if layer.name == "concatenate_8":
                break
        #for layer in model.layers:
            #print(layer.name + ": " + str(layer.trainable))

sgd = keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=pixelwise_crossentropy, optimizer=sgd, metrics=[pixelwise_accuracy])
plot_model(model, to_file='architecture.png', show_shapes=True, show_layer_names=True)
model.save(model_name)

def oneHotToLabel(one_hot):
    return one_hot.argmax(2).astype('uint8')

def makeLabelPretty(label):
    prettyLabel = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    colors = MAPILLARY_COLORS

    for i in range(num_classes):
        prettyLabel[np.where( (label==[i]) )] = colors[i]

    return prettyLabel

videogen = skvideo.io.vreader('/home/autobon/AutoSeg/output_5.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/autobon/AutoSeg/segmented_output.avi', fourcc, 20.0, img_size)

for rgb in videogen:
    if True:
        bgr = cv2.resize(rgb[...,::-1], (1280,720))#[0:360, 0:1280]
        bgr_in = (bgr.astype('float') - 128) / 128
        segmented_frame = makeLabelPretty( oneHotToLabel( model.predict(np.array([bgr_in])).squeeze(0) ) )
        overlay = bgr.copy()
        cv2.addWeighted(segmented_frame[...,::-1], 0.8, bgr, 0.2, 0, overlay)
        out.write(overlay)
        cv2.imshow('Input', bgr)
        cv2.moveWindow('Input', 10, 10)
        cv2.imshow('Output', overlay)
        cv2.moveWindow('Output', 800, 10)
        cv2.waitKey(1)
    else:
        break

out.release()
