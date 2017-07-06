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
from autoseg_backend import BackendHandler, pixelwise_crossentropy, pixelwise_accuracy

os.environ["CUDA_VISIBLE_DEVICES"]="1"

train_encoder = True
num_classes = 34
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
    with open('cityscapes_color_mappings.pickle', 'rb') as f:
        colors =  pickle.load(f)

    for i in range(num_classes):
        prettyLabel[np.where( (label==[i]) )] = colors[i]

    return prettyLabel

videogen = skvideo.io.vreader('/home/autobon/AutoSeg/output_5.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/autobon/AutoSeg/segmented_output.avi', fourcc, 20.0, img_size)

for rgb in videogen:
    if True:
        bgr = cv2.resize(rgb[...,::-1], img_size)
        bgr_in = (bgr.astype('float') - 128) / 128
        segmented_frame = makeLabelPretty( oneHotToLabel( model.predict(np.array([bgr_in])).squeeze(0) ) )
        overlay = bgr.copy()
        cv2.addWeighted(segmented_frame, 0.8, bgr, 0.2, 0, overlay)
        out.write(overlay)
        cv2.imshow('Input', bgr)
        cv2.moveWindow('Input', 10, 10)
        cv2.imshow('Output', overlay)
        cv2.moveWindow('Output', 800, 10)
        cv2.waitKey(1)
    else:
        break

out.release()
