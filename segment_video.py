import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
import os, sys, time
import autoseg_models
from autoseg_backend import BackendHandler, pixelwise_crossentropy, pixelwise_accuracy
import cv2

num_classes = 34
img_height = 720
img_width = 1280
img_size = (img_width, img_height)
input_shape = (img_height, img_width, 3)
batch_size = 4
cap = cv2.VideoCapture(sys.argv[1])
out = cv2.VideoWriter('output.avi',fourcc, 20.0, img_size)

model = autoseg_models.getModel(input_shape=input_shape,
                                num_classes=num_classes,
                                residual_encoder_connections=True,
                                dropout_rate=0.0)

model.load_weights('main.h5')

sgd = keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=pixelwise_crossentropy, optimizer=sgd, metrics=[pixelwise_accuracy])

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        seg = model.predict(np.array([frame]), batch_size=1)
        out.write(seg)
        cv2.imshow('Input', frame)
        cv2.imshow('Output', seg)
        cv2.waitKey(1)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()