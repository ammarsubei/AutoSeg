import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
import os, sys, time, string, random
import cv2
import autoseg_models
from autoseg_backend import BackendHandler, pixelwise_crossentropy, pixelwise_accuracy

os.environ["CUDA_VISIBLE_DEVICES"]="1"

train_encoder = True
num_classes = 34
data_dir = '/cityscapes_1024/'
img_height = 512
img_width = 1024
visualize_while_training = True
dropout_rate = 0.4
weight_decay=0.0002
img_size = (img_width, img_height)
mask_size = img_size
input_shape = (img_height, img_width, 3)
batch_size = 3
epochs = 10000000
if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name= 'show.h5'

model = autoseg_models.getModel(input_shape=input_shape,
                                num_classes=num_classes,
                                residual_encoder_connections=False,
                                dropout_rate=dropout_rate,
                                weight_decay=weight_decay)

if model_name in os.listdir(os.getcwd()):
    model.load_weights('holy_shit_i_think_it_works.h5', by_name=True)
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

backend = BackendHandler(data_dir=data_dir, num_classes=num_classes, visualize_while_training=visualize_while_training)

callbacks = backend.getCallbacks(model_name, patience=batch_size)

def getID(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def makeLabelPretty(label):
    prettyLabel = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    with open('cityscapes_color_mappings.pickle', 'rb') as f:
        self.colors =  pickle.load(f)
        assert self.num_classes <= len(self.colors)

    for i in range(self.num_classes):
        prettyLabel[np.where( (label==[i]) )] = self.colors[i]

    return prettyLabel

for x,y in backend.generateData(batch_size=3, validating=True):
    predictions = model.predict_on_batch(x)
    for i in range(len(predictions)):
        ID = getID()
        cv2.imshow('Image', x[i]*255)
        cv2.moveWindow('Image', 10, 10)
        cv2.imshow('Ground Truth', makeLabelPretty(y[i]))
        cv2.moveWindow('Ground Truth', 850, 10)
        cv2.imshow('Model Output', makeLabelPretty(predictions[i]))
        cv2.moveWindow('Model Output', 850, 500)
        cv2.waitKey(5000)
