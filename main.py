import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
import os, sys, time
import autoseg_models
from autoseg_backend import BackendHandler, pixelwise_crossentropy, class_weighted_pixelwise_crossentropy, pixelwise_accuracy, mIoU

os.environ["CUDA_VISIBLE_DEVICES"]="0"

train_encoder = False
num_classes = 34
data_dir = '/cityscapes_480/'
img_height = 240
img_width = 480
visualize_while_training = True
dropout_rate = 0.4
weight_decay=0.0002
img_size = (img_width, img_height)
mask_size = img_size
input_shape = (img_height, img_width, 3)
batch_size = 8
epochs = 10000000
if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name= 'main.h5'

model = autoseg_models.getResidualModel(input_shape=input_shape,
                                num_classes=num_classes,
                                residual_encoder_connections=False,
                                dropout_rate=dropout_rate,
                                weight_decay=weight_decay)

sgd = keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=class_weighted_pixelwise_crossentropy, optimizer=sgd, metrics=[pixelwise_accuracy])
K.get_session().run(tf.global_variables_initializer())
K.get_session().run(tf.local_variables_initializer())
plot_model(model, to_file='architecture.png', show_shapes=True, show_layer_names=True)

if model_name in os.listdir(os.getcwd()):
    model.load_weights(model_name, by_name=True)
    if not train_encoder:
        for layer in model.layers:
            layer.trainable = False
            if layer.name == "concatenate_8":
                break
        #for layer in model.layers:
            #print(layer.name + ": " + str(layer.trainable))

backend = BackendHandler(data_dir=data_dir, num_classes=num_classes, visualize_while_training=visualize_while_training)
callbacks = backend.getCallbacks(model_name, patience=100)

model.fit_generator(
    backend.generateData(batch_size),
    steps_per_epoch=len(backend.training_file_list) // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=backend.generateData(batch_size, validating=True),
    validation_steps=len(backend.validation_file_list) // batch_size)
    #class_weight=backend.class_weights)
