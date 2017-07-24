import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
import os, sys, time
import autoseg_models
from autoseg_backend import BackendHandler, pixelwise_crossentropy, class_weighted_pixelwise_crossentropy, pixelwise_accuracy

os.environ["CUDA_VISIBLE_DEVICES"]="1"

train_encoder = True
num_classes = 34
data_dir = '/cityscapes_800/'
img_height = 400
img_width = 800
visualize_while_training = False
dropout_rate = 0.4
weight_decay=0.0002
img_size = (img_width, img_height)
input_shape = (img_height, img_width, 3)
batch_size = 1
epochs = 10000000
if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name= 'ResNet38.h5'

model = autoseg_models.get_rn38(input_shape=input_shape,
                                num_classes=num_classes)

if model_name in os.listdir(os.getcwd()):
    model.load_weights(model_name, by_name=True)
    if not train_encoder:
        for layer in model.layers:
            layer.trainable = False
            if layer.name == "concatenate_8":
                break
        #for layer in model.layers:
            #print(layer.name + ": " + str(layer.trainable))

sgd = keras.optimizers.SGD(lr=1e-8, momentum=0.9, decay=1e-5)
model.compile(loss=pixelwise_crossentropy, optimizer=sgd, metrics=[pixelwise_accuracy])
plot_model(model, to_file='architecture.png', show_shapes=True, show_layer_names=True)

backend = BackendHandler(data_dir=data_dir, num_classes=num_classes, visualize_while_training=visualize_while_training)
callbacks = backend.get_callbacks(model_name, patience=250, logdir='./logs/ResNet38/')

start = time.clock()
model.evaluate_generator(backend.generate_data(1), 100)
end = time.clock()
print("Benchmarked at " + str(100 / (end - start)) + " frames per second.")

model.fit_generator(
    backend.generate_data(batch_size),
    steps_per_epoch=200, #len(backend.training_file_list) // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=backend.generate_data(batch_size, validating=True),
    validation_steps=len(backend.validation_file_list) // batch_size)
    #class_weight=backend.class_weights)
