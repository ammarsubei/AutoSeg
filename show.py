import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
import numpy as np
import os, sys, time
import autoseg_models
from autoseg_backend import BackendHandler, pixelwise_crossentropy, pixelwise_accuracy

train_encoder = False
num_classes = 10
data_dir = '/cityscapes_1024/'
img_height = 512
img_width = 1024
visualize_while_training = True
dropout_rate = 0.0
img_size = (img_width, img_height)
mask_size = img_size
input_shape = (img_height, img_width, 3)
batch_size = 4
epochs = 10000000
if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name= 'show.h5'

model = autoseg_models.getModel(input_shape=input_shape,
                                num_classes=num_classes,
                                dropout_rate=dropout_rate)

model.load_weights('test.h5', by_name=True)
if not train_encoder:
    for layer in model.layers:
        layer.trainable = False
        if layer.name == "concatenate_8":
            break
    #for layer in model.layers:
        #print(layer.name + ": " + str(layer.trainable))

model.compile(loss=pixelwise_crossentropy, optimizer='adam', metrics=[pixelwise_accuracy])
plot_model(model, to_file='architecture.png', show_shapes=True, show_layer_names=True)

backend = BackendHandler(data_dir=data_dir, num_classes=num_classes, visualize_while_training=visualize_while_training)

callbacks = backend.getCallbacks(model_name, patience=batch_size)

start = time.clock()
model.evaluate_generator(backend.generateData(batch_size=1, validating=True), 100)
end = time.clock()
print("Benchmarked at " + str(100 / (end - start)) + " frames per second.")

model.fit_generator(
    backend.generateData(batch_size),
    steps_per_epoch=len(backend.training_file_list) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=backend.generateData(batch_size, validating=True),
    validation_steps=len(backend.validation_file_list) / batch_size)
    #class_weight=backend.class_weights)
