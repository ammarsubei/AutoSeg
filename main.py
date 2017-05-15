import keras
from keras import backend as K
from keras.models import load_model
import numpy as np
import os, sys, time
import autoseg_models
from autoseg_backend import BackendHandler, pixelwise_crossentropy, pixelwise_accuracy

train_encoder = True
num_classes = 34
num_filters = 64
img_height = 480
img_width = 360
img_size = (img_width, img_height)
mask_size = img_size
input_shape = (img_width, img_height, 3)
batch_size = 8
epochs = 10000000
if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name= 'test.h5'

model = autoseg_models.getModel(input_shape, num_classes, num_filters)
if model_name in os.listdir(os.getcwd()):
    model.load_weights('test.h5', by_name=True)
    if not train_encoder:
        for layer in model.layers:
            layer.trainable = False
            if layer.name == "concatenate_8":
                break

model.compile(loss=pixelwise_crossentropy, optimizer='adadelta', metrics=[pixelwise_accuracy], loss_weights=[0.01,0.99])

backend = BackendHandler(data_dir='/cityscapes/', num_classes=num_classes)

callbacks = backend.getCallbacks(model_name, patience=batch_size)

start = time.clock()
model.evaluate_generator(backend.generateData(1), 100)
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
