import keras
from keras import backend as K
from keras.models import load_model
import numpy as np
import os, sys, time
import autoseg_models
from autoseg_backend import BackendHandler, pixelwise_crossentropy, pixelwise_accuracy

num_classes = 12
num_filters = 64
img_height = 480
img_width = 360
img_size = (img_width, img_height)
mask_size = img_size
input_shape = (img_width, img_height, 3)
batch_size = 1
epochs = 10000000
if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name= 'test.h5'

model = autoseg_models.getModel(input_shape, num_classes, num_filters)
if model_name in os.listdir(os.getcwd()):
    model.load_weights('test.h5')
model.compile(loss=pixelwise_crossentropy, optimizer='adadelta', metrics=[pixelwise_accuracy])

backend = BackendHandler(data_dir='/data/', num_classes=num_classes, reinitialize=False)

callbacks = backend.getCallbacks(model_name, patience=batch_size)
'''
start = time.clock()
model.evaluate_generator(backend.generateData(1), 100)
end = time.clock()
print("Benchmarked at " + str(100 / (end - start)) + " frames per second.")
'''
model.fit_generator(
    backend.generateData(batch_size),
    steps_per_epoch=len(backend.file_list) / batch_size, # change this when re-implementing validation
    epochs=epochs,
    callbacks=callbacks)
    #validation_data=backend.generateData(batch_size, validating=True),
    #validation_steps=len(backend.validation_file_list) / batch_size)
    #class_weight=backend.class_weights)
