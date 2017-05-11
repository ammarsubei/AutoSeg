import keras
from keras import backend as K
import numpy as np
import sys
import autoseg_models, autoseg_callbacks
from autoseg_datagen import BackendHandler

num_classes = 12
num_filters = 64
img_height = 480
img_width = 360
img_size = (img_height, img_width)
mask_size = img_size
input_shape = (img_height, img_width, 3)
batch_size = 8
epochs = 500
if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name= 'test.h5'

model = autoseg_models.getModel(input_shape, num_classes, num_filters)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

backend = BackendHandler(data_dir='/data/', num_classes=num_classes, reinitialize=True)

callbacks = autoseg_callbacks.getCallbacks(model_name, patience=batch_size)

model.fit_generator(
    generator.generateData(batch_size),
    steps_per_epoch=len(generator.training_file_list) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=generator.generateData(batch_size, validating=True),
    validation_steps=len(generator.validation_file_list) / batch_size)
