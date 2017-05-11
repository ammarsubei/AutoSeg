import keras
from keras import backend as K
import numpy as np
import sys
import autoseg_models
from autoseg_backend import BackendHandler

num_classes = 12
num_filters = 64
img_height = 480
img_width = 360
img_size = (img_width, img_height)
mask_size = img_size
input_shape = (img_width, img_height, 3)
batch_size = 8
epochs = 500
if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name= 'test.h5'

model = autoseg_models.getModel(input_shape, num_classes, num_filters)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

backend = BackendHandler(data_dir='/data/', num_classes=num_classes, reinitialize=True)

callbacks = backend.getCallbacks(model_name, patience=batch_size)

model_name = 'PDC_at_end_12_filters.h5'
model = autoseg_models.getPCModel(input_shape, num_classes, 12)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
callbacks = backend.getCallbacks(model_name, patience=batch_size)
model.fit_generator(
    backend.generateData(batch_size),
    steps_per_epoch=len(backend.training_file_list) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=backend.generateData(batch_size, validating=True),
    validation_steps=len(backend.validation_file_list) / batch_size)

model_name = 'no_PDC_at_end_64_filters.h5'
model = autoseg_models.getModel(input_shape, num_classes, num_filters)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
callbacks = backend.getCallbacks(model_name, patience=batch_size)
model.fit_generator(
    backend.generateData(batch_size),
    steps_per_epoch=len(backend.training_file_list) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=backend.generateData(batch_size, validating=True),
    validation_steps=len(backend.validation_file_list) / batch_size)
    #class_weight=backend.class_weights)

model_name = 'PDC_at_end_64_filters.h5'
model = autoseg_models.getPCModel(input_shape, num_classes, num_filters)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
callbacks = backend.getCallbacks(model_name, patience=batch_size)
model.fit_generator(
    backend.generateData(batch_size),
    steps_per_epoch=len(backend.training_file_list) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=backend.generateData(batch_size, validating=True),
    validation_steps=len(backend.validation_file_list) / batch_size)

model_name = 'no_PDC_at_end_32_filters.h5'
model = autoseg_models.getModel(input_shape, num_classes, 32)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
callbacks = backend.getCallbacks(model_name, patience=batch_size)
model.fit_generator(
    backend.generateData(batch_size),
    steps_per_epoch=len(backend.training_file_list) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=backend.generateData(batch_size, validating=True),
    validation_steps=len(backend.validation_file_list) / batch_size)

model_name = 'PDC_at_end_32_filters.h5'
model = autoseg_models.getPCModel(input_shape, num_classes, 32)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
callbacks = backend.getCallbacks(model_name, patience=batch_size)
model.fit_generator(
    backend.generateData(batch_size),
    steps_per_epoch=len(backend.training_file_list) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=backend.generateData(batch_size, validating=True),
    validation_steps=len(backend.validation_file_list) / batch_size)