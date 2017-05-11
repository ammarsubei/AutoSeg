import keras
from keras import backend as K
import numpy as np
import sys
import squeezebuild, squeezecallbacks
from squeezedatagen import SegGen

num_filters = 64
img_height = 480
img_width = 360
img_size = (img_height, img_width)
mask_size = img_size
input_shape = (img_height, img_width, 3)
batch_size = 8
epochs = 500
#model_name = sys.argv[1]
model_name= 'test.h5'

model = squeezebuild.getModel(input_shape, num_filters)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

generator = SegGen(data_dir='/data/', num_classes=12, reinitialize=True)

callbacks = squeezecallbacks.getCallbacks(model_name, patience=batch_size)

model.fit_generator(
    generator.trainingGenerator(batch_size),
    steps_per_epoch=len(generator.training_file_list) / batch_size,
    epochs=epochs,
    callbacks=callbacks)
