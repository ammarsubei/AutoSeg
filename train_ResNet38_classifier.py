import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
import os, sys, time
import autoseg_models
from autoseg_backend import BackendHandler, pixelwise_crossentropy, class_weighted_pixelwise_crossentropy, pixelwise_accuracy
from keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"]="1"

train_encoder = True
num_classes = 1000
data_dir = 'imagenet_train/'
img_height = 224
img_width = 224
visualize_while_training = False
dropout_rate = 0.4
weight_decay=0.0002
img_size = (img_width, img_height)
input_shape = (img_height, img_width, 3)
batch_size = 50
epochs = 10000000
if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name= 'ResNet38.h5'

model = autoseg_models.get_rn38_classifier(input_shape=input_shape,
                                num_classes=num_classes)

if model_name in os.listdir(os.getcwd()):
    model.load_weights(model_name, by_name=True)

sgd = keras.optimizers.SGD(lr=1e-8, momentum=0.9, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
plot_model(model, to_file='architecture.png', show_shapes=True, show_layer_names=True)

backend = BackendHandler(data_dir=data_dir, num_classes=num_classes, visualize_while_training=visualize_while_training)
callbacks = backend.get_callbacks(model_name, patience=250, logdir='./logs/ResNet38/')

datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical')


model.fit_generator(
    generator,
    steps_per_epoch=500, #len(backend.training_file_list) // batch_size,
    epochs=epochs,
    callbacks=[ModelCheckpoint('ResNet38_classifier.h5', monitor='loss', save_weights_only=True)])
