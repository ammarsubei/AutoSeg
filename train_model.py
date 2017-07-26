"""
This script trains a model, either from scratch or from pretrained weights.
All hyperparameters and details of training (which dataset to use, callbacks,
data augmentation) are set here. Model architecture is defined in models.py.

Usage: python train_model.py [model filename] [dataset directory] [flags]
- [model filename] is the file that the model weights will be loaded from/
saved to, e.g. "model.h5"
- [dataset directory] is the directory containing the dataset. See README for
details on expected directory structure.
- The only current flag is -n, which tells the script to ignore any pretrained
weights found and train the model from scratch.
"""

import os
import sys
import time
import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
import autoseg_models
from autoseg_datagen import generate_data
from autoseg_backend import get_callbacks
from autoseg_backend import pixelwise_crossentropy, pixelwise_accuracy
from autoseg_backend import cityscapes, cityscapes_stereo, mapillary

# If you have multiple GPUs, you can use this environment variable to choose
# which one the model should train on. Numbering starts at 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = sys.argv[1]
# Choose which dataset to train on here.
# Dataset params are defined in autoseg_backend.py
dataset = cityscapes_stereo
img_height = 384
img_width = 768
img_size = (img_width, img_height)
input_shape = (img_height, img_width, 3)
batch_size = 1
epochs = 10000000

# The method called here should return a keras.models.Model object, which
# specifies the architecture.
# Note that this method is required even if you are loading weights from a
# pretrained model.
model = autoseg_models.get_dense103(input_shape, dataset.num_classes)

if "-n" not in sys.argv:
    if model_name in os.listdir(os.getcwd()):
        old_model = load_model(model_name)
        # Iterate over the layers of model. If the names match, attempt to copy
        # over the weights from old_model. Catch and ignore ValueError, if the
        # shapes don't match simply don't load those weights.
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        for layer in model.layers:
            if layer.name in layer_dict:
                try:
                    layer.set_weights(layer_dict[layer.name].get_weights())
                except ValueError:
                    print("Ignoring weights for layer with same name but \
                           different shape.")

    else:
        print("FYI: Specified behavior is to load weights, but no \
        weights found. Initializing weights from scratch.")

sgd = keras.optimizers.SGD(lr=1e-8, momentum=0.9, decay=1e-3)
model.compile(loss=pixelwise_crossentropy, optimizer=sgd, metrics=[pixelwise_accuracy])
plot_model(model, to_file='architecture.png', show_shapes=True, show_layer_names=True)
callbacks = get_callbacks(model_name, patience=250, logdir='./logs/SQ/')

start = time.clock()
model.evaluate_generator(generate_data(dataset.training_data, 1, dataset.num_classes), 100)
end = time.clock()
print("Benchmarked at " + str(100 / (end - start)) + " frames per second.")

model.fit_generator(
    generate_data(dataset.training_data, batch_size, dataset.num_classes),
    steps_per_epoch=500, #len(dataset.training_data) // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=generate_data(dataset.validation_data, batch_size,
                                  dataset.num_classes),
    validation_steps=len(dataset.validation_data) // batch_size)
