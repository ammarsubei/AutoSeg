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
from enum import Enum
import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
import autoseg_models
from autoseg_backend import BackendHandler, pixelwise_crossentropy, pixelwise_accuracy, Datasets

# If you have multiple GPUs, you can use this environment variable to choose
# which one the model should train on. Numbering starts at 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = sys.argv[1]
dataset = Datasets.CITYSCAPES_STEREO # Choose which dataset to train on here.
img_height = 384
img_width = 768
img_size = (img_width, img_height)
input_shape = (img_height, img_width, 3)
batch_size = 1
epochs = 10000000


if dataset == Datasets.CAMVID:
    num_classes = 32
elif dataset == Datasets.CITYSCAPES or dataset == Datasets.CITYSCAPES_STEREO:
    num_classes = 34
elif dataset == Datasets.MAPILLARY:
    num_classes = 66
else:
    num_classes = None
    # throw an error of some kind

# The method called here should return a keras.models.Model object, which
# specifies the architecture.
# Note that this method is required even if you are loading weights from a
# pretrained model.
model = autoseg_models.get_dense103(input_shape, num_classes)

if "-n" not in sys.argv:
    if model_name in os.listdir(os.getcwd()):
        old_model = load_model(model_name)
        # Iterate over the layers of model. If the names match, attempt to copy
        # over the weights from old_model. Catch and ignore ValueError, if the
        # shapes don't match simply don't load those weights.
    else:
        print("FYI: Specified behavior is to load weights, but no \
        weights found. Initializing weights from scratch.")

sgd = keras.optimizers.SGD(lr=1e-4, decay=0.995)
model.compile(loss=pixelwise_crossentropy, optimizer=sgd, metrics=[pixelwise_accuracy])
plot_model(model, to_file='architecture.png', show_shapes=True, show_layer_names=True)
backend = BackendHandler(data_dir=data_dir, num_classes=num_classes, visualize_while_training=visualize_while_training)
callbacks = backend.get_callbacks(model_name, patience=250, logdir='./logs/SQ/')

start = time.clock()
model.evaluate_generator(backend.generate_data(1), 100)
end = time.clock()
print("Benchmarked at " + str(100 / (end - start)) + " frames per second.")

model.fit_generator(
    backend.generate_data(batch_size),
    steps_per_epoch=500, #len(backend.training_file_list) // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=backend.generate_data(batch_size, validating=True),
    validation_steps=len(backend.validation_file_list) // batch_size)
    #class_weight=backend.class_weights)
