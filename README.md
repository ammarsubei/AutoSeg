# AutoSeg

AutoSeg is a collection of scripts useful for training semantic segmentation models using Keras.

The master branch trains traditional monoscopic models on the Mapillary dataset.
The depth branch trains experimental stereoscopic models on the CityScapes dataset.
This readme discusses the master branch.

The most important files in the repo are:
- train_SQ.py
- autoseg_models.py
- autoseg_backend.py

## train_SQ.py
SQ is the model discussed in the paper Speeding up Semantic Segmentation for Autonomous Driving, and gives the best results.
The train_ResNet38.py script is a failed experiment and can be ignored.

The usage of this script is: python train_SQ.py WEIGHTS.h5

The architecture of SQ is defined in autoseg_models.py. train_SQ.py will create an empty model based on this architecture and load weights by name from WEIGHTS.h5.
This allows the re-use of weights after changes to the architecture - just make sure your layer names are consistent. If you change the shape of a layer, you must change its name as well, otherwise you will get an error when trying to load the old weights into it.

IMPORTANT NOTE: WEIGHTS.h5 will be overwritten as the model trains. The model overwrites WEIGHTS.h5 every time there is an improvement in the val_loss metric.

SQ is fully convolutional, and works on images of any size. To change the size of the input image, simply change the hyperparameters in train_SQ.py as well as the variable at the top of autoseg_backend.py.

Weights files are not in the repo due to their size, see the copy of the repo on Jeff's machine. The state-of-the art monoscopic model is SQ_mapillary.h5.

This file is not well commented, but see the generic train_model.py file under the development branch.

## autoseg_models.py
This script defines the architecture for various models, including SQ. Make changes to the architecture by either editing the get_SQ function or creating a new function which returns a model. If the latter, you must edit train_SQ.py to call this new function instead.

## autoseg_backend.py
Defines various backends such as image data generation. Well-commented.

## show_results.py
Run this script when a potential investor is coming over. Runs inference and shows the results. Works mostly the same as train_SQ.py Usage: python show_results.py WEIGHTS.h5
