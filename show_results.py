import sys
import pickle
import cv2
import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
import autoseg_models
from autoseg_datagen import generate_data
from autoseg_backend import get_callbacks
from autoseg_backend import pixelwise_crossentropy, pixelwise_accuracy
from autoseg_backend import cityscapes, cityscapes_stereo, mapillary

os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

model = load_model(model_name)

def oneHotToLabel(one_hot):
    return one_hot.argmax(2).astype('uint8')

def makeLabelPretty(label, num_classes):
    prettyLabel = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    with open('cityscapes_color_mappings.pickle', 'rb') as f:
        colors =  pickle.load(f)
    for i in range(num_classes):
        prettyLabel[np.where( (label==[i]) )] = colors[i]

    return prettyLabel

for x, y in generate_data(dataset.validation_data, 3, dataset.num_classes):
    predictions = model.predict_on_batch(x)
    for i in range(len(predictions)):
        ID = getID()
        img = x[0][i]*128.0+128.0
        cv2.imshow('Image', img.astype('uint8'))
        cv2.moveWindow('Image', 10, 10)
        cv2.imshow('Ground Truth', makeLabelPretty(oneHotToLabel(y[i])))
        cv2.moveWindow('Ground Truth', 850, 10)
        cv2.imshow('Model Output', makeLabelPretty(oneHotToLabel(predictions[i])))
        cv2.moveWindow('Model Output', 850, 500)
        cv2.waitKey(5000)
