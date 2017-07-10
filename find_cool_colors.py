import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
import os, sys, time, string, random, pickle
import cv2
import autoseg_models
from autoseg_backend import BackendHandler, pixelwise_crossentropy, pixelwise_accuracy, MAPILLARY_COLORS

os.environ["CUDA_VISIBLE_DEVICES"]=""

train_encoder = True
num_classes = 66
data_dir = '/Mapillary/'
img_height = 600
img_width = 800
visualize_while_training = True
dropout_rate = 0.4
weight_decay=0.0002
img_size = (img_width, img_height)
mask_size = img_size
input_shape = (img_height, img_width, 3)
batch_size = 1
epochs = 10000000
model_name= 'visualized_model.h5'


model = autoseg_models.get_SQ(input_shape=input_shape,
                                num_classes=num_classes,
                                dropout_rate=dropout_rate,
                                weight_decay=weight_decay,
                                batch_norm=True)
'''

model = autoseg_models.get_rn38(input_shape=input_shape,
                                num_classes=num_classes)
'''

sgd = keras.optimizers.SGD(lr=1e-8, momentum=0.9)
plot_model(model, to_file='architecture.png', show_shapes=True, show_layer_names=True)
model.save(model_name)
'''
saver = tf.train.Saver()
saver.save(K.get_session(), 'my_test_model')
'''

backend = BackendHandler(data_dir=data_dir, num_classes=num_classes, visualize_while_training=visualize_while_training)

def getID(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def oneHotToLabel(one_hot):
    return one_hot.argmax(2).astype('uint8')

def makeLabelPretty(label):
    prettyLabel = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    colors = MAPILLARY_COLORS

    for i in range(num_classes):
        prettyLabel[np.where( (label==[i]) )] = colors[i]

    return prettyLabel

for x,y in backend.generate_data(batch_size=3, validating=True):
    #predictions = model.predict_on_batch(x)
    while True:
        ID = getID()
        img = x[0]*128.0+128.0
        #print(x[i])
        cv2.imshow('Image', cv2.resize(img.astype('uint8'), (600, 450)))
        cv2.moveWindow('Image', 10, 10)
        cv2.imshow('Ground Truth', cv2.resize(makeLabelPretty( oneHotToLabel(y[0])), (600, 450)))
        cv2.moveWindow('Ground Truth', 850, 10)
        press = 0xFF & cv2.waitKey(0)
        if press == ord('r'):
            print("Randomizing color scheme on keypress r.")
            for i in range(len(MAPILLARY_COLORS)):
                MAPILLARY_COLORS[i] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        elif press == 32: # Space bar.
            print("Next image on space bar.")
            break
        elif press == 27: #Esc
            print("Exiting on Esc.")
            sys.exit()

