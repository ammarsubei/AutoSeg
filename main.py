import keras
from keras.models import Model, load_model
from keras.utils import plot_model, to_categorical
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.engine.topology import Layer
from keras import backend as K
import numpy as np
import sys

num_filters = 64
img_height = 480
img_width = 360
img_size = (img_height, img_width)
mask_size = img_size
input_shape = (img_height, img_width, 3)
batch_size = 1
epochs = 500
steps_per_epoch = int(600/batch_size) + 1
validation_steps = int(101/batch_size) + 1
seed = 1
model_name= sys.argv[1]

def addFireModule(x):
	squeeze = Conv2D(num_filters, (3,3), padding='same', activation='elu')(x)
	expand1 = Conv2D(num_filters, (1,1), padding='same', activation='elu')(squeeze)
	expand3 = Conv2D(num_filters, (3,3), padding='same', activation='elu')(squeeze)
	c = concatenate([expand1, expand3])
	
	return c

def addParallelDilatedConvolution(x):
	conv1 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=1)(x)
	conv2 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=2)(x)
	conv4 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=4)(x)
	conv8 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=8)(x)
	c = concatenate([conv1, conv2, conv4, conv8])

	return c

def addBypassRefinementModule(high, low):
	preConv = Conv2D(num_filters, (3,3), padding='same', activation='elu')(low)
	c = concatenate([preConv, high])
	postConv = Conv2D(num_filters, (3,3), padding='same', activation='elu')(c)

	return postConv

def getModel():
	i = Input(input_shape)
	convI = Conv2D(num_filters, (3,3), padding='same', activation='elu')(i)

	pool1 = MaxPooling2D((2,2))(convI)
	fire1_1 = addFireModule(pool1)
	fire1_2 = addFireModule(fire1_1)

	pool2 = MaxPooling2D((2,2))(fire1_2)
	fire2_1 = addFireModule(pool2)
	fire2_2 = addFireModule(fire2_1)

	pool3 = MaxPooling2D((2,2))(fire2_2)
	fire3_1 = addFireModule(pool3)
	fire3_2 = addFireModule(fire3_1)
	fire3_3 = addFireModule(fire3_2)
	fire3_4 = addFireModule(fire3_3)

	pdc = addParallelDilatedConvolution(fire3_4)

	trans_conv1 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu', strides=2)(pdc)
	ref1 = addBypassRefinementModule(trans_conv1, fire2_2)

	trans_conv2 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu', strides=2)(ref1)
	ref2 = addBypassRefinementModule(trans_conv2, fire1_2)

	trans_conv3 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu', strides=2)(ref2)
	ref3 = addBypassRefinementModule(trans_conv3, convI)

	trans_conv4 = Conv2DTranspose(11, (3,3), padding='same', activation='softmax')(ref3)

	model = Model(inputs=i, outputs=trans_conv4)

	return model

def zip3(*iterables):
    # zip('ABCD', 'xy') --> Ax By
    sentinel = object()
    iterators = [iter(it) for it in iterables]
    while iterators:
        result = []
        for it in iterators:
            elem = next(it, sentinel)
            if elem is sentinel:
                return
            result.append(elem)
        yield tuple(result)

train_image_datagen = ImageDataGenerator(rotation_range=30.,
                    						width_shift_range=0.2,
                    						height_shift_range=0.2,
                    						zoom_range=0.2,
             								fill_mode='constant',
                    						horizontal_flip=True,
                    						rescale=1./255)
train_mask_datagen = ImageDataGenerator(rotation_range=30.,
                    						width_shift_range=0.2,
                    						height_shift_range=0.2,
                    						zoom_range=0.2,
             								fill_mode='constant',
                    						horizontal_flip=True)

train_image_generator = train_image_datagen.flow_from_directory('data/training/images/',
            													target_size=img_size,
            													batch_size=batch_size,
													            class_mode=None,
													            seed=seed)
train_mask_generator = train_mask_datagen.flow_from_directory('data/training/masks/',
            													target_size=img_size,
            													color_mode='grayscale',
            													batch_size=batch_size,
													            class_mode=None,
													            seed=seed)

train_generator = zip3(train_image_generator, train_mask_generator)

val_image_datagen = ImageDataGenerator(rescale=1./255)
val_mask_datagen = ImageDataGenerator()

val_image_generator = val_image_datagen.flow_from_directory('data/validation/images/',
            													target_size=img_size,
            													batch_size=batch_size,
													            class_mode=None,
													            seed=seed)

val_mask_generator = val_mask_datagen.flow_from_directory('data/validation/masks/',
            													target_size=img_size,
            													color_mode='grayscale',
            													batch_size=batch_size,
													            class_mode=None,
													            seed=seed)

val_generator = zip3(val_image_generator, val_mask_generator)

checkpoint = ModelCheckpoint(
        model_name,
        monitor='val_loss',
        verbose=0,
        save_best_only=True)

tb = TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        write_graph=True,
        write_images=True)

early = EarlyStopping(patience=batch_size, verbose=1)

model = getModel()
model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])

'''model.fit_generator(
	train_generator,
	steps_per_epoch=steps_per_epoch,
	epochs=epochs,
	callbacks=[checkpoint, tb, early],
	validation_data=val_generator,
	validation_steps=validation_steps)'''

for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in train_generator:
        one_hot = K.one_hot(y_batch, 11)
        y_batch = K.eval(K.squeeze(one_hot, 3))
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= 600 / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break