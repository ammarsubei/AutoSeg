import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.engine.topology import Layer
from keras import backend as K
import numpy as np
import sys
import squeezebuild

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

model = squeezebuild.getModel(input_shape, num_filters)
model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])

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
            													batch_size=50,
													            class_mode=None,
													            seed=seed)
train_mask_generator = train_mask_datagen.flow_from_directory('data/training/masks/',
            													target_size=img_size,
            													color_mode='grayscale',
            													batch_size=50,
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

#sample_img = mpimg.imread('data/validation/images/imgs/0016E5_08151.png')


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
        model.fit(x_batch, y_batch, batch_size=batch_size)
        #pred = model.predict(sample_img)
        print(pred.shape)
        pred_img = np.zeros((img_height,img_width))
        batches += 1
        if batches >= 600 / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
