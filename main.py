import keras
from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras import backend as K
import numpy as np

num_filters = 64
input_size = #TODO

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
	i = Input(input_size)
	convI = conv2D(num_filters, (3,3), padding='same', activation='elu')(i)

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

	trans_conv1 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu')(pdc)
	ref1 = addBypassRefinementModule(trans_conv1, fire2_2)

	trans_conv2 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu')(ref1)
	ref2 = addBypassRefinementModule(trans_conv2, fire1_2)

	trans_conv3 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu')(ref2)
	ref3 = addBypassRefinementModule(trans_conv3, convI)

	trans_conv4 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu')(ref3)

	model = Model(inputs=i, outputs=trans_conv4)

	return model

