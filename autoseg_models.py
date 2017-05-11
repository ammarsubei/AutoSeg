from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, concatenate, add

def addFireModule(x, num_filters):
	squeeze = Conv2D(num_filters, (3,3), padding='same', activation='elu')(x)
	expand1 = Conv2D(num_filters, (1,1), padding='same', activation='elu')(squeeze)
	expand3 = Conv2D(num_filters, (3,3), padding='same', activation='elu')(squeeze)
	c = concatenate([expand1, expand3])

	return c

def addParallelDilatedConvolution(x, num_filters):
	conv1 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=1)(x)
	conv2 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=2)(x)
	conv4 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=4)(x)
	conv8 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=8)(x)
	a = add([conv1, conv2, conv4, conv8])

	return a

def addBypassRefinementModule(high, low, num_filters):
	preConv = Conv2D(num_filters, (3,3), padding='same', activation='elu')(low)
	c = concatenate([preConv, high])
	postConv = Conv2D(num_filters, (3,3), padding='same', activation='elu')(c)

	return postConv

def getModel(input_shape, num_classes, num_filters):
	i = Input(input_shape)
	convI = Conv2D(num_filters, (3,3), padding='same', activation='elu')(i)

	pool1 = MaxPooling2D((2,2))(convI)
	fire1_1 = addFireModule(pool1, num_filters)
	fire1_2 = addFireModule(fire1_1, num_filters)

	pool2 = MaxPooling2D((2,2))(fire1_2)
	fire2_1 = addFireModule(pool2, num_filters)
	fire2_2 = addFireModule(fire2_1, num_filters)

	pool3 = MaxPooling2D((2,2))(fire2_2)
	fire3_1 = addFireModule(pool3, num_filters)
	fire3_2 = addFireModule(fire3_1, num_filters)
	fire3_3 = addFireModule(fire3_2, num_filters)
	fire3_4 = addFireModule(fire3_3, num_filters)

	# Idea: change num_filters to num_classes below this point.
	pdc = addParallelDilatedConvolution(fire3_4, num_filters)

	trans_conv1 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu', strides=2)(pdc)
	ref1 = addBypassRefinementModule(trans_conv1, fire2_2, num_filters)

	trans_conv2 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu', strides=2)(ref1)
	ref2 = addBypassRefinementModule(trans_conv2, fire1_2, num_filters)

	trans_conv3 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu', strides=2)(ref2)
	ref3 = addBypassRefinementModule(trans_conv3, convI, num_filters)

	trans_conv4 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu')(ref3)

	prediction = Conv2D(num_classes, (1,1), padding='same', activation='softmax')(trans_conv4)

	model = Model(inputs=i, outputs=prediction)

	return model

def getPCModel(input_shape, num_classes, num_filters):
	i = Input(input_shape)
	convI = Conv2D(num_filters, (3,3), padding='same', activation='elu')(i)

	pool1 = MaxPooling2D((2,2))(convI)
	fire1_1 = addFireModule(pool1, num_filters)
	fire1_2 = addFireModule(fire1_1, num_filters)

	pool2 = MaxPooling2D((2,2))(fire1_2)
	fire2_1 = addFireModule(pool2, num_filters)
	fire2_2 = addFireModule(fire2_1, num_filters)

	pool3 = MaxPooling2D((2,2))(fire2_2)
	fire3_1 = addFireModule(pool3, num_filters)
	fire3_2 = addFireModule(fire3_1, num_filters)
	fire3_3 = addFireModule(fire3_2, num_filters)
	fire3_4 = addFireModule(fire3_3, num_filters)

	# Idea: change num_filters to num_classes below this point.
	pdc = addParallelDilatedConvolution(fire3_4, num_filters)

	trans_conv1 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu', strides=2)(pdc)
	ref1 = addBypassRefinementModule(trans_conv1, fire2_2, num_filters)

	trans_conv2 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu', strides=2)(ref1)
	ref2 = addBypassRefinementModule(trans_conv2, fire1_2, num_filters)

	trans_conv3 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu', strides=2)(ref2)
	ref3 = addBypassRefinementModule(trans_conv3, convI, num_filters)

	trans_conv4 = Conv2DTranspose(num_filters, (3,3), padding='same', activation='elu')(ref3)

	prediction = Conv2D(num_classes, (1,1), activation='softmax')(addParallelDilatedConvolution(trans_conv4, num_classes))

	model = Model(inputs=i, outputs=prediction)

	return model